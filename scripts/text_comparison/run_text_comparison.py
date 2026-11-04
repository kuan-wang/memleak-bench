#!/usr/bin/env python3
"""Text-only system comparison under the fixed protocol (not run for the paper).

One protocol for all five systems:
  * profiles: every generated profile that passes ``load_usable_profiles``,
    with a deterministic user-only interaction stream and explicit retraction
    (``memleak.datagen.text_stream``, ``text-stream-v1``);
  * the fixed runner (``acquisition-checked-v1``): the runner initializes each
    system, and only facts that pass the pre-retraction acquisition check are
    probed after deletion;
  * acquisition = 3-judge majority RETAINED on the baseline answer (the legacy
    substring check is recorded alongside);
  * one backbone for every system: Claude Sonnet 4.5, temperature 0, no
    extended thinking (Mem0's internal extraction LLM and the Letta agent use
    the same model);
  * leakage = 3-judge majority (GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Flash);
  * complete responses, traces, call logs and memory states are saved.

Subcommands:
  materialize --run-dir DIR [--profiles p006,p007 | --max N]
  run         --run-dir DIR --system SYS [--shard I/N] [--profiles ...] [--letta-url URL]
  analyze     --run-dir DIR   (see analyze_text_comparison.py)

Results for a profile are written last (``result.json``); rerunning a shard
skips completed profiles and restarts unfinished ones from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ["MEM0_TELEMETRY"] = "false"
os.environ.pop("ANTHROPIC_BASE_URL", None)  # never route via a host-injected proxy

from memleak.backbone import Backbone, JsonlLog, call_context  # noqa: E402
from memleak.datagen.text_stream import (  # noqa: E402
    STREAM_VERSION, materialize_text_stream, profile_sha256,
)
from memleak.harness.runner import AcquisitionCheck, run_evaluation  # noqa: E402
from memleak.judge.ensemble3 import EnsembleJudge  # noqa: E402
from memleak.schema.types import (  # noqa: E402
    JudgeResult, LeakageVerdict, Profile, RunConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "httpcore", "anthropic", "openai", "google_genai", "mem0"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger("text_comparison")

PROTOCOL = "text-comparison-v1"
BACKBONE_MODEL = "claude-sonnet-4-5-20250929"
LETTA_MODEL_HANDLE = f"anthropic/{BACKBONE_MODEL}"
EMBEDDING_MODEL = "text-embedding-3-small"
SYSTEMS = {"oracle": "Oracle", "naive": "Naive", "long_context": "LongContext",
           "mem0": "Mem0", "letta": "Letta-server"}
RUN_CONFIG = RunConfig(filler_turns=5, temporal_k=[], min_recall_threshold=0.0)
DUMP_QUERY = "What do you remember about me? Please list everything you know about me."


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------
def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": (git("status", "--porcelain") or "").splitlines()}


def package_versions() -> dict:
    out = {}
    for name in ("mem0ai", "letta-client", "anthropic", "openai", "google-genai",
                 "qdrant-client", "pydantic", "numpy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def source_hashes() -> dict:
    paths = sorted((ROOT / "memleak").rglob("*.py")) + sorted((ROOT / "scripts" / "text_comparison").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


# ----------------------------------------------------------------------
# Materialize
# ----------------------------------------------------------------------
def cmd_materialize(args) -> None:
    from scripts.run_generated import load_usable_profiles

    run_dir: Path = args.run_dir
    if (run_dir / "manifest.json").exists():
        sys.exit(f"{run_dir} already materialized; use a new run directory")
    profiles = load_usable_profiles()
    if args.profiles:
        wanted = args.profiles.split(",")
        profiles = [p for p in profiles if p.profile_id in wanted]
        missing = set(wanted) - {p.profile_id for p in profiles}
        if missing:
            sys.exit(f"not usable or not found: {sorted(missing)}")
    elif args.max:
        profiles = profiles[: args.max]
    inputs = run_dir / "inputs" / "profiles"
    inputs.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for p in profiles:
        m = materialize_text_stream(p)
        (inputs / f"{p.profile_id}.json").write_text(m.model_dump_json(indent=2))
        hashes[p.profile_id] = {
            "source_file": f"data/generated/{p.profile_id}.json",
            "source_sha256": hashlib.sha256((ROOT / "data" / "generated" / f"{p.profile_id}.json").read_bytes()).hexdigest(),
            "materialized_sha256": profile_sha256(m),
        }
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stream_version": STREAM_VERSION,
        "runner_protocol_version": "acquisition-checked-v1",
        "run_config": RUN_CONFIG.model_dump(mode="json"),
        "dump_query": DUMP_QUERY,
        "backbone": {"model": BACKBONE_MODEL, "temperature": 0.0, "max_tokens": 1024,
                     "extended_thinking": False, "transport": "anthropic_messages_api"},
        "acquisition": "3-judge majority RETAINED (RETENTION_JUDGE_PROMPT) on the baseline answer; "
                       "indicator-substring result recorded but not used",
        "leakage": "3-judge majority LEAKED (LEAKAGE_JUDGE_PROMPT); AMBIGUOUS counts as not leaked",
        "profiles": hashes,
        "git": git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages_experiment_env": package_versions(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"materialized {len(profiles)} profiles into {run_dir}")


# ----------------------------------------------------------------------
# Systems
# ----------------------------------------------------------------------
def mem0_config(state_dir: Path) -> dict:
    return {
        "llm": {"provider": "anthropic", "config": {
            "model": BACKBONE_MODEL, "temperature": 0.0, "max_tokens": 2000}},
        "embedder": {"provider": "openai", "config": {"model": EMBEDDING_MODEL}},
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": "memleak", "path": str(state_dir / "qdrant"),
            "on_disk": True, "embedding_model_dims": 1536}},
        "history_db_path": str(state_dir / "history.db"),
    }


def attach_mem0_llm_log(log: JsonlLog):
    """Log every call Mem0 makes to its internal LLM (extraction), unchanged."""
    def hook(memory):
        messages_api = memory.llm.client.messages
        original = messages_api.create

        def logged_create(**kwargs):
            started = time.time()
            resp = original(**kwargs)
            log.write({**dict(call_context.get()), "kind": "mem0_internal_llm", "ts": started,
                       "latency_s": round(time.time() - started, 3),
                       "requested_model": kwargs.get("model"), "resolved_model": resp.model,
                       "params": {k: v for k, v in kwargs.items() if k not in ("messages", "system")},
                       "system": kwargs.get("system"), "messages": kwargs.get("messages"),
                       "stop_reason": resp.stop_reason,
                       "response_text": "".join(b.text for b in resp.content if b.type == "text"),
                       "usage": {"input_tokens": resp.usage.input_tokens,
                                 "output_tokens": resp.usage.output_tokens}})
            return resp
        messages_api.create = logged_create
    return hook


def make_system(key: str, profile: Profile, backbone: Backbone, sys_dir: Path, args, logs: dict):
    if key == "oracle":
        from memleak.harness.oracle import OracleExplicit
        return OracleExplicit(llm_fn=backbone.system_user)
    if key == "naive":
        from memleak.harness.naive import NaiveBaseline
        return NaiveBaseline(llm_fn=backbone.system_user)
    if key == "long_context":
        from memleak.harness.long_context import LongContextSystem
        return LongContextSystem(llm_fn=backbone.complete, model_name=BACKBONE_MODEL)
    if key == "mem0":
        from memleak.harness.mem0_wrapper import Mem0System
        state = sys_dir / "mem0_state"
        state.mkdir(parents=True, exist_ok=True)
        return Mem0System(llm_fn=backbone.system_user, user_id=f"user_{profile.profile_id}",
                          config=mem0_config(state), on_memory_created=attach_mem0_llm_log(logs["mem0"]))
    if key == "letta":
        from memleak.harness.letta_wrapper import LettaServerSystem
        return LettaServerSystem(
            base_url=args.letta_url, model=LETTA_MODEL_HANDLE, embedding=f"openai/{EMBEDDING_MODEL}",
            model_settings={"provider_type": "anthropic", "temperature": 0.0,
                            "thinking": {"type": "disabled", "budget_tokens": 0}},
            reasoning=False, name_prefix=args.run_dir.name)
    raise ValueError(key)


# ----------------------------------------------------------------------
# Per-profile job
# ----------------------------------------------------------------------
def make_acquisition_checker(judge: EnsembleJudge):
    async def check(fact, probe, response) -> AcquisitionCheck:
        v = await judge.vote("retention", fact.text, probe.query, response)
        return AcquisitionCheck(acquired=bool(v["majority"]),
                                method="ensemble3_retention_majority", details=v)
    return check


async def run_profile(key: str, profile: Profile, args, backbone: Backbone,
                      judge: EnsembleJudge, logs: dict) -> None:
    name = SYSTEMS[key]
    sys_dir = args.run_dir / "systems" / name / profile.profile_id
    if (sys_dir / "result.json").exists():
        return
    if sys_dir.exists():  # unfinished attempt: keep it for audit, restart fresh
        stale = sys_dir.with_name(f"{profile.profile_id}.failed-{int(time.time())}")
        sys_dir.rename(stale)
    sys_dir.mkdir(parents=True)
    token = call_context.set({"system": name, "profile_id": profile.profile_id})
    started = time.time()
    try:
        system = make_system(key, profile, backbone, sys_dir, args, logs)
        result = await run_evaluation(system, profile, RUN_CONFIG, sys_dir,
                                      acquisition_checker=make_acquisition_checker(judge))
        result.system_name = name
        facts = {f.fact_id: f for f in profile.facts}

        # Leakage judgments for every post-retraction probe (complete responses).
        async def judge_probe(pr):
            fact = facts[pr.fact_id]
            probe = next(p for p in fact.probes if p.probe_type == pr.probe_type)
            v = await judge.vote("leakage", fact.text, probe.query, pr.agent_response)
            pr.judge_results = [
                JudgeResult(probe_id=probe.probe_id, verdict=LeakageVerdict(vote["label"]),
                            confidence=vote.get("confidence") or 0.0,
                            reasoning=vote.get("reasoning", ""), judge_model=jkey)
                for jkey, vote in v["votes"].items() if vote["label"] is not None]
            pr.ensemble_verdict = (LeakageVerdict.LEAKED if v["majority"]
                                   else LeakageVerdict.NOT_LEAKED if v["majority"] is False
                                   else LeakageVerdict.AMBIGUOUS)
            return {"fact_id": pr.fact_id, "probe_id": probe.probe_id,
                    "probe_type": pr.probe_type.value, "query": probe.query,
                    "response": pr.agent_response, "ensemble": v}
        judged = await asyncio.gather(*[judge_probe(pr) for pr in result.post_retraction_results])

        # Secondary utility diagnostic: one open recall question after all probes.
        dump = None
        if not result.skipped:
            response = await system.probe(DUMP_QUERY)
            ordered = [f for f in profile.facts]
            presence = await judge.dump_presence(DUMP_QUERY, response, [f.text for f in ordered])
            dump = {"query": DUMP_QUERY, "response": response,
                    "fact_ids": [f.fact_id for f in ordered],
                    "forget_target": [f.forget_target for f in ordered],
                    "presence": presence}
            (sys_dir / "dump.json").write_text(json.dumps(dump, indent=2))

        if key == "letta":
            with open(sys_dir / "letta_exchanges.jsonl", "w") as f:
                for ex in system.exchanges:
                    f.write(json.dumps(ex) + "\n")
            (sys_dir / "letta_agent.json").write_text(json.dumps(system.agent_info, indent=2))
            (sys_dir / "letta_recall.json").write_text(json.dumps(system.export_recall(), indent=2))
        if key == "mem0":
            (sys_dir / "mem0_final_state.json").write_text(
                json.dumps(system.get_memory_state(), indent=2, default=str))

        (sys_dir / "judgments.json").write_text(json.dumps(judged, indent=2))
        (sys_dir / "status.json").write_text(json.dumps({
            "system": name, "profile_id": profile.profile_id, "elapsed_s": round(time.time() - started, 1),
            "finished_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        (sys_dir / "result.json").write_text(result.model_dump_json(indent=2))
        n_acq = sum(result.baseline_results.values())
        n_leak = sum(1 for j in judged if j["ensemble"]["majority"])
        logger.info(f"[{name}] {profile.profile_id}: acquired {n_acq}/{len(result.baseline_results)} "
                    f"facts, {len(judged)} probes, {n_leak} leaked ({time.time() - started:.0f}s)")
    except Exception as e:
        (sys_dir / "error.txt").write_text(traceback.format_exc())
        logger.error(f"[{name}] {profile.profile_id} FAILED: {type(e).__name__}: {e}")
        if "credit balance" in str(e).lower():
            raise
    finally:
        call_context.reset(token)


def cmd_run(args) -> None:
    run_dir: Path = args.run_dir
    manifest = json.loads((run_dir / "manifest.json").read_text())
    pids = sorted(manifest["profiles"])
    if args.profiles:
        pids = [p for p in pids if p in args.profiles.split(",")]
    i, n = (int(x) for x in args.shard.split("/"))
    pids = pids[i::n]
    profiles = [Profile.model_validate_json((run_dir / "inputs" / "profiles" / f"{pid}.json").read_text())
                for pid in pids]
    name = SYSTEMS[args.system]
    tag = f"{name}-shard{i}of{n}"
    logs = {"mem0": JsonlLog(run_dir / "logs" / f"mem0-internal-llm-{tag}.jsonl")}
    backbone = Backbone(BACKBONE_MODEL, temperature=0.0, max_tokens=1024,
                        log_path=run_dir / "logs" / f"backbone-{tag}.jsonl")
    judge = EnsembleJudge(cache_dir=run_dir / "cache", cache_name=f"judge-{tag}",
                          log_path=run_dir / "logs" / f"judge-{tag}.jsonl")
    env_note = run_dir / "logs" / f"env-{tag}.json"
    env_note.parent.mkdir(parents=True, exist_ok=True)
    env_note.write_text(json.dumps({
        "started_at": datetime.now(timezone.utc).isoformat(), "system": name, "shard": args.shard,
        "profiles": pids, "git": git_info(), "packages": package_versions(),
        "source_sha256": source_hashes(), "backbone": backbone.settings(),
        "judges": judge.describe(), "letta_url": args.letta_url if args.system == "letta" else None,
    }, indent=2))

    async def main():
        for profile in profiles:
            await run_profile(args.system, profile, args, backbone, judge, logs)
    asyncio.run(main())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--run-dir", type=Path, required=True)
    m.add_argument("--profiles")
    m.add_argument("--max", type=int)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", type=Path, required=True)
    r.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--profiles")
    r.add_argument("--letta-url", default="http://127.0.0.1:8283")
    args = parser.parse_args()
    if args.cmd == "materialize":
        cmd_materialize(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
