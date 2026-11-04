"""Evaluation runner: orchestrates the 6-phase evaluation loop.

Phase 1 — INJECTION: Feed interaction stream to the system.
Phase 2 — BASELINE VERIFICATION (mandatory): Confirm the system recalls facts.
Phase 3 — RETRACTION: Issue forget instruction.
Phase 4 — FILLER: Inject unrelated turns.
Phase 5 — POST-RETRACTION PROBING: Cross-modal and same-modal probes.
Phase 6 — TEMPORAL PROBING: Re-probe after K unrelated turns.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from memleak.harness.base import AgentMemorySystem
from memleak.schema.types import (
    EvaluationResult,
    LeakageVerdict,
    Probe,
    ProbeResult,
    Profile,
    RetentionResult,
    RetentionVerdict,
    RunConfig,
    Turn,
)

logger = logging.getLogger(__name__)


FILLER_TOPICS = [
    "What's the weather like this week?",
    "Any interesting technology news recently?",
    "Know a simple recipe I could try tonight?",
    "Can you recommend a good podcast or audiobook?",
    "Help me schedule something for later this week.",
]


@dataclass
class AcquisitionCheck:
    """Outcome of the pre-retraction acquisition check for one fact."""
    acquired: bool
    method: str
    details: dict[str, Any] = field(default_factory=dict)


# (fact, baseline probe, baseline response) -> AcquisitionCheck
AcquisitionChecker = Callable[[Any, Probe, str], Awaitable[AcquisitionCheck]]


def substring_acquired(probe: Probe, response: str) -> bool:
    """Legacy check: any non-empty leakage indicator occurs in the response."""
    indicators = [i.strip() for i in probe.leakage_indicators if i.strip()]
    return any(indicator.lower() in response.lower() for indicator in indicators)


async def run_evaluation(
    system: AgentMemorySystem,
    profile: Profile,
    config: RunConfig | None = None,
    output_dir: Path | None = None,
    acquisition_checker: AcquisitionChecker | None = None,
) -> EvaluationResult:
    """Run the full 6-phase evaluation loop on one profile.

    Args:
        system: The agent memory system to evaluate.
        profile: The profile containing facts, stream, retraction, and probes.
        config: Run configuration (defaults if None).
        output_dir: If provided, save per-phase checkpoints and trace here.
        acquisition_checker: Optional semantic acquisition check. When given,
            it alone decides acquisition; the legacy substring result is still
            recorded in the trace. When None, the substring check decides.

    Returns:
        EvaluationResult with baseline, post-retraction, temporal, and
        retention results.
    """
    config = config or RunConfig()
    result = EvaluationResult(
        system_name=type(system).__name__,
        profile_id=profile.profile_id,
        protocol_version="acquisition-checked-v1",
        run_config=config.model_copy(deep=True),
    )
    trace: list[dict[str, Any]] = []

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: INJECTION
    # ------------------------------------------------------------------
    logger.info(f"[{profile.profile_id}] Phase 1: Injection")
    system.initialize(profile)

    retraction_turn = profile.retraction_point.turn_id if profile.retraction_point else len(profile.interaction_stream)

    for turn in profile.interaction_stream:
        if turn.turn_id >= retraction_turn:
            break  # stop before retraction
        response = await system.inject_turn(turn)
        trace.append({
            "phase": "injection",
            "turn_id": turn.turn_id,
            "role": turn.role,
            "text": turn.text,
            "response": response,
        })

    _save_checkpoint(output_dir, "phase1_injection", trace)
    _record_memory_state(system, "after_injection", trace)

    # ------------------------------------------------------------------
    # Phase 2: BASELINE VERIFICATION (mandatory)
    # ------------------------------------------------------------------
    logger.info(f"[{profile.profile_id}] Phase 2: Baseline verification")
    baseline: dict[str, bool] = {}
    pending: list[tuple[Any, Probe, str, dict[str, Any]]] = []

    # Probe in profile order first (conversation order matters for stateful
    # systems), then run the acquisition checks; checks never touch the system.
    for fact in profile.facts:
        if not fact.forget_target:
            continue
        # Use the first direct probe as baseline query
        baseline_probe = _get_baseline_probe(fact)
        indicators = (
            [indicator.strip() for indicator in baseline_probe.leakage_indicators
             if indicator.strip()]
            if baseline_probe else []
        )
        if not baseline_probe or (acquisition_checker is None and not indicators):
            logger.warning(f"  Fact {fact.fact_id}: no scorable baseline probe")
            baseline[fact.fact_id] = False
            trace.append({
                "phase": "baseline",
                "fact_id": fact.fact_id,
                "recalled": False,
                "reason": "missing_probe_or_indicators",
            })
            continue

        response = await system.probe(baseline_probe.query, baseline_probe.query_images or None)
        entry = {
            "phase": "baseline",
            "fact_id": fact.fact_id,
            "query": baseline_probe.query,
            "response": response,
            "substring_recalled": substring_acquired(baseline_probe, response),
        }
        trace.append(entry)
        pending.append((fact, baseline_probe, response, entry))

    if acquisition_checker is None:
        for fact, _, _, entry in pending:
            entry["recalled"] = entry["substring_recalled"]
            entry["acquisition_method"] = "indicator_substring"
            baseline[fact.fact_id] = entry["recalled"]
    else:
        checks = await asyncio.gather(*[
            acquisition_checker(fact, probe, response)
            for fact, probe, response, _ in pending
        ])
        for (fact, _, _, entry), check in zip(pending, checks):
            entry["recalled"] = bool(check.acquired)
            entry["acquisition_method"] = check.method
            entry["acquisition_details"] = check.details
            baseline[fact.fact_id] = bool(check.acquired)

    result.baseline_results = baseline

    # Check recall threshold
    recall_rate = sum(baseline.values()) / max(len(baseline), 1)
    if not any(baseline.values()) or recall_rate < config.min_recall_threshold:
        logger.warning(
            f"[{profile.profile_id}] Acquisition gate failed: recall {recall_rate:.1%}, "
            f"threshold {config.min_recall_threshold:.1%} — skipping profile"
        )
        result.skipped = True
        result.skip_reason = f"baseline_recall={recall_rate:.2f}"
        _save_checkpoint(output_dir, "phase2_baseline", trace)
        _save_trace(output_dir, trace)
        return result

    _save_checkpoint(output_dir, "phase2_baseline", trace)

    # ------------------------------------------------------------------
    # Phase 3: RETRACTION
    # ------------------------------------------------------------------
    logger.info(f"[{profile.profile_id}] Phase 3: Retraction")

    if profile.retraction_point:
        forget_result = await system.forget(profile.retraction_point.instruction)
        trace.append({
            "phase": "retraction",
            "instruction": profile.retraction_point.instruction,
            "result": {
                "success": forget_result.success,
                "deleted_items": forget_result.deleted_items,
                "method": forget_result.method,
                "metadata": forget_result.metadata,
            },
        })

    _save_checkpoint(output_dir, "phase3_retraction", trace)
    _record_memory_state(system, "after_retraction", trace)

    # ------------------------------------------------------------------
    # Phase 4: FILLER
    # ------------------------------------------------------------------
    logger.info(f"[{profile.profile_id}] Phase 4: Filler ({config.filler_turns} turns)")

    for i in range(config.filler_turns):
        topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
        filler_turn = Turn(
            turn_id=retraction_turn + 100 + i,  # offset to avoid ID collision
            role="user",
            text=topic,
        )
        response = await system.inject_turn(filler_turn)
        trace.append({
            "phase": "filler",
            "turn_id": filler_turn.turn_id,
            "text": topic,
            "response": response,
        })

    _save_checkpoint(output_dir, "phase4_filler", trace)

    # ------------------------------------------------------------------
    # Phase 5: POST-RETRACTION PROBING
    # ------------------------------------------------------------------
    logger.info(f"[{profile.profile_id}] Phase 5: Post-retraction probing")

    for fact in profile.facts:
        if not fact.forget_target:
            continue
        if not baseline.get(fact.fact_id, False):
            continue  # skip facts the system never recalled

        for probe in fact.probes:
            if probe.probe_type not in config.probe_types:
                continue

            response = await system.probe(probe.query, probe.query_images or None)

            probe_result = ProbeResult(
                profile_id=profile.profile_id,
                fact_id=fact.fact_id,
                probe_type=probe.probe_type,
                agent_response=response,
                retract_modality="text",  # default; override per probe type if needed
                probe_modality=_probe_modality(probe),
            )
            result.post_retraction_results.append(probe_result)

            trace.append({
                "phase": "post_retraction",
                "fact_id": fact.fact_id,
                "probe_type": probe.probe_type.value,
                "query": probe.query,
                "response": response,
            })

    _save_checkpoint(output_dir, "phase5_probing", trace)

    # ------------------------------------------------------------------
    # Phase 6: TEMPORAL PROBING
    # ------------------------------------------------------------------
    for k in config.temporal_k:
        logger.info(f"[{profile.profile_id}] Phase 6: Temporal probing K={k}")

        # Inject K filler turns
        for i in range(k):
            topic = FILLER_TOPICS[i % len(FILLER_TOPICS)]
            filler = Turn(
                turn_id=retraction_turn + 200 + k * 100 + i,
                role="user",
                text=topic,
            )
            await system.inject_turn(filler)

        # Re-run cross-modal probes
        temporal_probes: list[ProbeResult] = []
        for fact in profile.facts:
            if not fact.forget_target:
                continue
            if not baseline.get(fact.fact_id, False):
                continue
            for probe in fact.probes:
                if probe.probe_type not in config.probe_types:
                    continue
                response = await system.probe(probe.query, probe.query_images or None)
                temporal_probes.append(ProbeResult(
                    profile_id=profile.profile_id,
                    fact_id=fact.fact_id,
                    probe_type=probe.probe_type,
                    agent_response=response,
                    retract_modality="text",
                    probe_modality=_probe_modality(probe),
                ))
                trace.append({
                    "phase": f"temporal_k{k}",
                    "fact_id": fact.fact_id,
                    "probe_type": probe.probe_type.value,
                    "query": probe.query,
                    "response": response,
                })

        result.temporal_results[k] = temporal_probes

    _record_memory_state(system, "end", trace)
    _save_checkpoint(output_dir, "phase6_temporal", trace)
    _save_trace(output_dir, trace)

    logger.info(
        f"[{profile.profile_id}] Complete: "
        f"{len(result.post_retraction_results)} probes, "
        f"{len(result.temporal_results)} temporal rounds"
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_baseline_probe(fact) -> Probe | None:
    """Get the first direct probe for baseline verification."""
    for p in fact.probes:
        if p.is_direct:
            return p
    return fact.probes[0] if fact.probes else None


def _probe_modality(probe: Probe) -> str:
    """Determine probe modality from probe type and content."""
    if probe.query_images:
        return "image"
    return "text"


def _record_memory_state(system: AgentMemorySystem, label: str, trace: list) -> None:
    """Append the system's inspectable memory state, when it exposes one."""
    try:
        state = system.get_memory_state()
    except Exception as e:  # inspection must never change the measurement
        state = {"error": f"{type(e).__name__}: {e}"}
    if state is not None:
        trace.append({"phase": "memory_state", "label": label, "state": state})


def _save_checkpoint(output_dir: Path | None, phase: str, trace: list) -> None:
    if not output_dir:
        return
    path = output_dir / f"checkpoint_{phase}.json"
    path.write_text(json.dumps({"phase": phase, "trace_len": len(trace)}, indent=2))
    _save_trace(output_dir, trace)


def _save_trace(output_dir: Path | None, trace: list) -> None:
    if not output_dir:
        return
    path = output_dir / "trace.jsonl"
    with open(path, "w") as f:
        for entry in trace:
            f.write(json.dumps(entry) + "\n")
