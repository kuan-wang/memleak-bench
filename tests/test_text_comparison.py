"""Text-comparison protocol: stream materialization, semantic acquisition,
real-Letta wrapper bookkeeping, judge parsing and statistics. No API calls."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from memleak.datagen.text_stream import (
    FACT_TEMPLATE, STREAM_VERSION, fact_order, materialize_text_stream,
)
from memleak.harness.letta_wrapper import LettaServerSystem, _assistant_text
from memleak.harness.oracle import OracleExplicit
from memleak.harness.runner import AcquisitionCheck, run_evaluation
from memleak.judge.ensemble3 import parse_present, parse_verdict
from memleak.schema.types import (
    Fact, FactCategory, Probe, ProbeType, Profile, RunConfig, VisualGrounding,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "text_comparison"))
import analyze_text_comparison as analysis  # noqa: E402


def generated_profile() -> Profile:
    """Shape of data/generated/*.json: facts and probes, no stream, no retraction."""
    facts = []
    for cat in FactCategory:
        for i, target in ((1, False), (2, True)):
            fid = f"g_{cat.value}_{i}"
            facts.append(Fact(
                fact_id=fid, text=f"He has {cat.value} detail {i}.", category=cat,
                visual_grounding=VisualGrounding.ABSENT, forget_target=target,
                probes=[Probe(probe_id=f"{fid}_p", probe_type=ProbeType.INVISIBLE_PERSISTENCE,
                              query=f"What {cat.value} detail does he have?",
                              leakage_indicators=[cat.value])] if target else [],
            ))
    return Profile(profile_id="g001", name="Test Person", facts=facts)


def test_stream_injects_every_fact_and_retracts_every_target():
    raw = generated_profile()
    assert not raw.interaction_stream and raw.retraction_point is None
    p = materialize_text_stream(raw)
    rp = p.retraction_point
    assert rp is not None and rp.variant == "explicit"
    assert rp.target_facts == [f.fact_id for f in raw.facts if f.forget_target]
    injected = [t for t in p.interaction_stream if t.turn_id < rp.turn_id]
    assert all(t.role == "user" for t in p.interaction_stream)
    assert sorted(fid for t in injected for fid in t.facts_introduced) == sorted(f.fact_id for f in raw.facts)
    for fact in raw.facts:
        turn = next(t for t in injected if fact.fact_id in t.facts_introduced)
        assert turn.text == FACT_TEMPLATE.format(text=fact.text)
        assert (fact.text in rp.instruction) == fact.forget_target
    # category-interleaved, deterministic
    assert fact_order(raw)[:5] == [f"g_{c.value}_1" for c in FactCategory]
    assert materialize_text_stream(raw) == p
    assert STREAM_VERSION == "text-stream-v1"
    with pytest.raises(ValueError):
        materialize_text_stream(p)


@pytest.mark.asyncio
async def test_semantic_checker_overrides_substring_and_is_recorded(tmp_path):
    profile = materialize_text_stream(generated_profile())

    async def backend(context, query, images):
        # Echoes the query, so the legacy substring check (indicator = category
        # word, which appears in the query) passes for every fact.
        return f"I don't know which {query.split()[1]} detail he has."

    async def checker(fact, probe, response):
        return AcquisitionCheck(acquired=fact.category == FactCategory.LOCATION,
                                method="test_semantic", details={"fact": fact.fact_id})

    result = await run_evaluation(
        OracleExplicit(llm_fn=backend), profile,
        RunConfig(filler_turns=0, temporal_k=[], min_recall_threshold=0), tmp_path,
        acquisition_checker=checker,
    )
    assert result.baseline_results == {f.fact_id: f.category == FactCategory.LOCATION
                                       for f in profile.facts if f.forget_target}
    assert [p.fact_id for p in result.post_retraction_results] == ["g_location_2"]
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    baseline = [r for r in trace if r["phase"] == "baseline"]
    assert all(r["substring_recalled"] for r in baseline)
    assert {r["acquisition_method"] for r in baseline} == {"test_semantic"}
    states = [r["label"] for r in trace if r["phase"] == "memory_state"]
    assert states == ["after_injection", "after_retraction", "end"]


class FakeLetta:
    """Minimal stand-in for letta_client.Letta with a memory-editing agent."""

    def __init__(self):
        self.block = ""
        self.history = []
        outer = self

        def model(**kw):
            return SimpleNamespace(model_dump=lambda mode="json": kw, **kw)

        class Messages:
            def create(self, agent_id, messages):
                text = messages[0]["content"]
                outer.history.append(text)
                out = []
                if text.startswith("Please remember this about me: "):
                    fact = text.split(": ", 1)[1]
                    outer.block += fact + "\n"
                    out.append(model(message_type="tool_call_message",
                                     tool_call={"name": "memory_insert", "arguments": fact}))
                    reply = "Noted."
                elif text.startswith("Please forget"):
                    for line in text.splitlines()[1:]:
                        outer.block = outer.block.replace(line[2:] + "\n", "")
                        out.append(model(message_type="tool_call_message",
                                         tool_call={"name": "memory_replace", "arguments": line[2:]}))
                    reply = "Forgotten."
                else:
                    reply = "From our chat: " + " ".join(outer.history[:-1])
                out.append(model(message_type="assistant_message", content=reply))
                return SimpleNamespace(messages=out, usage=model(prompt_tokens=10, step_count=1),
                                       stop_reason=None)

            def list(self, agent_id, limit=1000, order=None):
                return [model(message_type="user_message", content=h) for h in outer.history]

        class Blocks:
            def list(self, agent_id):
                return [SimpleNamespace(label="human", value=outer.block)]

        class Passages:
            def list(self, agent_id, limit=1000):
                return []

        class Agents:
            messages = Messages()
            blocks = Blocks()
            passages = Passages()

            def create(self, **kw):
                outer.created = kw
                return SimpleNamespace(
                    id="agent-1", name=kw["name"], agent_type=kw["agent_type"], tools=[],
                    llm_config=model(model="fake"), embedding_config=None, system="sys")

            def retrieve(self, agent_id):
                return SimpleNamespace(message_ids=list(range(len(outer.history))))

        self.agents = Agents()


@pytest.mark.asyncio
async def test_letta_wrapper_uses_agent_deletion_and_records_retained_history(tmp_path):
    profile = materialize_text_stream(generated_profile())
    fake = FakeLetta()
    system = LettaServerSystem(client=fake, name_prefix="t")

    async def checker(fact, probe, response):
        return AcquisitionCheck(acquired=fact.text in fake.block, method="block")

    result = await run_evaluation(system, profile, RunConfig(filler_turns=1, temporal_k=[],
                                  min_recall_threshold=0), tmp_path, acquisition_checker=checker)
    assert fake.created["agent_type"] == "letta_v1_agent" and fake.created["reasoning"] is False
    assert all(result.baseline_results.values())
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    ret = next(r for r in trace if r["phase"] == "retraction")["result"]
    assert ret["method"] == "letta_agent_native_memory_tools"
    targets = [f.text for f in profile.facts if f.forget_target]
    assert sorted(i.split(":", 2)[2] for i in ret["deleted_items"]) == sorted(targets)
    assert all(t not in ret["metadata"]["blocks_after"]["human"] for t in targets)
    assert {c["name"] for c in ret["metadata"]["tool_calls"]} == {"memory_replace"}
    # The harness deletes nothing itself: the conversation still holds the facts.
    assert all(any(t in h for h in fake.history) for t in targets)
    kinds = [e["kind"] for e in system.exchanges]
    assert kinds.count("forget") == 1 and kinds[0] == "inject"
    assert _assistant_text([{"message_type": "assistant_message", "content": [{"text": "a"}]},
                            {"message_type": "tool_call_message"}]) == "a"


def test_judge_parsing_is_strict_about_labels():
    assert parse_verdict('```json\n{"verdict": "LEAKED", "confidence": 0.9}\n```', "leakage")[0] == "leaked"
    assert parse_verdict('Sure: {"verdict": "NOT LEAKED"}', "leakage")[0] == "not_leaked"
    label, _, _, ok = parse_verdict("no json here", "leakage")
    assert label == "ambiguous" and not ok
    assert parse_verdict('{"verdict": "RETAINED"}', "leakage")[0] == "ambiguous"
    assert parse_verdict('{"verdict": "RETAINED"}', "retention")[0] == "retained"
    assert parse_present('{"present": [1, 3, 99]}', 3) == ([1, 3], True)
    assert parse_present("nothing", 3) == (None, False)


def test_statistics():
    lo, hi = analysis.wilson(4, 1053)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (0.1, 1.0)  # published Oracle CI
    lo, hi = analysis.wilson(179, 1053)
    assert (round(100 * lo, 1), round(100 * hi, 1)) == (14.9, 19.4)  # published Long-context CI
    assert analysis.cluster_bootstrap([(1, 10)] * 20) == (0.1, 0.1)
    lo, hi = analysis.cluster_bootstrap([(0, 10)] * 10 + [(10, 10)] * 10)
    assert lo < 0.5 < hi
    assert analysis.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert analysis.fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == 1.0
