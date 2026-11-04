"""Core data types for the MemLeak benchmark.

Every downstream component — data generation, evaluation harness, judge,
metrics — depends on these types. Changes here propagate everywhere.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FactCategory(str, Enum):
    LOCATION = "location"
    PROFESSION = "profession"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    POSSESSION = "possession"


class VisualGrounding(str, Enum):
    EXPLICIT = "explicit"   # fact directly visible in image (text, label, landmark name)
    IMPLICIT = "implicit"   # fact inferrable from visual context (architecture, equipment)
    ABSENT = "absent"       # no visual evidence; text-only


class ProbeType(str, Enum):
    TEXT_RETRACT_IMAGE_PROBE = "text_retract_image_probe"          # P1
    IMAGE_RETRACT_TEXT_PROBE = "image_retract_text_probe"          # P2
    INVISIBLE_PERSISTENCE = "invisible_persistence"                 # P3
    CROSS_IMAGE_RECONSTRUCTION = "cross_image_reconstruction"       # P4
    BEHAVIORAL_RESIDUE = "behavioral_residue"                       # P5


class LeakageVerdict(str, Enum):
    LEAKED = "leaked"
    NOT_LEAKED = "not_leaked"
    AMBIGUOUS = "ambiguous"


class RetentionVerdict(str, Enum):
    RETAINED = "retained"
    NOT_RETAINED = "not_retained"
    AMBIGUOUS = "ambiguous"


# ---------------------------------------------------------------------------
# IPG deletion affordance (for annotation, not enforcement)
# ---------------------------------------------------------------------------

class DeletionAffordance(str, Enum):
    ADDRESSABLE = "addressable"  # directly deletable by ID/query
    LINKED = "linked"            # deletable only if provenance link exists
    UNREACHABLE = "unreachable"  # cannot be targeted by storage-level deletion


# ---------------------------------------------------------------------------
# Data components
# ---------------------------------------------------------------------------

class ImageEntry(BaseModel):
    path: str
    grounding_type: VisualGrounding
    grounding_annotation: str = ""  # what in the image encodes the fact
    vlm_verified: bool = False      # has VLM confirmed inference?


class EntangledFact(BaseModel):
    fact_id: str
    similarity_level: float  # 0.5, 0.7, or 0.9
    similarity_description: str = ""


class ChainHop(BaseModel):
    fact_id: str
    modality: str  # "text" or "image"


class MultiHopChain(BaseModel):
    chain_id: str
    hops: list[ChainHop]
    answer_if_remembered: str
    answer_if_forgotten: str  # must differ from above


class Probe(BaseModel):
    probe_id: str
    probe_type: ProbeType
    query: str
    query_images: list[str] = Field(default_factory=list)
    ground_truth_if_forgotten: str = ""
    leakage_indicators: list[str] = Field(default_factory=list)
    is_direct: bool = True  # direct vs indirect probe
    # Probe 3 specific: VLM verification results
    vlm_verification_passed: Optional[bool] = None
    vlm_stage_b_outputs: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    fact_id: str
    text: str                          # e.g. "Alex is a chef in Tokyo"
    category: FactCategory
    visual_grounding: VisualGrounding
    images: list[ImageEntry] = Field(default_factory=list)
    forget_target: bool = False
    entangled_facts: list[EntangledFact] = Field(default_factory=list)
    cross_modal_chains: list[MultiHopChain] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)


class Turn(BaseModel):
    turn_id: int
    role: str  # "user" or "assistant"
    text: str
    images: list[str] = Field(default_factory=list)
    facts_introduced: list[str] = Field(default_factory=list)


class RetractionPoint(BaseModel):
    turn_id: int
    instruction: str              # e.g. "Please forget where I live"
    target_facts: list[str]       # fact_ids to forget
    variant: str = "explicit"     # explicit, conversational, targeted


class ProbeScheduleEntry(BaseModel):
    turn_id: int
    probe_ref: str       # fact_id
    probe_type: ProbeType
    intervening_turns: int = 0  # unrelated turns since retraction


class Profile(BaseModel):
    profile_id: str
    name: str
    facts: list[Fact] = Field(default_factory=list)
    interaction_stream: list[Turn] = Field(default_factory=list)
    retraction_point: Optional[RetractionPoint] = None
    probe_schedule: list[ProbeScheduleEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Experiment result types
# ---------------------------------------------------------------------------

class JudgeResult(BaseModel):
    probe_id: str
    verdict: LeakageVerdict
    confidence: float = 0.0
    reasoning: str = ""
    judge_model: str = ""


class RetentionResult(BaseModel):
    fact_id: str
    verdict: RetentionVerdict
    confidence: float = 0.0
    reasoning: str = ""
    judge_model: str = ""


class ProbeResult(BaseModel):
    profile_id: str
    fact_id: str
    probe_type: ProbeType
    agent_response: str
    judge_results: list[JudgeResult] = Field(default_factory=list)
    ensemble_verdict: LeakageVerdict = LeakageVerdict.AMBIGUOUS
    retract_modality: str = "text"
    probe_modality: str = "text"


class RunConfig(BaseModel):
    retraction_variant: str = "explicit"
    filler_turns: int = 5
    temporal_k: list[int] = Field(default_factory=lambda: [10, 50])
    # Profile admission threshold; zero never waives per-fact acquisition.
    min_recall_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    probe_types: list[ProbeType] = Field(
        default_factory=lambda: [
            ProbeType.TEXT_RETRACT_IMAGE_PROBE,
            ProbeType.INVISIBLE_PERSISTENCE,
            ProbeType.CROSS_IMAGE_RECONSTRUCTION,
        ]
    )


class EvaluationResult(BaseModel):
    system_name: str
    profile_id: str
    protocol_version: str = "legacy-unversioned"
    run_config: Optional[RunConfig] = None
    baseline_results: dict[str, bool] = Field(default_factory=dict)
    post_retraction_results: list[ProbeResult] = Field(default_factory=list)
    temporal_results: dict[int, list[ProbeResult]] = Field(default_factory=dict)
    retained_fact_results: list[RetentionResult] = Field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None
