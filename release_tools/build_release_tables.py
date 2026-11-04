#!/usr/bin/env python3
"""Build MemLeak's flat metadata tables and Croissant file from the dataset repository.

Run from the root of the dataset repository:

    python code/release_tools/build_release_tables.py --repo-id memleak-bench/memleak-benchmark \
        [--local-croissant out.json]

Writes:
    metadata/facts.csv                 one row per profile fact (generated + pilot profiles)
    metadata/images.csv                one row per image file, with checksum, source and license
    metadata/multimodal_judgments.csv  one row per probe, with each judge's verdict and the ensemble verdict
    metadata/entailment_labels.csv     both entailment classifiers' labels for the 300 retained-image probes
    (metadata/real_photos_unsplash.csv and real_collections/*.csv are inputs: they are described in the
    Croissant file but written by the authors' staging tools, because their sources are not public.)
    croissant.json                     Croissant 1.1 metadata with Responsible-AI (RAI) and provenance fields

The script reads only files that are already in the repository and makes no network
calls. Git LFS pointer files are fine: an image's sha256 and size are read from its
pointer when the blob is not checked out. Requires Python >= 3.10 and no third-party
packages.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
from pathlib import Path

JUDGES = {"gpt41": "gpt-4.1", "claude_sonnet": None, "gemini_flash": "gemini-2.5-flash"}
SONNET4 = "claude-sonnet-4-20250514"

# Model IDs per experiment. "file" means the result file records the ID itself.
# "script" means it comes from the producer script in the authors' source repository.
EXPERIMENTS = [
    # (experiment, source file, mode key or None, record judge key, prober model, claude judge model, model id source)
    ("core_same_fact", "results/structural_floor_scale_llm_judged.json", "same_fact", "llm_judge", SONNET4, SONNET4, "script"),
    ("core_retained", "results/structural_floor_scale_llm_judged.json", "retained", "llm_judge", SONNET4, SONNET4, "script"),
    ("core_negative_control", "results/structural_floor_scale_llm_judged.json", "negative", "llm_judge", SONNET4, SONNET4, "script"),
    ("retained_gpt4o_prober", "results/structural_floor_gpt4o_retained.json", None, "llm_judge", "gpt-4o", SONNET4, "file+script"),
    ("retained_real_photos", "results/real_image_eval_retained.json", None, "judgment", SONNET4, SONNET4, "script"),
    ("retained_mem0_end_to_end", "results/mem0_multimodal_retained.json", None, "judgment", SONNET4, SONNET4, "script"),
    ("retained_after_semantic_deletion", "results/semantic_deletion_scale_n300.json", None, "post_deletion_judge", SONNET4, SONNET4, "script"),
    ("blind_inference", "results/blind_inference_baseline.json", None, "judgment", SONNET4, SONNET4,
     "authors (paper): repository default model at the time; generating script not preserved"),
    # prober per row: see RETAINED_TEXT_PROBER below
    ("retained_text_only", "results/retained_text_only_baseline.json", None, "judgment", None, "claude-sonnet-4-6", None),
    ("retained_sonnet5_rerun", "results/supplementary/structural_floor_sonnet5_retained.json", None, "llm_judge", "claude-sonnet-5", "claude-sonnet-4-5", "file"),
]


def sha256_and_size(path: Path) -> tuple[str, int]:
    """Return (sha256, size) of a file, reading the Git LFS pointer if the blob is absent."""
    with open(path, "rb") as fh:
        head = fh.read(200)
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        text = path.read_text()
        oid = re.search(r"oid sha256:([0-9a-f]{64})", text).group(1)
        size = int(re.search(r"size (\d+)", text).group(1))
        return oid, size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size


def retained_text_prober(r):
    """Retained-text baseline: 280 probes ran with the repository's default Claude model at the time
    (claude-sonnet-4-6); the 20 re-run after API failures (scripts/backfill_retained_text.py) used gpt-4.1
    and are the rows whose judgment carries a `verdicts` list."""
    return "gpt-4.1" if "verdicts" in (r.get("judgment") or {}) else "claude-sonnet-4-6"


def retained_text_id_source(r):
    if "verdicts" in (r.get("judgment") or {}):
        return "script (scripts/backfill_retained_text.py)"
    return "authors (paper): repository default model at the time; generating script not preserved"


def rows_of(obj, mode):
    if mode:
        return obj[mode]
    if isinstance(obj, list):
        return obj
    return obj["results"]


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def load_profiles(root: Path) -> dict[str, tuple[str, dict]]:
    out = {}
    for src, pattern in (("generated", "generated/*.json"), ("pilot", "pilots/*.json")):
        for f in sorted(glob.glob(str(root / pattern))):
            d = json.load(open(f))
            out[d["profile_id"]] = (src, d)
    return out


def build_facts(root, profiles, text_eval_ids, mm_eval_ids):
    header = ["profile_id", "profile_source", "profile_name", "n_facts_in_profile", "fact_id", "category",
              "visual_grounding", "forget_target", "text", "n_probes",
              "in_legacy_text_eval", "in_multimodal_eval"]
    rows = []
    for pid, (src, d) in sorted(profiles.items()):
        for f in d["facts"]:
            rows.append({
                "profile_id": pid, "profile_source": src, "profile_name": d.get("name", ""),
                "n_facts_in_profile": len(d["facts"]), "fact_id": f["fact_id"], "category": f["category"],
                "visual_grounding": f.get("visual_grounding", ""), "forget_target": str(bool(f.get("forget_target"))).lower(),
                "text": f["text"], "n_probes": len(f.get("probes") or []),
                "in_legacy_text_eval": str(pid in text_eval_ids).lower(),
                "in_multimodal_eval": str(pid in mm_eval_ids).lower(),
            })
    write_csv(root / "metadata/facts.csv", header, rows)
    return rows


def build_images(root, profiles, repo_id):
    header = ["path", "image_set", "profile_id", "fact_id", "duplicate_of", "sha256", "size_bytes",
              "source", "license", "content_url"]
    # map pilot synthetic file names to fact ids via the profile JSON image references
    by_basename = {}
    fact_ids = {}
    for pid, (_, d) in profiles.items():
        fact_ids[pid] = {f["fact_id"] for f in d["facts"]}
        for f in d["facts"]:
            for im in f.get("images") or []:
                by_basename[(pid, os.path.basename(im["path"]))] = f["fact_id"]
    prompts = set()
    if (root / "generation/image_prompts.json").exists():
        prompts = {(x["profile_id"], x["fact_id"]) for x in json.load(open(root / "generation/image_prompts.json"))}
    synth_hash = {}
    rows = []
    for p in sorted(Path(root, "images").rglob("*.jpg")):
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if parts[1] == "synthetic":
            image_set = "synthetic"
            pid = parts[2]
        else:
            image_set = "synthetic_legacy_path"
            pid = parts[1]
        stem = p.stem
        fid = stem if stem in fact_ids.get(pid, ()) else by_basename.get((pid, p.name), "")
        digest, size = sha256_and_size(p)
        row = {"path": rel, "image_set": image_set, "profile_id": pid, "fact_id": fid, "sha256": digest,
               "size_bytes": size, "content_url": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}"}
        if (pid, stem) in prompts:
            row["source"] = ("Generated with Google Gemini 3.1 Flash Image (gemini-3.1-flash-image-preview) in one "
                             "batch job, April 2026; prompt in generation/image_prompts.json")
        elif pid.startswith("pilot"):
            row["source"] = ("Pilot image, 20 April 2026. The generator is not logged in the image metadata; "
                             "the commit that added the 64 pilot images records Qwen-Image")
        else:
            row["source"] = ("Early structural-floor pilot image (21 April 2026), not used in the reported "
                             "evaluations; generator not recorded")
        row["license"] = "MIT"
        if image_set == "synthetic":
            synth_hash[(pid, p.name)] = rel
        rows.append(row)
    for r in rows:
        if r["image_set"] == "synthetic_legacy_path":
            r["duplicate_of"] = synth_hash.get((r["profile_id"], os.path.basename(r["path"])), "")
    write_csv(root / "metadata/images.csv", header, rows)
    return rows


def build_judgments(root):
    header = ["experiment", "source_file", "probe_id", "profile_id", "fact_id", "category", "visual_grounding",
              "n_images", "prober_model", "model_ids_source",
              "gpt41_model", "gpt41_verdict", "gpt41_confidence",
              "claude_judge_model", "claude_verdict", "claude_confidence",
              "gemini_model", "gemini_verdict", "gemini_confidence",
              "vote_count", "ensemble_verdict", "leaked"]
    rows = []
    for exp, src, mode, jkey, prober, claude_model, id_src in EXPERIMENTS:
        if not (root / src).exists():
            continue
        for r in rows_of(json.load(open(root / src)), mode):
            j = r.get(jkey) or {}
            row = {"experiment": exp, "source_file": src, "probe_id": r.get("probe_id", ""),
                   "profile_id": r["profile_id"], "fact_id": r["fact_id"], "category": r.get("category", ""),
                   "visual_grounding": r.get("visual_grounding", ""),
                   "n_images": r.get("n_images", r.get("n_remaining", "")),
                   "prober_model": r.get("vlm_model", prober or retained_text_prober(r)),
                   "model_ids_source": id_src or retained_text_id_source(r),
                   "vote_count": j.get("vote_count", ""), "ensemble_verdict": j.get("ensemble_verdict", "")}
            for key, prefix in (("gpt41", "gpt41"), ("claude_sonnet", "claude"), ("gemini_flash", "gemini")):
                v = j.get(key)
                if isinstance(v, dict):
                    row[f"{prefix}_verdict"] = v.get("verdict", "")
                    row[f"{prefix}_confidence"] = v.get("confidence", "")
            row["gpt41_model"] = "gpt-4.1"
            row["claude_judge_model"] = r.get("claude_judge_model", claude_model)
            row["gemini_model"] = "gemini-2.5-flash"
            leaked = r.get("leaked_llm", r.get("leaked", r.get("post_deletion_leaked")))
            row["leaked"] = "" if leaked is None else str(bool(leaked)).lower()
            rows.append(row)
    write_csv(root / "metadata/multimodal_judgments.csv", header, rows)
    return rows


def build_entailment(root):
    header = ["profile_id", "fact_id", "category", "fact_text", "leaked_core_retained",
              "text_entailment_label", "text_entailment_justification",
              "leak_taxonomy_label_v1_0", "leak_taxonomy_label", "leak_taxonomy_reason"]
    a = root / "results/supplementary/entailment_all300.json"
    if not a.exists():
        return []
    sub = {r["fact_id"]: r for r in json.load(open(root / "results/entailment_stratification.json"))["results"]}
    reb_path = root / "results/supplementary/entailment_stratification.json"
    reb = {r["fact_id"]: r for r in json.load(open(reb_path))["results"]} if reb_path.exists() else {}
    rows = []
    for r in json.load(open(a))["results"]:
        fid = r["fact_id"]
        rows.append({
            "profile_id": r["profile_id"], "fact_id": fid, "category": r["category"], "fact_text": r["fact_text"],
            "leaked_core_retained": str(bool(r["leaked"])).lower(),
            "text_entailment_label": r["label"], "text_entailment_justification": r["justification"],
            "leak_taxonomy_label_v1_0": sub.get(fid, {}).get("label", ""),
            "leak_taxonomy_label": reb.get(fid, {}).get("label", ""),
            "leak_taxonomy_reason": reb.get(fid, {}).get("reason", ""),
        })
    write_csv(root / "metadata/entailment_labels.csv", header, rows)
    return rows


def eval_profile_ids(root):
    text_ids, mm_ids = set(), set()
    p = root / "results/Oracle_generated.json"
    if p.exists():
        text_ids = {r["profile_id"] for r in json.load(open(p))}
    p = root / "results/structural_floor_scale_llm_judged.json"
    if p.exists():
        d = json.load(open(p))
        mm_ids = {r["profile_id"] for m in ("same_fact", "retained", "negative") for r in d[m]}
    return text_ids, mm_ids


# ---------------------------------------------------------------- Croissant

CONTEXT = {
    "@language": "en", "@vocab": "https://schema.org/", "citeAs": "cr:citeAs", "column": "cr:column",
    "conformsTo": "dct:conformsTo", "cr": "http://mlcommons.org/croissant/", "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
    "data": {"@id": "cr:data", "@type": "@json"}, "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/", "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"}, "extract": "cr:extract", "field": "cr:field",
    "fileProperty": "cr:fileProperty", "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
    "includes": "cr:includes", "isLiveDataset": "cr:isLiveDataset", "jsonPath": "cr:jsonPath", "key": "cr:key",
    "md5": "cr:md5", "parentField": "cr:parentField", "path": "cr:path", "recordSet": "cr:recordSet",
    "references": "cr:references", "regex": "cr:regex", "repeated": "cr:repeated", "replace": "cr:replace",
    "samplingRate": "cr:samplingRate", "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
    "subField": "cr:subField", "transform": "cr:transform",
}

CSV_RECORDSETS = [
    # (record set id, file path, description, [(column, dataType, description)])
    ("facts", "metadata/facts.csv", "One row per fact of every released profile (143 generated + 5 pilot profiles).", [
        ("profile_id", "sc:Text", "Profile identifier (pNNN or pilot_NNN)."),
        ("profile_source", "sc:Text", "generated or pilot."),
        ("profile_name", "sc:Text", "Fictional name of the synthetic person."),
        ("n_facts_in_profile", "sc:Integer", "Number of facts in the profile (benchmark profiles have 20)."),
        ("fact_id", "sc:Text", "Fact identifier."),
        ("category", "sc:Text", "location, profession, preference, relationship or possession."),
        ("visual_grounding", "sc:Text", "explicit, implicit or absent (assigned by the generation pipeline)."),
        ("forget_target", "sc:Boolean", "Whether the fact is a deletion target."),
        ("text", "sc:Text", "Fact text."),
        ("n_probes", "sc:Integer", "Number of probe questions attached to the fact in the profile JSON."),
        ("in_legacy_text_eval", "sc:Boolean", "Profile is among the 113 used in the legacy (v1.0) text-only evaluation."),
        ("in_multimodal_eval", "sc:Boolean", "Profile is among the 38 used in the n=300 multimodal evaluation."),
    ]),
    ("images", "metadata/images.csv", "One row per image file with checksum, provenance and license.", [
        ("path", "sc:Text", "Repository-relative path."),
        ("image_set", "sc:Text", "synthetic, or synthetic_legacy_path (byte-identical copy kept for old paths)."),
        ("profile_id", "sc:Text", "Profile the image belongs to."),
        ("fact_id", "sc:Text", "Fact the image was generated or retrieved for, when resolvable."),
        ("duplicate_of", "sc:Text", "For legacy-path copies, the canonical path."),
        ("sha256", "sc:Text", "SHA-256 of the file content."),
        ("size_bytes", "sc:Integer", "File size in bytes."),
        ("source", "sc:Text", "How the image was produced or obtained."),
        ("license", "sc:Text", "License that applies to this file."),
        ("content_url", "sc:URL", "Download URL of the file."),
    ]),
    ("multimodal_judgments", "metadata/multimodal_judgments.csv", "One row per probe of every multimodal and baseline condition, with each judge's verdict and the ensemble verdict (majority, at least 2 of 3 votes = leaked).", [
        ("experiment", "sc:Text", "Condition (e.g. core_retained, retained_sonnet5_rerun)."),
        ("source_file", "sc:Text", "Result file the row was extracted from."),
        ("probe_id", "sc:Text", "Probe identifier."),
        ("profile_id", "sc:Text", "Profile identifier."),
        ("fact_id", "sc:Text", "Deleted fact being probed."),
        ("category", "sc:Text", "Fact category."),
        ("visual_grounding", "sc:Text", "Visual grounding of the deleted fact."),
        ("n_images", "sc:Text", "Number of images shown to the prober."),
        ("prober_model", "sc:Text", "Model that answered the probe."),
        ("model_ids_source", "sc:Text", "Whether model IDs are recorded in the result file or taken from the producer script."),
        ("gpt41_model", "sc:Text", "OpenAI judge model ID."),
        ("gpt41_verdict", "sc:Text", "OpenAI judge verdict (leaked, not_leaked or ambiguous)."),
        ("gpt41_confidence", "sc:Text", "OpenAI judge confidence."),
        ("claude_judge_model", "sc:Text", "Anthropic judge model ID."),
        ("claude_verdict", "sc:Text", "Anthropic judge verdict."),
        ("claude_confidence", "sc:Text", "Anthropic judge confidence."),
        ("gemini_model", "sc:Text", "Google judge model ID."),
        ("gemini_verdict", "sc:Text", "Google judge verdict."),
        ("gemini_confidence", "sc:Text", "Google judge confidence."),
        ("vote_count", "sc:Text", "Leaked votes / valid votes."),
        ("ensemble_verdict", "sc:Text", "Majority verdict."),
        ("leaked", "sc:Text", "Final leak label used in the paper (true/false)."),
    ]),
    ("entailment_labels", "metadata/entailment_labels.csv", "Both entailment classifiers' labels for the 300 retained-image probes (see results/supplementary/entailment_classifier_prompts.md).", [
        ("profile_id", "sc:Text", "Profile identifier."),
        ("fact_id", "sc:Text", "Deleted fact."),
        ("category", "sc:Text", "Fact category."),
        ("fact_text", "sc:Text", "Deleted fact text."),
        ("leaked_core_retained", "sc:Text", "Leak label in the core retained-image condition."),
        ("text_entailment_label", "sc:Text", "Classifier B (text-only, all 300): ENTAILED, CORRELATED or UNRELATED."),
        ("text_entailment_justification", "sc:Text", "Classifier B justification."),
        ("leak_taxonomy_label_v1_0", "sc:Text", "Classifier A on the 36 leaks as released in v1.0 (includes 4 unparsed labels, UNKNOWN)."),
        ("leak_taxonomy_label", "sc:Text", "Classifier A on the 36 leaks, with the 4 unparsed labels re-classified."),
        ("leak_taxonomy_reason", "sc:Text", "Classifier A justification."),
    ]),
]

CSV_RECORDSETS += [
    ("real_photos_unsplash", "metadata/real_photos_unsplash.csv", "References for the 523 Unsplash photos of the real-photo condition. The photos are not redistributed (most are Unsplash+ previews); hashes, watermark audit labels and search queries are kept so they can be re-sourced.", [
        ("former_path", "sc:Text", "Path of the photo in v1.0."),
        ("profile_id", "sc:Text", "Profile identifier."),
        ("fact_id", "sc:Text", "Fact the photo was retrieved for."),
        ("sha256", "sc:Text", "SHA-256 of the photo file."),
        ("size_bytes", "sc:Integer", "File size in bytes."),
        ("unsplash_plus_watermark", "sc:Text", "Visual audit: the Unsplash+ preview watermark is visible (yes/no/unclear). A lower bound: 'no' and 'unclear' include misses."),
        ("search_query", "sc:Text", "Unsplash search query built from the fact text (scripts/source_real_images.py)."),
        ("fallback_query", "sc:Text", "First four words of the query, used by the script when the full query returned nothing (which query hit was not logged)."),
        ("status", "sc:Text", "Release status."),
    ]),
    ("real_collections_images", "real_collections/images.csv", "Wikimedia Commons images of the 9 genuine personal collections, linked (not rehosted) under pseudonymous collection IDs, with license and author attribution.", [
        ("collection_id", "sc:Text", "Pseudonymous collection ID (RC01-RC09)."),
        ("image_index", "sc:Integer", "Position of the image in its stream."),
        ("commons_file_page", "sc:URL", "Wikimedia Commons file page."),
        ("image_url", "sc:URL", "Image file URL on Wikimedia Commons."),
        ("license", "sc:Text", "License short name as given on Commons."),
        ("license_url", "sc:URL", "License URL (empty for public-domain marks)."),
        ("attribution_required", "sc:Text", "Whether the license requires attribution."),
        ("author_attribution", "sc:Text", "Author attribution as given on Commons."),
        ("credit", "sc:Text", "Credit line as given on Commons."),
        ("ai_generated_tag", "sc:Text", "Commons marks the image as AI-generated."),
        ("stream_note", "sc:Text", "Audit note for the stream (e.g. third-party uploads)."),
    ]),
    ("real_collections_probes", "real_collections/probes.csv", "The 69 genuine-collection probe outcomes with per-judge verdicts and blind controls, shuffled and without collection or image identifiers. Facts and model responses are withheld.", [
        ("category", "sc:Text", "Fact category."),
        ("n_retained_images", "sc:Integer", "Number of retained images shown to the prober."),
        ("judge_gpt41", "sc:Text", "GPT-4.1 verdict."),
        ("judge_claude", "sc:Text", "Claude Sonnet 4.5 verdict."),
        ("judge_gemini", "sc:Text", "Gemini 2.5 Flash verdict."),
        ("vote_count", "sc:Text", "Leaked votes out of 3."),
        ("leaked", "sc:Text", "Ensemble leak label (true/false)."),
        ("blind_vote_count", "sc:Text", "Leaked votes for the blind (no-image) control."),
        ("blind_leaked", "sc:Text", "Blind-control leak label."),
    ]),
    ("real_collections_gate_labels", "real_collections/gate_labels.csv", "Gate label of each of the 18 sourced Commons streams under pseudonymous IDs.", [
        ("collection_id", "sc:Text", "Pseudonymous ID (RC = kept, RX = excluded)."),
        ("gate_label", "sc:Text", "personal_stream, excluded_bot, institutional or single_subject."),
        ("kept", "sc:Text", "Whether the stream was evaluated."),
        ("n_images_sourced", "sc:Integer", "Number of images sourced from the stream."),
    ]),
]

HUMAN_COLUMNS = [
    ("id", "sc:Integer", "Item number."), ("mode", "sc:Text", "same_fact, retained or negative."),
    ("profile_id", "sc:Text", "Profile identifier."), ("fact_id", "sc:Text", "Deleted fact."),
    ("category", "sc:Text", "Fact category."), ("fact_text", "sc:Text", "Deleted fact text."),
    ("vlm_response", "sc:Text", "Prober response shown to the annotator."),
    ("ensemble_verdict", "sc:Text", "Judge-ensemble verdict (this was visible to annotators; see dataset card)."),
    ("vote_count", "sc:Text", "Ensemble vote count."), ("human_verdict", "sc:Text", "The annotator's label."),
]


def file_object(root, rel, repo_id, desc, fmt):
    digest, _ = sha256_and_size(root / rel)
    return {"@type": "cr:FileObject", "@id": rel, "name": rel, "description": desc,
            "contentUrl": f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel}",
            "encodingFormat": fmt, "sha256": digest}


def record_set(rs_id, file_id, desc, cols):
    return {"@type": "cr:RecordSet", "@id": rs_id, "name": rs_id, "description": desc,
            "field": [{"@type": "cr:Field", "@id": f"{rs_id}/{c}", "name": c, "description": d, "dataType": t,
                       "source": {"fileObject": {"@id": file_id}, "extract": {"column": c}}} for c, t, d in cols]}


def build_croissant(root, repo_id, card):
    dist = [{"@type": "cr:FileObject", "@id": "repo", "name": "repo",
             "description": "The Hugging Face dataset repository (git).",
             "contentUrl": f"https://huggingface.co/datasets/{repo_id}", "encodingFormat": "git+https", "sha256": "main"}]
    filesets = [
        ("generated-profiles", "Synthetic profile JSON files (143; the 113 with exactly 20 facts form the benchmark).", "application/json", "generated/*.json"),
        ("pilot-profiles", "Hand-built pilot profile JSON files (5).", "application/json", "pilots/*.json"),
        ("synthetic-images", "Gemini-generated images, one folder per profile.", "image/jpeg", "images/synthetic/**/*.jpg"),
        ("result-files", "Per-probe result files of v1.0 (JSON). results/README.md marks which are reported in the paper; the legacy files listed in the 'legacy-unreported-results' file set do not measure deletion.", "application/json", "results/*.json"),
        ("legacy-unreported-results", "Legacy text-only system results and pilot deletion-policy ablations of v1.0. They do NOT measure deletion (no fact was injected or deletion requested; runner reset bug) and are not reported in the paper. Kept for auditability.", "application/json", ["results/*_generated.json", "results/ablations.json"]),
        ("supplementary-result-files", "Result files of the supplementary analyses added in v1.1 (JSON).", "application/json", "results/supplementary/*.json"),
        ("code", "Evaluation harness, judge and analysis scripts (Python, MIT).", "text/x-python", "code/**/*.py"),
    ]
    for fid, desc, fmt, inc in filesets:
        if not any(glob.glob(str(root / i), recursive=True) for i in ([inc] if isinstance(inc, str) else inc)):
            continue
        dist.append({"@type": "cr:FileSet", "@id": fid, "name": fid, "description": desc,
                     "containedIn": {"@id": "repo"}, "encodingFormat": fmt, "includes": inc})
    if (root / "generation/image_prompts.json").exists():
        dist.append(file_object(root, "generation/image_prompts.json", repo_id,
                                "Prompts (and negative prompts) for the 536 batch-generated Gemini images.", "application/json"))
    if (root / "real_collections/summary.json").exists():
        dist.append(file_object(root, "real_collections/summary.json", repo_id,
                                "Aggregate results of the genuine-collections experiment.", "application/json"))
    record_sets = []
    for rs_id, rel, desc, cols in CSV_RECORDSETS:
        if not (root / rel).exists():
            continue
        dist.append(file_object(root, rel, repo_id, desc, "text/csv"))
        record_sets.append(record_set(rs_id, rel, desc, cols))
    for k, rel in ((1, "results/human_validation_sample.csv"), (2, "results/human_validation_sample_annotator2.csv")):
        dist.append(file_object(root, rel, repo_id, f"Human validation labels, annotator {k} (100 items).", "text/csv"))
        record_sets.append(record_set(f"human_validation_annotator{k}", rel,
                                      f"Annotator {k}'s labels for the 100-item stratified sample of judge verdicts.", HUMAN_COLUMNS))
    meta = {"@context": CONTEXT, "@type": "sc:Dataset", **card,
            "conformsTo": ["http://mlcommons.org/croissant/1.1", "http://mlcommons.org/croissant/RAI/1.0"],
            "url": f"https://huggingface.co/datasets/{repo_id}",
            "distribution": dist, "recordSet": record_sets}
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--card", default="code/release_tools/croissant_card.json",
                    help="JSON with dataset-level and RAI/provenance fields merged into croissant.json")
    ap.add_argument("--local-croissant", help="also write a copy whose contentUrls are local paths (for offline record checks)")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    profiles = load_profiles(root)
    text_ids, mm_ids = eval_profile_ids(root)
    facts = build_facts(root, profiles, text_ids, mm_ids)
    images = build_images(root, profiles, a.repo_id)
    judg = build_judgments(root)
    ent = build_entailment(root)
    repo_url = f"https://huggingface.co/datasets/{a.repo_id}"
    card = json.loads(open(root / a.card).read().replace("{REPO_URL}", repo_url))
    meta = build_croissant(root, a.repo_id, card)
    with open(root / "croissant.json", "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    if a.local_croissant:
        local = json.loads(json.dumps(meta))
        prefix = f"https://huggingface.co/datasets/{a.repo_id}/resolve/main/"
        for d in local["distribution"]:
            if d.get("contentUrl", "").startswith(prefix):
                d["contentUrl"] = os.path.relpath(root / d["contentUrl"][len(prefix):], Path(a.local_croissant).resolve().parent)
        json.dump(local, open(a.local_croissant, "w"), indent=2, ensure_ascii=False)
    print(f"facts={len(facts)} images={len(images)} judgments={len(judg)} entailment={len(ent)} "
          f"record_sets={len(meta['recordSet'])} distribution={len(meta['distribution'])}")


if __name__ == "__main__":
    main()
