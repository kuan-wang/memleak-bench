# MemLeak: Diagnosing Information Leaks in Multimodal Agent Memory

**Kuan Wang and Chao Zhang** · Georgia Institute of Technology

NeurIPS 2026 · Paper: [arXiv:2606.29788](https://arxiv.org/abs/2606.29788) · Dataset: [memleak-bench/memleak-benchmark](https://huggingface.co/datasets/memleak-bench/memleak-benchmark) on Hugging Face

An agent memory can delete a stored fact while keeping the user's other memories, such as photos and other facts. MemLeak tests whether a vision-language model can reconstruct the deleted fact from what was kept. For example, it probes a model with only the photos tied to retained facts, then asks an ensemble of three LLM judges whether the answer reveals the deleted fact.

This repository holds the evaluation code: the profile schema, the memory-system adapters, the judges, the metrics and the experiment scripts. The profiles, images, per-probe results, human labels, dataset card and Croissant metadata are on [Hugging Face](https://huggingface.co/datasets/memleak-bench/memleak-benchmark). The dataset card documents every experiment, model and known issue.

> **Read before use.** The legacy text-only system results and the pilot ablations (`results/*_generated.json`, `results/ablations.json` in the dataset) do not measure deletion: no fact was ever injected and no forget instruction was issued. The paper does not report them, and they were not rerun. See "Known issues and corrections" on the [dataset card](https://huggingface.co/datasets/memleak-bench/memleak-benchmark).

## Repository layout

| Path | Contents |
|---|---|
| `memleak/schema/` | Profile, fact and probe types, and validators |
| `memleak/datagen/` | Profile, image-prompt and image generation; `text_stream.py` builds a conversation with an explicit deletion request |
| `memleak/harness/` | Memory-system adapters (Oracle, Naive, LongContext, Mem0, a version-pinned Letta server, and the legacy simulated Letta wrapper) and the evaluation runner |
| `memleak/judge/` | Leakage judge prompts and the pinned 3-judge ensemble (`ensemble3.py`) |
| `memleak/metrics/` | Cross-modal leakage rate, directional asymmetry, entanglement ratio, retention integrity, bootstrap and permutation tests |
| `memleak/llm.py`, `memleak/backbone.py` | Model clients and the recorded backbone used by the system runs |
| `scripts/` | Experiment entry points (see [Which script produced which result](#which-script-produced-which-result)) |
| `scripts/text_comparison/` | The fixed text-stream protocol and the Letta 0.16.8 server launcher (released, not run for the paper) |
| `tests/` | 72 unit tests |
| `env/` | Pinned environments: `experiment-freeze.txt`, the Letta server freeze and source commit, Python version |
| `release_tools/` | Builds the dataset's flat tables and Croissant file; run it from a dataset checkout |
| `setup_data_layout.sh` | Downloads the dataset and builds the `data/` layout that the scripts expect |

## Setup

From a clone of this repository:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r env/experiment-freeze.txt   # pinned; or: pip install -e ".[dev,mem0]"
pip install huggingface_hub                # provides the `hf` CLI; adds packages without changing the pinned ones
bash setup_data_layout.sh                  # downloads the dataset into data/ (about 560 MB; --no-images for about 20 MB)
cp .env.example .env                       # add your API keys
python -m pytest -q tests                  # 72 tests; they need data/pilots but no API keys
```

- **Python version.** The pinned environment uses Python 3.12.15 (`env/python-versions.txt`).
- **Data.** `setup_data_layout.sh` runs `hf download` on the dataset at the tag `v1.1` and places the files under `data/hf_dataset/`. It then links `data/generated`, `data/pilots`, `data/images` and `data/image_batch/image_prompts.json`, and copies `results/` to `data/results/` so that re-running a script never overwrites the released files. Options:
  - `--revision REV` (or `MEMLEAK_REVISION=REV`) selects another tag, branch or commit. Version v1.0 is the tag `v1.0`.
  - `--no-images` skips the 646 synthetic images.
  - `--from-local DIR` links an existing clone of the dataset instead of downloading it. Run `git lfs pull` in the clone first.
  - `--dry-run` lists the files that would be downloaded.
- **Not downloaded.** The Unsplash photos of the real-photo condition are not redistributed. `metadata/real_photos_unsplash.csv` in the dataset lists each photo's search query and sha256, so `scripts/source_real_images.py` can re-source them into `data/real_images/`; search results change over time. The genuine-collection images are linked from `real_collections/images.csv` in the dataset.
- **Running scripts.** Run them from the repository root, for example `python scripts/run_structural_floor_at_scale.py`. They call the OpenAI, Anthropic and Google APIs, so they need keys in `.env` and incur API costs.
- **Letta server adapter.** It expects Letta 0.16.8 at commit `1131535` (`env/letta-source.txt`, `env/letta-server-freeze.txt`). `scripts/text_comparison/launch_letta_server.py` starts it.
- **Claude backend.** Claude calls go through the Anthropic API by default. The optional `CLAUDE_BACKEND=cc` setting routes them through the Claude Code CLI instead (`scripts/cc_llm.py`); vision calls then pass images as files.

## Which script produced which result

Result paths are relative to the dataset root. After setup, the `results/` files are also under `data/results/`.

| Result file | Script(s) |
|---|---|
| `results/structural_floor_scale_llm_judged.json` | `scripts/run_structural_floor_at_scale.py` (probes) → `scripts/rejudge_structural_floor.py` (3-judge verdicts) |
| `results/structural_floor_gpt4o_retained.json` | `scripts/run_gpt4o_retained.py` |
| `results/real_image_eval_retained.json` | `scripts/source_real_images.py` (Unsplash retrieval) → `scripts/run_real_image_eval.py` |
| `results/mem0_multimodal_retained.json` | `scripts/run_mem0_multimodal.py` |
| `results/semantic_deletion_scale_n300.json` | `scripts/run_semantic_deletion_at_scale.py` |
| `results/retained_text_only_baseline.json` | First 280 probes: the generating script was not preserved (prober: Claude Sonnet 4.6, the repository's default model at the time). Last 20 probes: `scripts/backfill_retained_text.py` |
| `results/blind_inference_baseline.json` | The generating script was not preserved (prober and Claude judge: Claude Sonnet 4, the repository's default model at the time) |
| `results/human_validation_sample*.csv` | `scripts/gen_human_validation_csv.py`, labeled with `scripts/human_validation_ui.py` |
| `results/fc_inventory.json` | `scripts/run_parallel_tasks.py` |
| `results/entailment_stratification.json` (v1.0 file) | `scripts/run_entailment_stratification.py` |
| `results/supplementary/entailment_all300.json`, `results/supplementary/entailment_stratification.json` | `scripts/run_entailment_all300.py` (also re-classifies the 4 UNKNOWNs) |
| `results/supplementary/judge_robustness_analysis.json` | `scripts/judge_robustness_analysis.py` |
| `results/supplementary/entailment_validation/` (blank sheets, not completed) | `score.py` in that folder scores completed sheets |
| `results/supplementary/structural_floor_sonnet5_retained.json` | `scripts/run_sonnet5_retained.py` |
| `results/supplementary/semdel_utility_cost.json` | `scripts/run_semdel_utility_cost.py` |
| `results/supplementary/semantic_deletion_text_n300.json` | `scripts/run_semantic_deletion_text.py` |
| `real_collections/*` | `scripts/run_real_collections.py` (stages: source, extract, gate, probe). The released tables are de-identified exports; see `real_collections/README.md` in the dataset |
| `results/*_generated.json`, `results/ablations.json` (legacy; do not measure deletion) | `scripts/run_generated.py` and the merge scripts (`gen_and_run_5new.py`, `run_new8_all_systems.py`, `run_missing_profiles.py`, `run_missing_nlm.py`, `run_mem0_5new.py`); `scripts/run_ablations.py` |

## Reproducibility notes

- **Retired model.** Claude Sonnet 4 (`claude-sonnet-4-20250514`), the prober and Claude judge of the v1.0 experiments, was retired on 2026-06-15. Those runs cannot be repeated exactly. Pass a current model to re-run them; the later reruns used Claude Sonnet 4.5 and Claude Sonnet 5.
- **Sampling variation.** Model outputs are not deterministic, so re-runs give rates within sampling variation, not identical files.
- **Fixed harness, not used for paper results.** The runner in this version fixes the defects behind the legacy text-only results. It materializes a conversation with an explicit deletion request (`memleak/datagen/text_stream.py`), checks acquisition before scoring, and pins the 3-judge ensemble (`memleak/judge/ensemble3.py`). `scripts/text_comparison/run_text_comparison.py` and the Letta server adapter are released for future evaluations. **They were not run for the paper**, and no results from them are included.

## Relation to the dataset's `code/` folder

This repository holds the same code as the `code/` folder of the dataset at tag `v1.1`. The code was exported from the authors' development repository; development-only scripts are left out, comments and a few file names were edited for the public release, and a non-public image-generation backend (used only for the 64 pilot images) was removed from `memleak/datagen/images.py`. To make the code usable on its own, this repository:
- replaces `code/README.md` with this file;
- rewrites `setup_data_layout.sh` to download the dataset from Hugging Face, where the dataset's version links the sibling folders of a dataset clone;
- points `.env.example` at `.env` in the repository root;
- adds `LICENSE`, `CITATION.cff` and `.gitignore`.

All other files are byte-identical to the dataset's `code/` folder. The v1.0 code is in the dataset at the tag `v1.0`.

## Citation

```bibtex
@inproceedings{wang2026memleak,
  title     = {{MemLeak}: Diagnosing Information Leaks in Multimodal Agent Memory},
  author    = {Wang, Kuan and Zhang, Chao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2606.29788},
  archivePrefix = {arXiv}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The dataset contains material under more than one license. Its `LICENSE.md` and `metadata/images.csv` give the license of each part.

## Contact

Kuan Wang, kuanwang@gatech.edu. Please report problems with the code as issues on this repository. Takedown requests for any image or image reference in the dataset are honored; see the dataset card.
