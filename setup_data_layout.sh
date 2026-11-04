#!/usr/bin/env bash
# Build the data/ layout that the scripts and tests expect, from the MemLeak dataset on Hugging Face
# (https://huggingface.co/datasets/memleak-bench/memleak-benchmark).
#
#   bash setup_data_layout.sh                    download from the Hub with the `hf` CLI (about 560 MB)
#   bash setup_data_layout.sh --no-images        profiles, prompts and results only (about 20 MB)
#   bash setup_data_layout.sh --revision REV     another tag, branch or commit of the dataset
#   bash setup_data_layout.sh --from-local DIR   use a dataset clone you already have (run `git lfs pull` in it)
#   bash setup_data_layout.sh --dry-run          list what would be downloaded, change nothing
#
# The default revision is the release tag v1.1; MEMLEAK_REVISION overrides it.
# Run from anywhere. The result, relative to this repository:
#   data/hf_dataset/                     the downloaded dataset files (download mode only)
#   data/generated                       -> generated/           profile JSONs
#   data/pilots                          -> pilots/              pilot profile JSONs
#   data/images                          -> images/              images/<profile>/<fact_id>.jpg
#   data/image_batch/image_prompts.json  -> generation/image_prompts.json
#   data/results/                        a copy of results/, so re-running a script never overwrites the released files
set -euo pipefail

REPO_ID="memleak-bench/memleak-benchmark"
REVISION="${MEMLEAK_REVISION:-v1.1}"
LOCAL_SRC=""
WITH_IMAGES=1
DRY_RUN=0

usage() { sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --revision)   REVISION="${2:?--revision needs a value}"; shift 2 ;;
    --from-local) LOCAL_SRC="${2:?--from-local needs a directory}"; shift 2 ;;
    --no-images)  WITH_IMAGES=0; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *)            echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")"

if [ -n "$LOCAL_SRC" ]; then
  SRC="$(cd "$LOCAL_SRC" && pwd)"   # absolute, so the links resolve from data/
  LINK_BASE="$SRC"
  [ -d "$SRC/generated" ] || { echo "error: $SRC has no generated/ folder; is it a MemLeak dataset checkout?" >&2; exit 1; }
  echo "Using the local dataset checkout $SRC"
else
  command -v hf >/dev/null 2>&1 || { echo "error: the 'hf' CLI is required: pip install -U huggingface_hub" >&2; exit 1; }
  SRC="$(pwd)/data/hf_dataset"
  LINK_BASE="hf_dataset"            # relative to data/, so the repository can be moved
  patterns=(--include 'generated/*' --include 'pilots/*' --include 'generation/*' --include 'results/*')
  if [ "$WITH_IMAGES" = 1 ]; then patterns+=(--include 'images/*'); fi
  dry=(); if [ "$DRY_RUN" = 1 ]; then dry=(--dry-run); fi
  echo "Downloading $REPO_ID at revision $REVISION into data/hf_dataset/"
  # images/synthetic/ holds byte-identical copies of images/<profile>/, which is the path the profiles use.
  if ! hf download "$REPO_ID" --repo-type dataset --revision "$REVISION" --local-dir "$SRC" \
         "${patterns[@]}" --exclude 'images/synthetic/*' --exclude 'images/real_cc/*' ${dry[@]+"${dry[@]}"}; then
    echo "error: download failed. If the revision '$REVISION' does not exist, pass --revision main." >&2
    exit 1
  fi
  if [ "$DRY_RUN" = 1 ]; then echo "Dry run: nothing was downloaded or linked."; exit 0; fi
fi

mkdir -p data/image_batch

link() {  # link <path inside the dataset> <name under data/>
  local target="$LINK_BASE/$1" name="data/$2"
  if [ ! -e "$SRC/$1" ]; then
    echo "  skipped  $name ($1 is not in this copy of the dataset)"
    return 0
  fi
  if [ -e "$name" ] && [ ! -L "$name" ]; then
    echo "error: $name exists and is not a link; move it away and re-run" >&2
    exit 1
  fi
  ln -sfn "$target" "$name"
  echo "  linked   $name -> $target"
}

link generated generated
link pilots pilots
if [ "$WITH_IMAGES" = 1 ]; then link images images; fi
# One level deeper, so a relative target needs one more ../
if [ -n "$LOCAL_SRC" ]; then LINK_BASE="$SRC"; else LINK_BASE="../hf_dataset"; fi
link generation/image_prompts.json image_batch/image_prompts.json

if [ -e data/results ]; then
  echo "  kept     data/results (already exists; delete it to restore the released files)"
elif [ -d "$SRC/results" ]; then
  cp -R "$SRC/results" data/results
  echo "  copied   data/results"
fi

# A clone made without git-lfs holds pointer files instead of JPEGs.
first_image="$(find -L data/images -name '*.jpg' 2>/dev/null | head -n 1 || true)"
if [ -n "$first_image" ] && head -c 40 "$first_image" | grep -q 'git-lfs'; then
  echo "warning: data/images holds Git LFS pointer files, not images. Run 'git lfs pull' in the dataset clone." >&2
fi

echo "data/ layout ready under $(pwd)/data"
echo "Not included: data/real_images (Unsplash photos, not redistributed; re-source with scripts/source_real_images.py)"
echo "              data/real_collections (Wikimedia Commons streams, linked from real_collections/images.csv in the dataset)"
