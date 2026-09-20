#!/usr/bin/env bash
# dev-env.sh -- POSIX twin of dev-env.ps1. Same job: keep ~11 GB of models and
# caches off a system drive with ~32 GB free. Source it, do not execute it:
#
#     source scripts/dev-env.sh

: "${INDICRAG_DATA_ROOT:=/g/indicrag-data}"

mkdir -p "$INDICRAG_DATA_ROOT"/{hf,uv-cache,corpus,models}

export INDICRAG_DATA_ROOT
export HF_HOME="$INDICRAG_DATA_ROOT/hf"
export TRANSFORMERS_CACHE="$INDICRAG_DATA_ROOT/hf/transformers"
export SENTENCE_TRANSFORMERS_HOME="$INDICRAG_DATA_ROOT/hf/sentence-transformers"

export UV_CACHE_DIR="$INDICRAG_DATA_ROOT/uv-cache"
export UV_PROJECT_ENVIRONMENT="$INDICRAG_DATA_ROOT/venv"

export INDICRAG_DATA_DIR="$INDICRAG_DATA_ROOT/corpus"

# CPU-only: cap at physical cores. Oversubscribing 8 logical cores makes
# encoding slower, not faster.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

free_gb=$(df -BG "$INDICRAG_DATA_ROOT" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "IndicRAG dev environment"
echo "  data root : $INDICRAG_DATA_ROOT  (${free_gb:-?} GB free)"
echo "  HF_HOME   : $HF_HOME"
echo "  uv venv   : $UV_PROJECT_ENVIRONMENT"
echo "  threads   : $OMP_NUM_THREADS"
if [ -n "$free_gb" ] && [ "$free_gb" -lt 15 ]; then
    echo "  WARNING: under 15 GB free on the data drive; model pulls may fail."
fi
