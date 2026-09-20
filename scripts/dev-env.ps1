# dev-env.ps1 -- redirect every large artefact off the system drive.
#
# Why this exists: the system drive has ~32 GB free. A default install puts the
# HuggingFace cache in ~/.cache/huggingface and the uv cache in %LOCALAPPDATA%,
# and the five encoders plus two GGUF generators plus an NLI model come to ~11 GB
# before any data. Run this BEFORE the first `uv sync` or the first model pull --
# a half-downloaded 2 GB model on a full drive is a bad afternoon.
#
# Usage (dot-source it, so the variables survive into your shell):
#     . .\scripts\dev-env.ps1

$ErrorActionPreference = "Stop"

# Pick the data root: G: has the most headroom on this machine. Override by
# setting INDICRAG_DATA_ROOT before dot-sourcing.
if (-not $env:INDICRAG_DATA_ROOT) {
    $env:INDICRAG_DATA_ROOT = "G:\indicrag-data"
}
$root = $env:INDICRAG_DATA_ROOT

foreach ($d in @($root, "$root\hf", "$root\uv-cache", "$root\corpus", "$root\models")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

# HuggingFace: HF_HOME covers hub downloads, datasets and the transformers cache
# in current versions. TRANSFORMERS_CACHE is set too for older code paths that
# still read it directly.
$env:HF_HOME            = "$root\hf"
$env:TRANSFORMERS_CACHE = "$root\hf\transformers"
$env:SENTENCE_TRANSFORMERS_HOME = "$root\hf\sentence-transformers"

# uv: cache and the project venv both off C:
$env:UV_CACHE_DIR          = "$root\uv-cache"
$env:UV_PROJECT_ENVIRONMENT = "$root\venv"

# Where the project writes corpora, embeddings and generation caches.
$env:INDICRAG_DATA_DIR = "$root\corpus"

# CPU-only machine: cap thread counts at the physical core count. Left unset,
# torch and OpenMP oversubscribe the 8 logical cores and encoding gets slower,
# not faster.
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

$free = [math]::Round((Get-PSDrive ($root.Substring(0,1))).Free / 1GB, 1)
Write-Host "IndicRAG dev environment" -ForegroundColor Cyan
Write-Host "  data root : $root  ($free GB free)"
Write-Host "  HF_HOME   : $env:HF_HOME"
Write-Host "  uv venv   : $env:UV_PROJECT_ENVIRONMENT"
Write-Host "  threads   : $env:OMP_NUM_THREADS"
if ($free -lt 15) {
    Write-Host "  WARNING: under 15 GB free on the data drive; model pulls may fail." -ForegroundColor Yellow
}
