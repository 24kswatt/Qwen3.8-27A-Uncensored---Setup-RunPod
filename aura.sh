#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Qwen3.8-27B Huihui Abliterated - RTX PRO 6000 96 GB
#
# TARGET:
#   huihui-ai/Huihui-Qwen3.8-27B-abliterated
#   BF16 weights
#
# DRAFT:
#   incoai/Qwen3.8-27B-DFlash2
#
# SERVING:
#   SGLang 0.5.20
#   FlashInfer
#   FP8 E4M3 KV cache
#   DFlash2 block=8
#   Native context 262144
#   TP=1
#   2 concurrent users
#
# FILES:
#   /workspace/qwen38-sglang/
#   /workspace/models/qwen/
#
# IMPORTANT:
#   HF token is used only during setup/download.
#   It is NOT persisted to disk.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

WORKSPACE="${WORKSPACE:-/workspace}"

APP_DIR="${APP_DIR:-${WORKSPACE}/qwen38-sglang}"
MODEL_ROOT="${MODEL_ROOT:-${WORKSPACE}/models/qwen}"
HF_CACHE="${HF_CACHE:-${WORKSPACE}/hf-cache}"

TARGET_REPO="huihui-ai/Huihui-Qwen3.8-27B-abliterated"
TARGET_DIR="${MODEL_ROOT}/Huihui-Qwen3.8-27B-abliterated-BF16"

DRAFT_REPO="incoai/Qwen3.8-27B-DFlash2"
DRAFT_DIR="${MODEL_ROOT}/Qwen3.8-27B-DFlash2"

SGLANG_VERSION="0.5.20"

CONTEXT_LENGTH=262144
MAX_RUNNING_REQUESTS=2

# extra_buffer_lazy = 4 state slots / request
# 2 users x 4 = 8
MAX_MAMBA_CACHE_SIZE=8

MEM_FRACTION_STATIC="0.90"
CHUNKED_PREFILL_SIZE=2048

PORT="${PORT:-8000}"

VENV="${APP_DIR}/.venv"
LOG_DIR="${APP_DIR}/logs"

ENV_FILE="${APP_DIR}/.env"
PID_FILE="${APP_DIR}/server.pid"
LOG_FILE="${LOG_DIR}/server.log"


# ============================================================
# COLORS / HELPERS
# ============================================================

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

info() {
    echo -e "\n${CYAN}[INFO]${RESET} $*"
}

ok() {
    echo -e "${GREEN}[OK]${RESET} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*" >&2
}

die() {
    echo -e "${RED}[ERROR]${RESET} $*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    echo
    echo -e "${RED}============================================================"
    echo "SETUP FALLITO"
    echo "Linea: ${BASH_LINENO[0]}"
    echo "Exit code: ${exit_code}"
    echo -e "============================================================${RESET}"
    exit "${exit_code}"
}

trap on_error ERR


# ============================================================
# ROOT / OS
# ============================================================

[[ "$(uname -s)" == "Linux" ]] ||
    die "Questo setup richiede Linux."

ARCH="$(uname -m)"

[[ "${ARCH}" == "x86_64" ]] ||
    warn "Architettura rilevata ${ARCH}; setup progettato per x86_64."

if [[ "${EUID}" -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    die "Servono privilegi root o sudo."
fi


# ============================================================
# HF TOKEN - ASK IMMEDIATELY
# ============================================================

echo
echo "============================================================"
echo " Hugging Face authentication"
echo "============================================================"
echo
echo "Inserisci un HF token READ."
echo "L'input è nascosto e il token NON verrà scritto nel .env."
echo

if [[ -n "${HF_TOKEN:-}" ]]; then
    info "HF_TOKEN già presente nell'ambiente."
else
    read -rsp "HF token: " HF_TOKEN
    echo
fi

[[ -n "${HF_TOKEN}" ]] ||
    die "HF token vuoto."

export HF_TOKEN


# ============================================================
# BOOTSTRAP PACKAGES
# ============================================================

if command -v apt-get >/dev/null 2>&1; then

    info "Installazione dipendenze base..."

    export DEBIAN_FRONTEND=noninteractive

    "${SUDO[@]}" apt-get update -qq

    "${SUDO[@]}" apt-get install -y \
        ca-certificates \
        curl \
        git \
        jq \
        openssl \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        pciutils \
        procps \
        iproute2

else
    warn "apt-get non presente: assumo che le dipendenze siano già installate."
fi


# ============================================================
# HF TOKEN VALIDATION
# ============================================================

info "Verifico il token Hugging Face..."

HF_WHOAMI="$(
    curl \
        --fail \
        --silent \
        --show-error \
        --retry 3 \
        --connect-timeout 10 \
        -H "Authorization: Bearer ${HF_TOKEN}" \
        https://huggingface.co/api/whoami-v2
)" || die "HF token non valido o Hugging Face non raggiungibile."

HF_USER="$(echo "${HF_WHOAMI}" | jq -r '.name // empty')"

[[ -n "${HF_USER}" ]] ||
    die "Hugging Face non ha riconosciuto il token."

ok "HF autenticato come: ${HF_USER}"


# ============================================================
# NVIDIA CHECKS
# ============================================================

info "Controllo NVIDIA..."

command -v nvidia-smi >/dev/null 2>&1 ||
    die "nvidia-smi non trovato."

GPU_NAME="$(
    nvidia-smi \
        --query-gpu=name \
        --format=csv,noheader \
        | head -n1 \
        | xargs
)"

GPU_MEMORY="$(
    nvidia-smi \
        --query-gpu=memory.total \
        --format=csv,noheader,nounits \
        | head -n1 \
        | xargs
)"

GPU_DRIVER="$(
    nvidia-smi \
        --query-gpu=driver_version \
        --format=csv,noheader \
        | head -n1 \
        | xargs
)"

GPU_COUNT="$(
    nvidia-smi \
        --query-gpu=name \
        --format=csv,noheader \
        | wc -l
)"

echo
echo "GPU:       ${GPU_NAME}"
echo "VRAM:      ${GPU_MEMORY} MiB"
echo "Driver:    ${GPU_DRIVER}"
echo "GPU count: ${GPU_COUNT}"
echo

DRIVER_MAJOR="${GPU_DRIVER%%.*}"

if (( DRIVER_MAJOR < 580 )); then
    die "Driver ${GPU_DRIVER} troppo vecchio per lo stack CUDA 13 di SGLang 0.5.20. Serve ramo NVIDIA 580+."
fi

if (( GPU_MEMORY < 90000 )); then
    die "Questo profilo è progettato per ~96 GB VRAM. Rilevati solo ${GPU_MEMORY} MiB."
fi

if [[ "${GPU_NAME,,}" != *"pro 6000"* ]]; then
    warn "GPU non identificata come RTX PRO 6000."
    warn "Continuo, ma i parametri sono tarati per RTX PRO 6000 Blackwell 96GB."
else
    ok "RTX PRO 6000 rilevata."
fi


# ============================================================
# GLIBC CHECK
# SGLang 0.5.20 wheel uses manylinux glibc >= 2.34
# ============================================================

GLIBC_VERSION="$(
    ldd --version 2>&1 \
        | head -n1 \
        | grep -oE '[0-9]+\.[0-9]+' \
        | tail -n1
)"

info "glibc: ${GLIBC_VERSION}"

if [[ "$(printf '%s\n' "2.34" "${GLIBC_VERSION}" | sort -V | head -n1)" != "2.34" ]]; then
    die "glibc ${GLIBC_VERSION} troppo vecchia. Serve >= 2.34."
fi


# ============================================================
# DIRECTORY STRUCTURE
# ============================================================

info "Creo directory..."

mkdir -p \
    "${APP_DIR}" \
    "${MODEL_ROOT}" \
    "${HF_CACHE}" \
    "${LOG_DIR}"


# ============================================================
# DISK SPACE
# ============================================================

FREE_GIB="$(
    df -BG "${WORKSPACE}" \
        | awk 'NR==2 {gsub("G","",$4); print $4}'
)"

echo "Spazio libero su ${WORKSPACE}: ${FREE_GIB} GiB"

# Target 55.6 GB + draft 3.85 GB + Python/CUDA packages/cache/headroom.
if [[ ! -f "${TARGET_DIR}/model.safetensors.index.json" ]]; then

    if (( FREE_GIB < 80 )); then
        die "Spazio insufficiente. Consiglio almeno 80 GiB liberi prima del download BF16."
    fi

fi


# ============================================================
# PYTHON SELECTION
# ============================================================

info "Cerco Python compatibile..."

PYTHON_BIN=""

for candidate in \
    python3.12 \
    python3.11 \
    python3.13 \
    python3.10 \
    python3
do

    command -v "${candidate}" >/dev/null 2>&1 || continue

    VERSION="$(
        "${candidate}" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
    )"

    MAJOR="${VERSION%%.*}"
    MINOR="${VERSION##*.}"

    if (( MAJOR == 3 && MINOR >= 10 && MINOR <= 13 )); then
        PYTHON_BIN="$(command -v "${candidate}")"
        break
    fi

done

[[ -n "${PYTHON_BIN}" ]] ||
    die "Non trovo Python 3.10-3.13."

ok "$("${PYTHON_BIN}" --version)"


# ============================================================
# VENV
# ============================================================

if [[ ! -d "${VENV}" ]]; then

    info "Creo virtualenv..."
    "${PYTHON_BIN}" -m venv "${VENV}"

else

    info "Virtualenv già esistente."

fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export PYTHONNOUSERSITE=1

python -m pip install --upgrade \
    pip \
    setuptools \
    wheel \
    uv


# ============================================================
# INSTALL SGLANG
# ============================================================

info "Installazione SGLang ${SGLANG_VERSION}..."

uv pip install \
    --prerelease=allow \
    "sglang==${SGLANG_VERSION}"

uv pip install \
    --upgrade \
    "huggingface_hub[hf_xet]"


# ============================================================
# CUDA / TORCH / BLACKWELL CHECK
# ============================================================

info "Controllo runtime CUDA..."

python <<'PY'
import sys
import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)

if not torch.cuda.is_available():
    raise SystemExit(
        "ERRORE: torch.cuda.is_available() == False"
    )

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)

total_gib = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("GPU:", name)
print("Compute capability:", f"{major}.{minor}")
print("VRAM:", f"{total_gib:.2f} GiB")

if torch.version.cuda is None:
    raise SystemExit("PyTorch installato senza CUDA.")

cuda_major = int(torch.version.cuda.split(".")[0])

if cuda_major < 13:
    raise SystemExit(
        f"Serve CUDA runtime 13.x. "
        f"PyTorch riporta {torch.version.cuda}"
    )

if major != 12:
    print(
        "WARNING: questa configurazione è stata "
        "preparata per Blackwell SM120."
    )
PY


# ============================================================
# FLASHINFER CHECK
# ============================================================

info "Controllo FlashInfer..."

python <<'PY'
from importlib.metadata import version, PackageNotFoundError
from packaging.version import Version

try:
    import flashinfer
except Exception as e:
    raise SystemExit(
        f"Import FlashInfer fallito: {e}"
    )

try:
    v = version("flashinfer-python")
except PackageNotFoundError:
    try:
        v = version("flashinfer")
    except PackageNotFoundError:
        v = "unknown"

print("FlashInfer:", v)

if v != "unknown":
    if Version(v) <= Version("0.6.15.post1"):
        raise SystemExit(
            "FlashInfer troppo vecchio per il percorso "
            "FlashInfer + speculative Qwen3.8."
        )
PY

ok "FlashInfer importabile."


# ============================================================
# VERIFY SGLANG CLI FLAGS BEFORE 60GB DOWNLOAD
# ============================================================

info "Verifico che SGLang supporti tutti i flag richiesti..."

SGLANG_HELP="$(
    python -m sglang.launch_server --help 2>&1
)"

REQUIRED_FLAGS=(
    "--model-path"
    "--served-model-name"
    "--context-length"
    "--max-running-requests"
    "--kv-cache-dtype"
    "--attention-backend"
    "--chunked-prefill-size"
    "--mem-fraction-static"
    "--mamba-radix-cache-strategy"
    "--max-mamba-cache-size"
    "--mamba-ssm-dtype"
    "--speculative-algorithm"
    "--speculative-draft-model-path"
    "--speculative-num-draft-tokens"
    "--speculative-draft-attention-backend"
    "--cuda-graph-max-bs-decode"
    "--disable-prefill-cuda-graph"
    "--reasoning-parser"
    "--tool-call-parser"
    "--api-key"
)

for flag in "${REQUIRED_FLAGS[@]}"; do

    if ! grep -Fq -- "${flag}" <<< "${SGLANG_HELP}"; then
        die "SGLang ${SGLANG_VERSION} non riconosce ${flag}"
    fi

done

ok "Tutti i flag necessari sono presenti."


# ============================================================
# HF ENVIRONMENT
# ============================================================

export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"

export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_TELEMETRY=1


# ============================================================
# RESOLVE EXACT MODEL REVISIONS
# ============================================================

info "Risolvo revisioni esatte Hugging Face..."

export TARGET_REPO DRAFT_REPO

TARGET_SHA="$(
python <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

info = api.model_info(
    os.environ["TARGET_REPO"],
    revision="main",
)

print(info.sha)
PY
)"

DRAFT_SHA="$(
python <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])

info = api.model_info(
    os.environ["DRAFT_REPO"],
    revision="main",
)

print(info.sha)
PY
)"

[[ -n "${TARGET_SHA}" ]] ||
    die "Impossibile risolvere revision target."

[[ -n "${DRAFT_SHA}" ]] ||
    die "Impossibile risolvere revision DFlash."

echo "Target SHA: ${TARGET_SHA}"
echo "Draft SHA:  ${DRAFT_SHA}"


# ============================================================
# DOWNLOAD TARGET
# hf download resumes existing partial downloads automatically.
# ============================================================

info "Download target BF16..."
echo "${TARGET_REPO}"
echo "-> ${TARGET_DIR}"

hf download \
    "${TARGET_REPO}" \
    --revision "${TARGET_SHA}" \
    --local-dir "${TARGET_DIR}"


# ============================================================
# DOWNLOAD DFLASH2
# ============================================================

info "Download DFlash2..."
echo "${DRAFT_REPO}"
echo "-> ${DRAFT_DIR}"

hf download \
    "${DRAFT_REPO}" \
    --revision "${DRAFT_SHA}" \
    --local-dir "${DRAFT_DIR}"


# ============================================================
# MODEL INTEGRITY / CONFIG CHECK
# ============================================================

info "Verifica target BF16..."

export TARGET_DIR DRAFT_DIR

python <<'PY'
import json
import os
from pathlib import Path

from safetensors import safe_open


target = Path(os.environ["TARGET_DIR"])
draft = Path(os.environ["DRAFT_DIR"])


# ------------------------------------------------------------
# Target config
# ------------------------------------------------------------

config_file = target / "config.json"
index_file = target / "model.safetensors.index.json"

if not config_file.exists():
    raise SystemExit("Target config.json mancante.")

if not index_file.exists():
    raise SystemExit(
        "Target model.safetensors.index.json mancante."
    )

config = json.loads(config_file.read_text())

text = config.get("text_config", config)

dtype = text.get("dtype")
ctx = text.get("max_position_embeddings")
layers = text.get("num_hidden_layers")

print("Target dtype:", dtype)
print("Target context:", ctx)
print("Target layers:", layers)

if dtype != "bfloat16":
    raise SystemExit(
        f"Target non risulta BF16: dtype={dtype}"
    )

if ctx != 262144:
    raise SystemExit(
        f"Context dichiarato inatteso: {ctx}"
    )

if layers != 64:
    raise SystemExit(
        f"Numero layer target inatteso: {layers}"
    )


# ------------------------------------------------------------
# Validate every indexed shard exists
# ------------------------------------------------------------

index = json.loads(index_file.read_text())

files = sorted(set(index["weight_map"].values()))

print("Indexed target shards:", len(files))

if len(files) != 18:
    raise SystemExit(
        f"Attesi 18 shard; trovati {len(files)}"
    )

for name in files:

    path = target / name

    if not path.exists():
        raise SystemExit(
            f"Shard mancante: {name}"
        )

    if path.stat().st_size < 1024 * 1024:
        raise SystemExit(
            f"Shard sospettosamente piccolo: {name}"
        )

    # Parse safetensors header.
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    if not keys:
        raise SystemExit(
            f"Safetensors vuoto/corrotto: {name}"
        )


# ------------------------------------------------------------
# Draft
# ------------------------------------------------------------

draft_config_file = draft / "config.json"
draft_model_file = draft / "model.safetensors"

if not draft_config_file.exists():
    raise SystemExit("DFlash config.json mancante.")

if not draft_model_file.exists():
    raise SystemExit("DFlash model.safetensors mancante.")

dc = json.loads(draft_config_file.read_text())

block = dc.get("dflash_config", {}).get("block_size")
draft_layers = dc.get("num_hidden_layers")
draft_ctx = dc.get("max_position_embeddings")

print("DFlash block:", block)
print("DFlash layers:", draft_layers)
print("DFlash context:", draft_ctx)

if block != 8:
    raise SystemExit(
        f"DFlash block_size inatteso: {block}"
    )

if draft_layers != 5:
    raise SystemExit(
        f"DFlash dovrebbe avere 5 layer; trovati {draft_layers}"
    )

if draft_ctx != 262144:
    raise SystemExit(
        f"DFlash max_position_embeddings inatteso: {draft_ctx}"
    )

if draft_model_file.stat().st_size < 3_000_000_000:
    raise SystemExit(
        "DFlash model.safetensors sembra incompleto."
    )

with safe_open(
    draft_model_file,
    framework="pt",
    device="cpu",
) as f:

    if not list(f.keys()):
        raise SystemExit("DFlash safetensors vuoto.")

print()
print("Model integrity check: PASS")
PY

ok "Target e DFlash verificati."


# ============================================================
# API KEY
# ============================================================

if [[ -f "${ENV_FILE}" ]]; then

    OLD_API_KEY="$(
        grep '^SGLANG_API_KEY=' "${ENV_FILE}" \
            | head -n1 \
            | cut -d= -f2- \
            || true
    )"

else
    OLD_API_KEY=""
fi

if [[ -n "${OLD_API_KEY}" ]]; then

    API_KEY="${OLD_API_KEY}"
    info "Mantengo API key già esistente."

else

    API_KEY="$(openssl rand -hex 32)"
    info "Generata nuova API key."

fi


# ============================================================
# SAFE BIND ADDRESS
#
# Prefer Tailscale when available.
# Otherwise localhost.
# ============================================================

if [[ -n "${BIND_HOST:-}" ]]; then

    SERVER_HOST="${BIND_HOST}"

elif command -v tailscale >/dev/null 2>&1; then

    TS_IP="$(
        tailscale ip -4 2>/dev/null \
            | head -n1 \
            || true
    )"

    if [[ -n "${TS_IP}" ]]; then
        SERVER_HOST="${TS_IP}"
    else
        SERVER_HOST="127.0.0.1"
    fi

else

    SERVER_HOST="127.0.0.1"

fi


# ============================================================
# .ENV
# HF TOKEN INTENTIONALLY NOT SAVED
# ============================================================

umask 077

cat > "${ENV_FILE}" <<EOF
# Qwen3.8 SGLang runtime configuration

SGLANG_API_KEY=${API_KEY}

HOST=${SERVER_HOST}
PORT=${PORT}

TARGET_DIR=${TARGET_DIR}
DRAFT_DIR=${DRAFT_DIR}

TARGET_SHA=${TARGET_SHA}
DRAFT_SHA=${DRAFT_SHA}

CONTEXT_LENGTH=${CONTEXT_LENGTH}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS}
MAX_MAMBA_CACHE_SIZE=${MAX_MAMBA_CACHE_SIZE}

MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}
CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE}

CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

chmod 600 "${ENV_FILE}"


# ============================================================
# VERSION RECORD
# ============================================================

cat > "${APP_DIR}/versions.txt" <<EOF
Generated: $(date -Iseconds)

Target repo:
${TARGET_REPO}

Target revision:
${TARGET_SHA}

Draft repo:
${DRAFT_REPO}

Draft revision:
${DRAFT_SHA}

SGLang:
${SGLANG_VERSION}

GPU:
${GPU_NAME}

Driver:
${GPU_DRIVER}
EOF


# ============================================================
# COMMON RUNTIME
# ============================================================

cat > "${APP_DIR}/common.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/.env"
source "${APP_DIR}/.venv/bin/activate"

PID_FILE="${APP_DIR}/server.pid"
LOG_DIR="${APP_DIR}/logs"
LOG_FILE="${LOG_DIR}/server.log"

mkdir -p "${LOG_DIR}"

export \
    CUDA_VISIBLE_DEVICES \
    PYTORCH_CUDA_ALLOC_CONF

health_host() {

    if [[ "${HOST}" == "0.0.0.0" ]]; then
        echo "127.0.0.1"
    else
        echo "${HOST}"
    fi
}

check_port() {

    if command -v ss >/dev/null 2>&1; then

        if ss -ltnH \
            | awk '{print $4}' \
            | grep -Eq ":${PORT}$"; then

            echo "ERROR: porta ${PORT} già occupata."
            echo
            ss -ltnp | grep ":${PORT}" || true
            exit 1
        fi
    fi
}

check_pid() {

    if [[ -f "${PID_FILE}" ]]; then

        PID="$(cat "${PID_FILE}")"

        if kill -0 "${PID}" 2>/dev/null; then
            echo "Server già attivo. PID=${PID}"
            exit 0
        fi

        rm -f "${PID_FILE}"
    fi
}

wait_ready() {

    LOCAL_HOST="$(health_host)"

    echo
    echo "Controllo avvio server..."

    for _ in $(seq 1 600); do

        if curl \
            --silent \
            --fail \
            --max-time 2 \
            -H "Authorization: Bearer ${SGLANG_API_KEY}" \
            "http://${LOCAL_HOST}:${PORT}/v1/models" \
            >/dev/null 2>&1; then

            echo
            echo "========================================"
            echo " SERVER READY"
            echo "========================================"
            echo
            echo "Endpoint:"
            echo "  http://${HOST}:${PORT}/v1"
            echo
            echo "Model:"
            echo "  qwen"
            echo
            echo "API key:"
            echo "  ${SGLANG_API_KEY}"
            echo
            echo "Log:"
            echo "  ${LOG_FILE}"
            echo

            # Show memory / scheduler info if present
            grep -Ei \
                'max_running_requests|max_total_tokens|kv cache|mamba|dflash' \
                "${LOG_FILE}" \
                | tail -n 30 \
                || true

            return 0
        fi

        PID="$(
            cat "${PID_FILE}" 2>/dev/null \
                || true
        )"

        if [[ -n "${PID}" ]] &&
           ! kill -0 "${PID}" 2>/dev/null; then

            echo
            echo "ERROR: SGLang è crashato."
            echo

            tail -n 150 "${LOG_FILE}" || true
            exit 1
        fi

        sleep 2
    done

    echo
    echo "ERROR: health-check timeout."
    tail -n 150 "${LOG_FILE}" || true
    exit 1
}
EOF

chmod +x "${APP_DIR}/common.sh"


# ============================================================
# FAST PROFILE - DFLASH2
# ============================================================

cat > "${APP_DIR}/start.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/common.sh"

check_pid
check_port

echo
echo "============================================"
echo " Qwen3.8 Huihui BF16"
echo " SGLang + FP8 KV + DFlash2"
echo "============================================"
echo
echo "Context per request: ${CONTEXT_LENGTH}"
echo "Concurrency:         ${MAX_RUNNING_REQUESTS}"
echo "Target:              ${TARGET_DIR}"
echo "Draft:               ${DRAFT_DIR}"
echo

: > "${LOG_FILE}"

CMD=(
    python -m sglang.launch_server

    --model-path "${TARGET_DIR}"
    --served-model-name qwen

    --host "${HOST}"
    --port "${PORT}"
    --api-key "${SGLANG_API_KEY}"

    --tp-size 1

    --context-length "${CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"

    --kv-cache-dtype fp8_e4m3

    --attention-backend flashinfer
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"

    --mem-fraction-static "${MEM_FRACTION_STATIC}"

    --mamba-radix-cache-strategy extra_buffer_lazy
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"

    # Qwen checkpoint declares FP32 SSM state.
    # Keep it FP32 for the quality-first profile.
    --mamba-ssm-dtype float32

    # Two concurrent decode batches.
    --cuda-graph-max-bs-decode 2

    # DFlash2
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${DRAFT_DIR}"
    --speculative-num-draft-tokens 8
    --speculative-draft-attention-backend flashinfer

    # Current robust workaround for DFlash/prefill graph.
    # Decode throughput is unaffected.
    --disable-prefill-cuda-graph

    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --sampling-defaults model
)

nohup "${CMD[@]}" \
    > "${LOG_FILE}" \
    2>&1 &

PID=$!

echo "${PID}" > "${PID_FILE}"

echo "PID: ${PID}"

wait_ready
EOF

chmod +x "${APP_DIR}/start.sh"


# ============================================================
# SAFE PROFILE - NO SPECULATIVE DECODING
# ============================================================

cat > "${APP_DIR}/start-safe.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/common.sh"

check_pid
check_port

echo
echo "============================================"
echo " Qwen3.8 Huihui BF16 BASELINE"
echo " DFlash OFF"
echo "============================================"
echo

: > "${LOG_FILE}"

CMD=(
    python -m sglang.launch_server

    --model-path "${TARGET_DIR}"
    --served-model-name qwen

    --host "${HOST}"
    --port "${PORT}"
    --api-key "${SGLANG_API_KEY}"

    --tp-size 1

    --context-length "${CONTEXT_LENGTH}"
    --max-running-requests "${MAX_RUNNING_REQUESTS}"

    --kv-cache-dtype fp8_e4m3

    --attention-backend flashinfer
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"

    --mem-fraction-static "${MEM_FRACTION_STATIC}"

    --mamba-radix-cache-strategy extra_buffer_lazy
    --max-mamba-cache-size "${MAX_MAMBA_CACHE_SIZE}"
    --mamba-ssm-dtype float32

    --cuda-graph-max-bs-decode 2

    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --sampling-defaults model
)

nohup "${CMD[@]}" \
    > "${LOG_FILE}" \
    2>&1 &

PID=$!

echo "${PID}" > "${PID_FILE}"

echo "PID: ${PID}"

wait_ready
EOF

chmod +x "${APP_DIR}/start-safe.sh"


# ============================================================
# STOP
# ============================================================

cat > "${APP_DIR}/stop.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

PID_FILE="${APP_DIR}/server.pid"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "Server non risulta attivo."
    exit 0
fi

PID="$(cat "${PID_FILE}")"

if ! kill -0 "${PID}" 2>/dev/null; then

    echo "PID ${PID} non esistente."
    rm -f "${PID_FILE}"
    exit 0
fi

echo "Stop PID ${PID}..."

kill "${PID}"

for _ in $(seq 1 30); do

    if ! kill -0 "${PID}" 2>/dev/null; then

        rm -f "${PID_FILE}"
        echo "Server fermato."
        exit 0
    fi

    sleep 1
done

echo "SIGTERM timeout -> SIGKILL"

kill -9 "${PID}" 2>/dev/null || true

rm -f "${PID_FILE}"

echo "Server fermato."
EOF

chmod +x "${APP_DIR}/stop.sh"


# ============================================================
# STATUS
# ============================================================

cat > "${APP_DIR}/status.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/.env"

PID_FILE="${APP_DIR}/server.pid"
LOG_FILE="${APP_DIR}/logs/server.log"

echo
echo "========== PROCESS =========="

if [[ -f "${PID_FILE}" ]]; then

    PID="$(cat "${PID_FILE}")"

    if kill -0 "${PID}" 2>/dev/null; then
        echo "RUNNING PID=${PID}"
    else
        echo "STALE PID FILE (${PID})"
    fi

else
    echo "STOPPED"
fi

echo
echo "========== GPU =============="

nvidia-smi \
    --query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,power.draw \
    --format=csv

echo
echo "========== CONFIG ==========="

echo "Context:       ${CONTEXT_LENGTH}"
echo "Concurrency:   ${MAX_RUNNING_REQUESTS}"
echo "Mamba slots:   ${MAX_MAMBA_CACHE_SIZE}"
echo "Host:          ${HOST}"
echo "Port:          ${PORT}"

echo
echo "========== API =============="

if [[ "${HOST}" == "0.0.0.0" ]]; then
    TEST_HOST="127.0.0.1"
else
    TEST_HOST="${HOST}"
fi

curl \
    --silent \
    --show-error \
    --max-time 3 \
    -H "Authorization: Bearer ${SGLANG_API_KEY}" \
    "http://${TEST_HOST}:${PORT}/v1/models" \
    | python -m json.tool \
    || echo "API non raggiungibile."

echo
echo "========== MEMORY/SCHEDULER =="

grep -Ei \
    'max_running_requests|max_total_tokens|kv cache|mamba|dflash' \
    "${LOG_FILE}" \
    | tail -n 50 \
    || true

echo
echo "========== LAST LOG ========="

tail -n 40 "${LOG_FILE}" 2>/dev/null || true
EOF

chmod +x "${APP_DIR}/status.sh"


# ============================================================
# DOCTOR
# ============================================================

cat > "${APP_DIR}/doctor.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/.env"
source "${APP_DIR}/.venv/bin/activate"

echo
echo "========== NVIDIA =========="
nvidia-smi

echo
echo "========== PYTHON =========="

python <<'PY'
import torch
import sglang

print("SGLang:", getattr(sglang, "__version__", "unknown"))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "========== FLASHINFER ======"

python <<'PY'
import flashinfer
from importlib.metadata import version

try:
    print("FlashInfer:", version("flashinfer-python"))
except Exception:
    print("FlashInfer import: OK")
PY

echo
echo "========== TARGET =========="

python - <<PY
import json
from pathlib import Path

p = Path("${TARGET_DIR}") / "config.json"
c = json.loads(p.read_text())

t = c.get("text_config", c)

print("dtype:", t.get("dtype"))
print("context:", t.get("max_position_embeddings"))
print("layers:", t.get("num_hidden_layers"))
print("mamba_ssm_dtype:", t.get("mamba_ssm_dtype"))
PY

echo
echo "========== FILE SIZE ========"

du -sh "${TARGET_DIR}"
du -sh "${DRAFT_DIR}"

echo
echo "========== DISK ============="

df -h /workspace

echo
echo "Doctor complete."
EOF

chmod +x "${APP_DIR}/doctor.sh"


# ============================================================
# API TEST
# ============================================================

cat > "${APP_DIR}/test.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/.env"

if [[ "${HOST}" == "0.0.0.0" ]]; then
    TEST_HOST="127.0.0.1"
else
    TEST_HOST="${HOST}"
fi

curl \
    --silent \
    --show-error \
    "http://${TEST_HOST}:${PORT}/v1/chat/completions" \
    -H "Authorization: Bearer ${SGLANG_API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen",
        "messages": [
            {
                "role": "user",
                "content": "Write a small C binary-search implementation. Be concise."
            }
        ],
        "temperature": 0,
        "max_tokens": 512
    }' \
    | python -m json.tool
EOF

chmod +x "${APP_DIR}/test.sh"


# ============================================================
# SIMPLE TOK/S BENCH
# ============================================================

cat > "${APP_DIR}/bench.sh" <<'EOF'
#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${APP_DIR}/.env"
source "${APP_DIR}/.venv/bin/activate"

if [[ "${HOST}" == "0.0.0.0" ]]; then
    TEST_HOST="127.0.0.1"
else
    TEST_HOST="${HOST}"
fi

export TEST_HOST

python <<'PY'
import json
import os
import time
import urllib.request

host = os.environ["TEST_HOST"]
port = os.environ["PORT"]
key = os.environ["SGLANG_API_KEY"]

url = f"http://{host}:{port}/v1/chat/completions"

payload = {
    "model": "qwen",
    "messages": [{
        "role": "user",
        "content": (
            "Write a detailed explanation of how a lock-free "
            "ring buffer works in C, including memory ordering."
        ),
    }],
    "temperature": 0,
    "max_tokens": 1024,
}

data = json.dumps(payload).encode()

req = urllib.request.Request(
    url,
    data=data,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    },
)

start = time.perf_counter()

with urllib.request.urlopen(req) as response:
    result = json.loads(response.read())

elapsed = time.perf_counter() - start

usage = result.get("usage", {})
tokens = usage.get("completion_tokens")

print(f"Elapsed: {elapsed:.2f}s")

if tokens:
    print(f"Completion tokens: {tokens}")
    print(f"Client-observed: {tokens / elapsed:.2f} tok/s")
else:
    print("Server non ha restituito completion_tokens.")
PY
EOF

chmod +x "${APP_DIR}/bench.sh"


# ============================================================
# FINAL RUNTIME VERIFICATION
# ============================================================

info "Verifica finale ambiente..."

python <<'PY'
import torch
import sglang

print()
print("SGLang:", getattr(sglang, "__version__", "unknown"))
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
PY


# ============================================================
# REMOVE HF TOKEN FROM ENV AS MUCH AS POSSIBLE
# ============================================================

unset HF_TOKEN


# ============================================================
# DONE
# ============================================================

echo
echo "============================================================"
echo " SETUP COMPLETATO"
echo "============================================================"
echo
echo "Target BF16:"
echo "  ${TARGET_DIR}"
echo
echo "DFlash2:"
echo "  ${DRAFT_DIR}"
echo
echo "SGLang:"
echo "  ${SGLANG_VERSION}"
echo
echo "Context per request:"
echo "  ${CONTEXT_LENGTH}"
echo
echo "Max utenti/richieste concorrenti:"
echo "  ${MAX_RUNNING_REQUESTS}"
echo
echo "Bind:"
echo "  ${SERVER_HOST}:${PORT}"
echo
echo "API key:"
echo "  ${API_KEY}"
echo
echo "La key è salvata in:"
echo "  ${ENV_FILE}"
echo
echo "HF TOKEN NON è stato salvato."
echo
echo "------------------------------------------------------------"
echo "AVVIO DFLASH2:"
echo
echo "  cd ${APP_DIR}"
echo "  ./start.sh"
echo
echo "BASELINE SENZA DFLASH:"
echo
echo "  ./start-safe.sh"
echo
echo "STATUS:"
echo
echo "  ./status.sh"
echo
echo "DOCTOR:"
echo
echo "  ./doctor.sh"
echo
echo "TEST:"
echo
echo "  ./test.sh"
echo
echo "BENCH:"
echo
echo "  ./bench.sh"
echo
echo "STOP:"
echo
echo "  ./stop.sh"
echo
echo "LOG LIVE:"
echo
echo "  tail -f ${LOG_FILE}"
echo
echo "============================================================"
