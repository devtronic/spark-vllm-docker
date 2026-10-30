#!/bin/bash
set -euo pipefail

# Runtime overlays for the GLM-5.3-Flash DFlash2 SPEED recipe (lane B) on the
# glm53-roce image (knapcio Dockerfile.roce on the tonyd2wild v11 base).
# Mirrors the bind-mounts of knapcio's start.sh @770d115: the fixed site-packages
# files (always-mounted + the GLM5NEXT_PATCH/GLM_KPOOL_FIX/GLM_ARGMAX_CLAMP sets
# that profiles/current.env arms) and the PYTHONPATH overlay tree at /overlay.

PREFIX="[glm53-dflash2-speed]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_MOD_DIR="$MOD_DIR/overlay"
EXPECTED_VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate and validate vLLM." >&2
    exit 1
fi

if [ -z "${VLLM_PACKAGE_ROOT:-}" ]; then
    VLLM_PACKAGE_ROOT=$(python3 - <<'PY'
import importlib.util

spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
    )
fi

if [ ! -d "$VLLM_MOD_DIR" ]; then
    echo "$PREFIX overlay tree missing from mod: $VLLM_MOD_DIR" >&2
    exit 1
fi

# The launcher mounts 13 overlay files over the vLLM tree (the five always-on
# files plus the three opt-in sets that profiles/current.env enables). The
# kv_cache_coordinator.py replacement already carries the #18 prefix-cache
# repair, so no patch script is needed on this lane.
MOUNTS=(
    "sparse_attn_indexer_kpool.py:model_executor/layers/sparse_attn_indexer_kpool.py"
    "glm47_moe.py:parser/glm47_moe.py"
    "abstract_parser.py:parser/abstract_parser.py"
    "kv_cache_coordinator.py:v1/core/kv_cache_coordinator.py"
    "E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json:model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json"
    "glm5next_kda.py:models/glm5next/nvidia/kda.py"
    "glm5next_model.py:models/glm5next/nvidia/model.py"
    "glm5next_kpool_compress.py:models/glm5next/nvidia/ops/kpool_compress.py"
    "glm5next_attention.py:models/glm5next/nvidia/attention.py"
    "mla_indexer.py:v1/attention/backends/mla/indexer.py"
    "mamba_hybrid.py:v1/worker/gpu/model_states/mamba_hybrid.py"
    "gumbel.py:v1/worker/gpu/sample/gumbel.py"
    "rejection_sampler_utils.py:v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
)

if [ ! -f "$VLLM_PACKAGE_ROOT/model_executor/models/qwen3_dflash2.py" ]; then
    echo "$PREFIX vLLM tree has no DFlash2 model (model_executor/models/qwen3_dflash2.py)." >&2
    echo "$PREFIX Expected the glm53-roce image (knapcio Dockerfile.roce on the" >&2
    echo "$PREFIX tonyd2wild v11 base). Refusing to patch a foreign image." >&2
    exit 1
fi
if [ "$VLLM_PACKAGE_ROOT" != "$EXPECTED_VLLM_ROOT" ]; then
    echo "$PREFIX vLLM lives at $VLLM_PACKAGE_ROOT, not $EXPECTED_VLLM_ROOT." >&2
    echo "$PREFIX The overlay targets are written for the dist-packages layout; refusing." >&2
    exit 1
fi

for pair in "${MOUNTS[@]}"; do
    src="$VLLM_MOD_DIR/${pair%%:*}"
    dst="$VLLM_PACKAGE_ROOT/${pair#*:}"
    if [ ! -f "$src" ]; then
        echo "$PREFIX overlay source missing: $src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
done
echo "$PREFIX Installed ${#MOUNTS[@]} overlay files into the vLLM tree."

# PYTHONPATH tree: the adaptive/lever scheduler, hooks and sitecustomize that
# the recipe env (PYTHONPATH=/overlay/overlay) and the GLM_* switches consume.
mkdir -p /overlay
rm -rf /overlay/overlay /overlay/profiles
cp -a "$VLLM_MOD_DIR" /overlay/overlay
cp -a "$MOD_DIR/profiles" /overlay/profiles
if [ ! -f /overlay/profiles/levers_policy.json ]; then
    echo "$PREFIX /overlay/profiles/levers_policy.json missing (SPEC_PROBE_CONTROL target)." >&2
    exit 1
fi
python3 -m compileall -q /overlay/overlay >/dev/null
echo "$PREFIX Staged /overlay/overlay tree and levers policy."

# Compile-check the Python files written into the vLLM tree (the MoE tune JSON
# entry is data, skip it).
for pair in "${MOUNTS[@]}"; do
    dst="$VLLM_PACKAGE_ROOT/${pair#*:}"
    case "$dst" in
        *.py) python3 -m py_compile "$dst" ;;
    esac
done
echo "$PREFIX Compile checks passed."

# The RoCE runtime (b12x + glm_roce shim) is baked into the glm53-roce image
# itself; the GLM_ROCE_ALLREDUCE=1 switch lives in the recipe env. Warn (do
# not fail) if a plain base image is used without it.
if ! python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("glm_roce") else 1)
PY
then
    echo "$PREFIX note: glm_roce shim not found -> RoCE all-reduce OFF (NCCL)." >&2
    echo "$PREFIX note: build the intended image with knapcio Dockerfile.roce (see recipe header)." >&2
fi

echo "$PREFIX GLM-5.3-Flash DFlash2 speed-stack overlays are in place."
