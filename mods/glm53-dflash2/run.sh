#!/bin/bash
set -euo pipefail

# Runtime overlays for the GLM-5.3-Flash + DFlash2 recipe (lane A) on the
# ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 image. Applies the
# three runtime pieces the published tag lacks and the optional RoCE
# all-reduce, mirroring launch-glm53-vllm-tp2-dflash2.sh's bind-mounts.

PREFIX="[glm53-dflash2]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KPOOL_FILE="$MOD_DIR/sparse_attn_indexer_kpool_sm121.py"
PREFIX_PATCH="$MOD_DIR/patch_prefix_cache_draft_group.py"
CHAT_TEMPLATE="$MOD_DIR/chat_template_mm.jinja"
TEMPLATE_DIR="/opt/glm53-dflash2"
# The prefix-cache patch script hardcodes this absolute path; refuse to run
# on an image whose vLLM lives elsewhere so it cannot edit the wrong tree.
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

for src in "$KPOOL_FILE" "$PREFIX_PATCH" "$CHAT_TEMPLATE"; do
    if [ ! -f "$src" ]; then
        echo "$PREFIX required file missing from mod: $src" >&2
        exit 1
    fi
done

# Fail loudly on a mismatched image: the v11-dflash2 tag carries the DFlash2
# model + speculator, and the prefix-cache patch anchors are written for the
# dist-packages layout of this image chain.
if [ ! -f "$VLLM_PACKAGE_ROOT/model_executor/models/qwen3_dflash2.py" ]; then
    echo "$PREFIX vLLM tree has no DFlash2 model (model_executor/models/qwen3_dflash2.py)." >&2
    echo "$PREFIX Expected ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2." >&2
    exit 1
fi
if [ "$VLLM_PACKAGE_ROOT" != "$EXPECTED_VLLM_ROOT" ]; then
    echo "$PREFIX vLLM lives at $VLLM_PACKAGE_ROOT, not $EXPECTED_VLLM_ROOT." >&2
    echo "$PREFIX The prefix-cache patch hardcodes the dist-packages path; refusing to patch." >&2
    exit 1
fi

# 1. SM121 kpool top-k fix. Without it the engine dies on any decode past
#    ~24K context (GLM repo docs/SM121-CRASH-FORENSICS-2026-08-27.md).
KPOOL_TARGET="$VLLM_PACKAGE_ROOT/model_executor/layers/sparse_attn_indexer_kpool.py"
if [ ! -f "$KPOOL_TARGET" ]; then
    echo "$PREFIX kpool file not found in vLLM tree: $KPOOL_TARGET" >&2
    exit 1
fi
cp "$KPOOL_FILE" "$KPOOL_TARGET"
python3 -m py_compile "$KPOOL_TARGET"
echo "$PREFIX Installed SM121 kpool top-k fix."

# 2. #18 prefix-cache repair for the drafter group. The published v11 tag was
#    built before it; without it prefix-cache hits stay at 0 with DFlash2 on.
COORDINATOR="$VLLM_PACKAGE_ROOT/v1/core/kv_cache_coordinator.py"
if ! [ -f "$COORDINATOR" ]; then
    echo "$PREFIX coordinator file not found: $COORDINATOR" >&2
    exit 1
fi
if grep -q "_glm53_is_draft_swa_spec" "$COORDINATOR"; then
    echo "$PREFIX Prefix-cache repair (#18) is already present; skipping."
else
    python3 "$PREFIX_PATCH"
    if ! grep -q "_glm53_is_draft_swa_spec" "$COORDINATOR"; then
        echo "$PREFIX Prefix-cache patch self-check failed." >&2
        exit 1
    fi
    python3 -m py_compile "$COORDINATOR"
    echo "$PREFIX Applied #18 prefix-cache repair."
fi

# 3. Vision chat template (image requests 500 without it).
mkdir -p "$TEMPLATE_DIR"
cp "$CHAT_TEMPLATE" "$TEMPLATE_DIR/chat_template_mm.jinja"
echo "$PREFIX Installed chat template at $TEMPLATE_DIR/chat_template_mm.jinja."

# 4. b12x RoCE one-shot all-reduce (optional; +5-18% aggregate). The adapter
#    files are shipped in the mod, but the b12x runtime they import is not
#    part of this image (the source launcher bind-mounts it from a per-node
#    bundle built by speed-night-2026-09-18/roce/build-roce-bundle.sh from a
#    DS4 image). Apply only when the runtime is present; otherwise the image
#    keeps stock NCCL all-reduce and the VLLM_ROCE_*/B12X_ROCE_* env vars are
#    inert. All-or-nothing: a half-applied adapter breaks the boot.
has_roce_runtime() {
    python3 - <<'PY'
import importlib.util
import sys

sys.exit(0 if importlib.util.find_spec("b12x") and __import__("os").path.isdir("/opt/b12x-roce") else 1)
PY
}

ROCE_TARGETS=(
    "b12x_roce_all_reduce.py:$VLLM_PACKAGE_ROOT/distributed/device_communicators/b12x_roce_all_reduce.py"
    "cuda_communicator.py:$VLLM_PACKAGE_ROOT/distributed/device_communicators/cuda_communicator.py"
    "parallel_state.py:$VLLM_PACKAGE_ROOT/distributed/parallel_state.py"
    "envs.py:$VLLM_PACKAGE_ROOT/envs.py"
    "gpu_worker.py:$VLLM_PACKAGE_ROOT/v1/worker/gpu_worker.py"
)

if has_roce_runtime; then
    for pair in "${ROCE_TARGETS[@]}"; do
        src="$MOD_DIR/roce/${pair%%:*}"
        dst="${pair#*:}"
        if [ ! -f "$src" ]; then
            echo "$PREFIX RoCE adapter file missing: $src" >&2
            exit 1
        fi
        cp "$src" "$dst"
    done
    for pair in "${ROCE_TARGETS[@]}"; do
        python3 -m py_compile "${pair#*:}"
    done
    echo "$PREFIX Applied b12x RoCE all-reduce adapter."
else
    echo "$PREFIX note: b12x runtime not in this image -> RoCE all-reduce OFF (NCCL)." >&2
    echo "$PREFIX note: to enable, bake the bundle from speed-night-2026-09-18/roce/build-roce-bundle.sh into the image (b12x, b12x-1.3.0.dist-info, /opt/b12x-roce)." >&2
fi

echo "$PREFIX GLM-5.3-Flash DFlash2 overlays are in place."
