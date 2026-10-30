#!/bin/bash
set -euo pipefail

PREFIX="[glm-5.3-dflash2]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$PWD"

if ! command -v python3 >/dev/null 2>&1; then
    echo "$PREFIX python3 is required to locate and patch vLLM." >&2
    exit 1
fi

# Discover vLLM package root without importing it. Mirrors mods/radixark-dspark
# so cluster-boot CUDA init cannot race on the mod-application step.
VLLM_ROOT=$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM package is not installed for the active Python interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
)
export VLLM_ROOT
echo "$PREFIX vLLM package root: $VLLM_ROOT"

# Required paths. Hybrid KV layout + prefix-cache fixes anchor on stable
# upstream files; the glm5next target model has moved between vLLM versions,
# so discover it rather than hardcode.
KV_UTILS_FILE="$VLLM_ROOT/v1/core/kv_cache_utils.py"
KV_COORD_FILE="$VLLM_ROOT/v1/core/kv_cache_coordinator.py"
REGISTRY_FILE="$VLLM_ROOT/model_executor/models/registry.py"

for f in "$KV_UTILS_FILE" "$KV_COORD_FILE" "$REGISTRY_FILE"; do
    if [ ! -f "$f" ]; then
        echo "$PREFIX required vLLM source not found: $f" >&2
        echo "$PREFIX The installed vLLM does not match the layout this mod patches." >&2
        exit 1
    fi
done

GLM_MODEL_FILE=$(find "$VLLM_ROOT" -type f -path '*glm5next*/model.py' 2>/dev/null | head -n 1)
if [ -z "$GLM_MODEL_FILE" ]; then
    echo "$PREFIX glm5next target model file not found under $VLLM_ROOT." >&2
    echo "$PREFIX This mod requires a vLLM build with GLM-5-Next support (Glm5NextForConditionalGeneration)." >&2
    exit 1
fi
echo "$PREFIX glm5next model file: $GLM_MODEL_FILE"

# Stage the new modules into site-packages (idempotent; cp -f overwrites).
install -m 0644 "$MOD_DIR/qwen3_dflash2.py" \
    "$VLLM_ROOT/model_executor/models/qwen3_dflash2.py"
install -m 0644 "$MOD_DIR/sparse_attn_indexer_kpool.py" \
    "$VLLM_ROOT/model_executor/layers/sparse_attn_indexer_kpool.py"
mkdir -p "$VLLM_ROOT/v1/worker/gpu/spec_decode/dflash2"
install -m 0644 "$MOD_DIR/dflash2/__init__.py" \
    "$VLLM_ROOT/v1/worker/gpu/spec_decode/dflash2/__init__.py"
install -m 0644 "$MOD_DIR/dflash2/speculator.py" \
    "$VLLM_ROOT/v1/worker/gpu/spec_decode/dflash2/speculator.py"

# Apply the four anchored patches. Each self-skips when its marker is present,
# so re-running the mod is safe and prints only [skip] lines.
python3 "$MOD_DIR/patch_registry_and_select.py" "$VLLM_ROOT"
python3 "$MOD_DIR/patch_glm_aux_capture.py" --model-file "$GLM_MODEL_FILE"
python3 "$MOD_DIR/patch_glm5_drafter_group.py" --kv-file "$KV_UTILS_FILE"
python3 "$MOD_DIR/patch_prefix_cache_draft_group.py"

# Stage the chat template beside the launch cwd so recipes can reference it
# as --chat-template chat_template_mm.jinja (same pattern mods/diffusiongemma
# uses for fixed_chat_template.jinja).
CHAT_TEMPLATE_TARGET_DIR="${WORKSPACE_DIR:-$START_DIR}"
cp "$MOD_DIR/chat_template_mm.jinja" "$CHAT_TEMPLATE_TARGET_DIR/chat_template_mm.jinja"
echo "$PREFIX chat template staged at $CHAT_TEMPLATE_TARGET_DIR/chat_template_mm.jinja"

# Postcondition: DFlash2DraftModel must now be registered. Grep source text to
# avoid importing vllm here.
if ! grep -q '"DFlash2DraftModel"' "$REGISTRY_FILE"; then
    echo "$PREFIX DFlash2DraftModel is not registered after patching." >&2
    exit 1
fi
echo "$PREFIX DFlash2 drafter is registered; GLM-5.3 drafter-group KV layout + prefix-cache fix applied."
