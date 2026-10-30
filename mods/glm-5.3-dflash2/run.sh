#!/bin/bash
set -euo pipefail

# This mod used to carry a large overlay (DFlash2 drafter model, speculator
# module, GLM-5.3 drafter-group KV layout patch, EAGLE aux-capture glue,
# prefix-cache repair, and the SM121 top-k indexer shim). Upstream vLLM has
# since absorbed almost all of it: DFlash2DraftModel + DFlash2Qwen3ForCausalLM
# are natively registered, the DFlash2 speculator ships at
# v1/worker/gpu/spec_decode/dflash2/speculator.py, Glm5Next already carries
# EagleModelMixin + SupportsEagle3 with aux-hidden-state capture, and the
# GLM-5-Next KV grouping has been refactored out of kv_cache_utils.py. The
# only remaining mod responsibility is staging the multimodal chat template
# into the launch cwd so the recipe can reference it as
# --chat-template chat_template_mm.jinja (same pattern mods/diffusiongemma
# uses). If the recipe exposes a real correctness issue, add a focused patch
# here against the current source rather than restoring the full overlay.

PREFIX="[glm-5.3-dflash2]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_DIR="$PWD"

CHAT_TEMPLATE_TARGET_DIR="${WORKSPACE_DIR:-$START_DIR}"
cp "$MOD_DIR/chat_template_mm.jinja" "$CHAT_TEMPLATE_TARGET_DIR/chat_template_mm.jinja"
echo "$PREFIX chat template staged at $CHAT_TEMPLATE_TARGET_DIR/chat_template_mm.jinja"
