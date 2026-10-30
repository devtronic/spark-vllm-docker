# GLM-5.3-Flash DFlash2 chat-template stage

Pair with `recipes/glm-5.3-flash-dflash2.yaml`. At launch, `run.sh` stages
`chat_template_mm.jinja` into the container's launch cwd so the recipe can
reference it as `--chat-template chat_template_mm.jinja` — same pattern
`mods/diffusiongemma` uses for `fixed_chat_template.jinja`.

## Why this mod is tiny

The sibling repo `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/overlay-dflash2/`
originally bundled a 2000+ line overlay (DFlash2 drafter model class,
speculator module, GLM-5.3 drafter-group KV layout fix, EAGLE aux-capture
glue, prefix-cache repair, SM121 top-k indexer shim). The current
`vllm-node` image has absorbed essentially all of it natively:

- `DFlash2DraftModel` and `DFlash2Qwen3ForCausalLM` are registered in
  `vllm/model_executor/models/registry.py`.
- The DFlash2 speculator ships at
  `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` and is routed in
  `v1/worker/gpu/spec_decode/__init__.py`.
- `Glm5NextModel` already inherits `EagleModelMixin`, both wrapper classes
  implement `SupportsEagle3`, and the decoder loop captures aux hidden
  states via `self.aux_hidden_state_layers`.
- `_dflash_layer_causal` already resolves a top-level `is_causal` on the
  drafter config.
- `_get_kv_cache_groups_glm5_next` no longer lives in `kv_cache_utils.py` —
  the GLM-5-Next KV grouping has been refactored into the MLA attention
  backend path (see `v1/attention/backends/mla/indexer.py`).

Only the chat template is still mod responsibility. If a run surfaces a real
correctness issue (not perf), add a focused patch against the current source
here — do not restore the full overlay; it would overwrite native
first-class code with older copies.

## Known perf gaps vs. the sibling repo's bespoke stack

- Prefix-cache repair for the drafter group (sibling repo's
  `patch_prefix_cache_draft_group.py`): the shrink-prevention anchor in
  `kv_cache_coordinator.py:find_longest_cache_hit` still matches upstream
  exactly, but the fix is a repeat-prompt perf optimization, not a
  correctness blocker. Add back as a focused patch here if you observe a
  low `vllm:prefix_cache_hit_rate` on long repeated prompts.
- RoCE all-reduce (+5–18% aggregate at C1–C6): out of scope; future
  `mods/glm-roce-b12x` follow-up.
