# GLM-5.3-Flash DFlash2 overlay

This runtime mod ports **DFlash2 speculative decoding** (upstream vLLM PR #52816)
onto a GLM-5.3-Flash target model and teaches the hybrid KV cache layout to
accommodate the drafter's sliding-window attention group. On 2x DGX Spark
(GB10 / SM121) this takes GLM-5.3-Flash NVFP4 from ~21.8 tok/s (MTP-4) to
~46.9 tok/s decode — a ~2.15x speedup — without changing capacity.

Pair the mod with `recipes/glm-5.3-flash-dflash2.yaml`.

## What it does

At container launch, `run.sh` applies four anchored patches and installs three
new modules into the running vLLM:

- `patch_registry_and_select.py` — registers `DFlash2DraftModel` and routes its
  config through the DFlash2 speculator.
- `patch_glm_aux_capture.py` — adds `EagleModelMixin` + `SupportsEagle3` to the
  `Glm5Next*` target model so the drafter can tap aux hidden states at layers
  5/14/24/33/42 (with mHC contraction, matching the deepseek_v4 convention).
- `patch_glm5_drafter_group.py` — teaches `_get_kv_cache_groups_glm5_next` to
  keep the fast-path KV layout when the drafter adds 5 `SlidingWindowSpec`
  layers. Without this, the model falls back to the generic uniform-page path
  and warmup dies on `tail_kv_cache.shape[2] == pool_size`. **This is the
  critical patch**; the drafter's per-block byte cost ends up bit-identical to
  the base model.
- `patch_prefix_cache_draft_group.py` — repairs prefix-cache coordination for
  the drafter group so repeated prompts hit the cache (13K-token repeat TTFT
  21.1 s → 6.3 s; measured hit rate 0.986 at 262K context).
- New modules: `qwen3_dflash2.py` (DFlash2 draft model class) and
  `v1/worker/gpu/spec_decode/dflash2/{__init__,speculator}.py` (DFlash2
  speculator with local convolution + candidate selector).
- SM121-safe `sparse_attn_indexer_kpool.py` (replaces `torch.empty` with
  `torch.full(-1)` so uninitialized int32 pool ids cannot produce bogus token
  indices; required for stable decode past ~24K context on GB10).
- Stages `chat_template_mm.jinja` into the launch cwd so recipes can reference
  it as `--chat-template chat_template_mm.jinja` (same pattern
  `mods/diffusiongemma` uses).

## Requirements

- A `vllm-node` image whose vLLM already supports `Glm5NextForConditionalGeneration`
  (serves `nvidia/GLM-5.3-Flash-NVFP4`). The baseline recipe
  `recipes/glm-5.3-flash.yaml` targets this family on `vllm-node-b12x` — the
  mod does not depend on the b12x build, but it does require the glm5next
  model source to be present and `KpoolTailSpec` / `SlidingWindowSpec` to live
  at `vllm.v1.kv_cache_interface`.
- 2x DGX Spark cluster (GB10, SM121). Topology, NCCL/RoCE env, and swappiness
  come from `setup-cluster.sh` / `launch-cluster.sh`.
- Drafter checkpoint on each node (recipe default: `incoai/GLM-5.3-Flash-DFlash2`,
  ~2.2 GB).

## Idempotency

Every patch script is marker-based and self-skips when its anchor is already
present. Running the mod twice is safe and prints only `[skip]` lines on the
second invocation. The postcondition grep (`"DFlash2DraftModel"` in
`registry.py`) ensures the mod never claims success without the drafter
actually registered.

## Known constraints

- Hardcoded assumption that `vllm/v1/core/kv_cache_coordinator.py`,
  `vllm/v1/core/kv_cache_utils.py`, and the glm5next target model are present.
  The mod discovers paths via `importlib.util.find_spec` + `find` rather than
  hardcoding `/usr/local/lib/python3.12/dist-packages`, so unusual image
  layouts still work.
- Patch anchors match specific upstream vLLM source strings (around commit
  `g487ecf187`, which registers `Glm5NextForConditionalGeneration`). If the
  `vllm-node` image advances past these anchors they will no longer match and
  the patch scripts will fail loudly (ANCHOR NOT FOUND) rather than silently
  patching nothing.
- The `block_size: 2304` default in the paired recipe is non-obvious: it is
  co-designed with `patch_glm5_drafter_group.py`'s exact-fit gate (divisible by
  64 to cover FlashInfer's registered SWA kernel block sizes, and a multiple
  of the MLA page size so slot-sharing stays sound under any kernel split).
  Changing it in a recipe override will likely cause the drafter group to fall
  back to the standalone path or fail allocation. Do not override.
- The optional RoCE all-reduce (+5–18% aggregate at C1–C6) is out of scope
  here; it is a future `mods/glm-roce-b12x` follow-up.

## References

- Overlay source of record (verbatim files): the sibling repo
  `GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/overlay-dflash2/`.
- Upstream vLLM PR: #52816 (DFlash2 speculator, merged 2026-08-21).
- SM121 top-k fix forensics: the sibling repo's
  `docs/SM121-CRASH-FORENSICS-2026-08-27.md`.
