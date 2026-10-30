# glm53-dflash2

Runtime overlays for `recipes/glm-5.3-flash-dflash2.yaml` on the
`ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` image. The image bakes
the DFlash2 drafter support (upstream vLLM PR #52816 port), but the recipe's
source launcher (`launch-glm53-vllm-tp2-dflash2.sh` in
[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark))
bind-mounts three runtime files the published tag lacks; this mod installs
them into the running container instead:

| file | target in container | why |
|---|---|---|
| `sparse_attn_indexer_kpool_sm121.py` | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` | SM121 top-k fix; without it the engine dies on any decode past ~24K context (repo `docs/SM121-CRASH-FORENSICS-2026-08-27.md`) |
| `patch_prefix_cache_draft_group.py` | edits `vllm/v1/core/kv_cache_coordinator.py` | #18 prefix-cache repair for the drafter group; without it prefix-cache hits stay 0 (repo `docs/PREFIX-CACHE-DFLASH2-SM121.md`) |
| `chat_template_mm.jinja` | `/opt/glm53-dflash2/` | vision chat template; image requests 500 without it |
| `roce/*.py` | five vLLM files | optional b12x RoCE one-shot all-reduce; applied only when the b12x runtime (`b12x` package + `/opt/b12x-roce`) is present in the image, else skipped and NCCL stays |

Provenance: all bundled files are copied verbatim from the GLM repo above
(`docker/`, `docker/dflash2-overlay/`, `speed-night-2026-09-18/roce/`, repo
root). The RoCE adapters are the b12x RoCEnante port from
local-inference-lab/vllm#597 (runtime: Luke Alonso, Apache-2.0).

The mod refuses to run unless the tree looks like `sm121-v11-dflash2`
(presence of `model_executor/models/qwen3_dflash2.py` and the dist-packages
layout), so an image mismatch fails at launch instead of booting broken.

The b12x RoCE runtime is not distributable from source repos: the source
launcher extracts it from a locally built DS4 image
(`vllm-dsv41:exl3b-roce`, `speed-night-2026-09-18/roce/build-roce-bundle.sh`).
To get RoCE here, bake `b12x`, `b12x-1.3.0.dist-info` and `/opt/b12x-roce`
into a derivative of the v11-dflash2 image; this mod then applies the adapter
automatically.
