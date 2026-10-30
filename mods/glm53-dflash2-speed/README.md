# glm53-dflash2-speed

Runtime overlays for `recipes/glm-5.3-flash-dflash2-speed.yaml` on the
`glm53-roce:v11-b58f34ea` image.

Vendored verbatim from [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
at commit `770d115` (MIT; vLLM-derived files Apache-2.0): the whole `overlay/`
tree plus `profiles/levers_policy.json` (the `SPEC_PROBE_CONTROL` policy). The
TP2 port follows
[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark `runs/2026-09-29-knapcio-tp2/`](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/main/runs/2026-09-29-knapcio-tp2)
(TP4-only pieces off: `GLM_PREFILL_SHARD=0`, `PF3_ARM=off`, `GATHER_ROUTE=0`).

`run.sh` reproduces the bind-mounts of knapcio's `start.sh @770d115`:

- 13 vLLM files replaced from `overlay/` — always-on set (`sparse_attn_indexer_kpool.py`,
  `glm47_moe.py`, `abstract_parser.py`, `kv_cache_coordinator.py`, the GB10 Marlin
  tune JSON) plus the `GLM5NEXT_PATCH`, `GLM_KPOOL_FIX` and `GLM_ARGMAX_CLAMP`
  sets that `profiles/current.env` arms. `kv_cache_coordinator.py` already
  carries the #18 prefix-cache repair.
- `/overlay/overlay` — the PYTHONPATH tree (levers scheduler, hooks,
  sitecustomize) that the recipe's `PYTHONPATH` and `GLM_*` env consume.
- `/overlay/profiles/levers_policy.json` — `SPEC_PROBE_CONTROL` target.

The mod refuses to run unless the tree looks like the v11-dflash2 base
(`model_executor/models/qwen3_dflash2.py` present, dist-packages layout).

Unlike the sibling `mods/glm53-dflash2`, no per-node patch/bundle building is
done here: the RoCE runtime (`b12x` at `b58f34ea` + the `glm_roce` import-time
shim + the compiled RDMA proxy) is baked into the image by knapcio's
`Dockerfile.roce`; it activates only with `GLM_ROCE_ALLREDUCE=1` (set in the
recipe env). A base image without the shim boots on stock NCCL with a warning.
