# ModelExpress P2P — Load Time Comparisons

**Model**: Kimi K2.5 (DeepSeek V3, NVFP4, 256 experts, 84.5 GB/rank)
**Cluster**: GCP GB200 NVL36, ARM64
**Image**: `kavink:dynamo-trtllm-mx-v1.8.0`

---

## Aggregated TP=8 (2 nodes × 4 GPUs)

### Source (PVC → GPU)

| Phase | Time |
|-------|------|
| MPI session + model init | ~25s |
| Weight loading from PVC (118 safetensors, 8 ranks) | ~17 min |
| `post_load_weights()` (NVFP4 padding, FP8 resmooth, indexer fusion) | ~2 min |
| ModelExpress publish (NIXL registration + gRPC metadata) | <1s |
| NCCL init + CUDA graphs | ~2 min |
| **Total (Model init total)** | **~1453s (~24 min)** |

### Target (P2P → GPU)

| Phase | Time |
|-------|------|
| MPI session + model meta-init | ~25s |
| MX query + NIXL RDMA transfer (90.75 GB/rank, 1815 params) | **1.6s** |
| `post_load_weights()` (same transforms as source) | ~2 min |
| NCCL init + CUDA graphs | ~2 min |
| **Total** | **~5 min** |

### Comparison

| Metric | Source (PVC) | Target (P2P) | Speedup |
|--------|-------------|-------------|---------|
| Weight loading | ~17 min | **1.6s** | **~640x** |
| Total startup | ~24 min | ~5 min | **~5x** |

Weight loading dominates the source's startup time. P2P eliminates it entirely.
The remaining ~5 min on the target is `post_load_weights` + NCCL init + CUDA graphs,
which are independent of how weights are loaded.

---

## P2P Transfer Details

| Metric | Value |
|--------|-------|
| Transfer speed | 363–479 Gbps (RoCE) |
| Data per rank | 90.75 GB |
| Parameters per rank | 1815 |
| Parameters matched | 1815/1815 (100%) |
| Size mismatches | 0 |
| Dtype casts needed | 0 |
| Transport | RoCE RC (`UCX_TLS=self,sm,rc,cuda_copy,gdr_copy,tcp`) |

---

## Disaggregated TP=8 (PVC baseline)

Both prefill and decode load from PVC independently.

### Prefill (TP=8, 2 nodes)

| Phase | Time |
|-------|------|
| Model init total | ~1453s (~24 min) |
| Weight loading from PVC | ~17 min |
| `post_load_weights` | ~2 min |
| NCCL init | ~2 min |

### Decode (TP=8, 2 nodes)

| Phase | Time |
|-------|------|
| Model init total | ~821–1019s (~14–17 min) |
| Weight loading from PVC | ~10 min |
| `post_load_weights` | ~2 min |
| NCCL init + CUDA graphs | ~2 min |

### Disagg Inference Latency

| Metric | Value |
|--------|-------|
| TTFT (prefill time) | 480–640ms |
| Total (50 tokens) | 2.3s |
| Total (100 tokens) | 2.9s |
| KV cache backend | DEFAULT (NIXL) |
| KV hit rate | 0.0 (first request) |

---

## Disaggregated TP=8 (P2P targets)

*Pending deployment — compute domain scheduling in progress.*

Expected improvements:
- Prefill weight load: ~17 min → **~2s** (P2P)
- Decode weight load: ~10 min → **~2s** (P2P)
- Total disagg startup: ~24 min → **~5 min** (bottleneck: `post_load_weights` + NCCL)

---

## Historical: Aggregated TP=4 (single node)

| Metric | Source (PVC) | Target (P2P) |
|--------|-------------|-------------|
| Weight load time | ~68 min | **3.4s** |
| Transfer speed | — | 369 Gbps |
| TTFT | — | 3.4s |
| Data transferred | — | 648 GB (4 ranks × 162 GB) |

---

## Key Observations

1. **P2P transfer is not the bottleneck** — at 458 Gbps, 90.75 GB transfers in 1.6s.
   The remaining startup time is dominated by `post_load_weights()` (~2 min),
   NCCL initialization (~2 min), and CUDA graph capture.

2. **`post_load_weights()` must run on targets** — this was the root cause of
   garbage inference. Source publishes weights BEFORE `post_load_weights()`,
   targets run the same transforms on received data. See `TRTLLM_MULTINODE.md`.

3. **Disagg adds minimal overhead** — the KV cache transfer between prefill
   and decode takes <1ms for small prompts. The `DEFAULT` transceiver backend
   avoids the UCX UD timeout that blocked `UCX` backend.

4. **Scaling benefit** — once a source is loaded, each new target replica starts
   in ~5 min instead of ~24 min. With N targets, total GPU-hours saved per scale
   event is `N × 19 min`.
