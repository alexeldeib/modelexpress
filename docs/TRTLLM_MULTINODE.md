# ModelExpress P2P Weight Transfer for TRT-LLM Multinode Inference

**Status**: Working (validated March 16 2026)
**Model**: Kimi K2.5 (DeepSeek V3 architecture, 256 experts, NVFP4 quantization)
**Config**: TP=8 across 2 nodes (4 GPUs per node), GCP GB200 NVL36
**Speed**: 458 Gbps RoCE P2P, 1815 parameters, 90.75 GB per rank

---

## Overview

ModelExpress enables GPU-to-GPU (P2P) weight transfer for TRT-LLM models running
under the Dynamo orchestrator on Kubernetes. A **source** instance loads model weights
from disk (PVC) and publishes them via NIXL RDMA. **Target** instances receive weights
directly into GPU memory at RoCE line rate, skipping disk I/O entirely.

This enables fast worker scaling — new inference replicas can be ready in seconds
rather than the 15–20 minutes required for PVC loading.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Kubernetes (GCP GB200)                │
│                                                          │
│  ┌─────────────────────┐     ┌─────────────────────┐     │
│  │   Source DGD (TP=8)  │     │   Target DGD (TP=8)  │    │
│  │                     │     │                      │     │
│  │  Node A (4 GPUs)    │     │  Node C (4 GPUs)     │     │
│  │  ┌──┐┌──┐┌──┐┌──┐  │     │  ┌──┐┌──┐┌──┐┌──┐   │     │
│  │  │R0││R1││R2││R3│  │     │  │R0││R1││R2││R3│   │     │
│  │  └──┘└──┘└──┘└──┘  │     │  └──┘└──┘└──┘└──┘   │     │
│  │  Node B (4 GPUs)    │     │  Node D (4 GPUs)     │     │
│  │  ┌──┐┌──┐┌──┐┌──┐  │     │  ┌──┐┌──┐┌──┐┌──┐   │     │
│  │  │R4││R5││R6││R7│  │     │  │R4││R5││R6││R7│   │     │
│  │  └──┘└──┘└──┘└──┘  │     │  └──┘└──┘└──┘└──┘   │     │
│  └────────┬────────────┘     └────────┬─────────────┘     │
│           │                           │                   │
│           │  1. Publish metadata      │  3. RDMA pull     │
│           ▼                           ▼                   │
│  ┌─────────────────────────────────────────┐              │
│  │      ModelExpress Server + Redis        │              │
│  │  (coordination, metadata, NIXL descs)   │              │
│  └─────────────────────────────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### Weight Loading Pipeline

The key insight is **when** the source publishes its weights relative to TRT-LLM's
internal weight processing pipeline:

```
Source (LoadFormat.AUTO):
  1. checkpoint_loader.load_weights()     ← loads from PVC, TP shards
  2. model.load_weights(weights)          ← name mapping, fusing (QKV, gate_up)
  ──── PUBLISH POINT ────────────────────── ← ModelExpress publishes HERE
  3. post_load_weights()                  ← NVFP4 padding, FP8 resmooth, indexer fusion

Target (LoadFormat.PRESHARDED):
  1. MxLiveCheckpointLoader.load_weights() ← NIXL RDMA into param buffers
  2. (skip model.load_weights)             ← data already in GPU memory
  3. post_load_weights()                   ← same transforms as source
```

The source publishes weights AFTER `load_weights()` (step 2) but BEFORE
`post_load_weights()` (step 3). This means the target receives pre-processed
weights in the same format the source had before kernel-ready transforms,
and both sides run identical `post_load_weights()` pipelines.

This design was critical for correctness. Publishing AFTER `post_load_weights()`
caused garbage inference because several modules (NVFP4 Linear, FP8 MoE, DeepSeek
Attention) allocate new `nn.Parameter` tensors with different sizes during
`post_load_weights()`, creating a format mismatch between source and target.

## Workflow

### Prerequisites

1. **Dynamo platform** deployed (etcd + NATS + operator)
2. **ModelExpress server** + Redis running (`mx-infra-decode.yaml`)
3. **Shared PVC** (`shared-model-cache`) with model files
4. **ComputeDomain** with enough GPU nodes

### Deployment Steps

1. Deploy source DGD:
   ```bash
   kubectl -n <ns> apply -f kimi-source-decode-dgd.yaml
   ```
   Source loads weights from PVC (~15 min for Kimi K2.5), publishes to MX server.

2. Deploy target DGD:
   ```bash
   kubectl -n <ns> apply -f kimi-target-agg-tp8-dgd.yaml
   ```
   Target pulls weights via RDMA (~2 seconds at 458 Gbps), runs post_load_weights,
   starts serving.

3. Test inference:
   ```bash
   kubectl -n <ns> exec <frontend-pod> -- curl -s http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"baseten-admin/Kimi-2.5-text-nvfp4-v3",
          "messages":[{"role":"user","content":"What is the capital of France?"}],
          "max_tokens":50}'
   ```

### DGD Configuration

**Source** (`model_express_role: source`):
- `model_express_url`: gRPC address of MX server
- `model_express_role: source` — loads from PVC, publishes via NIXL
- Uses `LoadFormat.AUTO` internally (normal PVC loading)

**Target** (`model_express_role: target`):
- `model_express_url`: same MX server
- `model_express_role: target` — receives weights via RDMA
- Uses `LoadFormat.PRESHARDED` internally (P2P into param buffers)
- Does NOT need PVC access for model weights (only for config/tokenizer)

### Network Configuration (GCP GB200)

RoCE requires specific UCX/MPI settings:

```yaml
env:
  UCX_TLS: "self,sm,rc,cuda_copy,gdr_copy,tcp"
  UCX_IB_GID_INDEX: "3"               # Critical for GCP RoCEv2
  TRTLLM_UCX_INTERFACE: eth0           # Prevents 169.254.x.x binding
  OMPI_MCA_pml: ob1                    # Avoids UCX UD timeout
  OMPI_MCA_btl: "tcp,self,vader"
  OMPI_MCA_btl_tcp_if_include: eth0
  OMPI_MCA_oob_tcp_if_include: eth0
securityContext:
  privileged: true                     # Required for RDMA mem registration
```

---

## Code Changes

Three repositories are modified. All changes are on feature branches.

### 1. ModelExpress (`kavink/trtllm` branch)

**`modelexpress_client/python/modelexpress/trtllm_live_transfer.py`**

Core P2P logic for TRT-LLM integration:

- `publish_model_params(model)` — called from `ModelLoader.load()` BEFORE
  `post_load_weights()`. Iterates `model.named_parameters()`, registers with NIXL,
  publishes tensor metadata (name, address, size, dtype, device) to MX server via
  gRPC. Uses MPI gather to collect all ranks' metadata on rank 0 before publishing.

- `publish_from_worker(worker)` — legacy hook called from `worker_main()` AFTER
  `post_load_weights()`. Now only runs if `publish_model_params` didn't already
  publish (checked via `model._mx_source_published` flag).

- `MxLiveCheckpointLoader` — implements TRT-LLM's `BaseCheckpointLoader` interface:
  - `load_config()` — loads HF config from PVC (tokenizer, model config)
  - `load_weights(model=model)` — queries MX server for source metadata, matches
    source/target params by name, initializes NIXL agent, performs RDMA transfer.
    Returns empty dict when P2P succeeds (signals model_loader to skip `load_weights()`).

- `NixlTransferManager` (`nixl_transfer.py`) — manages NIXL agent lifecycle:
  register tensors, load remote metadata, execute transfers.

**`trtllm_patches/v1.3.0rc5/apply_patches.py`**

Patches TRT-LLM 1.3.0rc5 inside the container:
- Adds `LoadFormat.PRESHARDED = 3` to `llm_args.py`
- Adds PRESHARDED branch to `model_loader.py` (TP skip via `_weights_presharded`)
- Patches `linear.py` helper functions to skip TP slicing when `_weights_presharded`
- Adds `publish_from_worker()` hook to `worker.py`

**`trtllm_patches/v1.3.0rc5/patch_model_loader.py`**

Applies on top of `apply_patches.py`:
- Inserts source publish hook (`publish_model_params`) before `post_load_weights` loop
- Updates worker.py to skip duplicate publish if already published from model_loader

**`trtllm_patches/v1.3.0rc5/patch_tp_allgather.py`**

Fixes MPI truncation with ob1 TCP BTL:
- Replaces `comm.allgather()` with chunked version (64KB chunks) in `dist.py`

**`examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200`**

ARM64 Docker image layered on `karenc:dynamo-trtllm-v1.0.0`:
- Installs ModelExpress client (gRPC + NIXL)
- Copies Dynamo P2P hooks from `dynamo` repo
- Applies all TRT-LLM patches
- Verifies installation

### 2. Dynamo (`kavink/trtllm-p2p` branch)

**`components/src/dynamo/trtllm/backend_args.py`**

Adds CLI arguments and config fields:
- `--model-express-url` — MX server gRPC address
- `--model-express-role` — `source` or `target`

**`components/src/dynamo/trtllm/engine.py`**

Engine-level P2P integration:
- `_setup_modelexpress_source()` — sets `MODEL_EXPRESS_SOURCE=1` and
  `MODEL_EXPRESS_URL` env vars before LLM init, so the model_loader publish
  hook fires during weight loading
- `_verify_workers_published()` — polls MX server gRPC to confirm all workers
  published metadata (up to 300s timeout)
- `_setup_modelexpress_loader()` — for target role, creates `MxLiveCheckpointLoader`,
  sets `load_format=LoadFormat.PRESHARDED` and `checkpoint_loader` on LLM args

**`components/src/dynamo/trtllm/workers/llm_worker.py`**

Passes `model_express_url` and `model_express_role` from config into the engine.

### 3. TensorRT-LLM (`kavink/presharded-weight-loading` branch)

**`tensorrt_llm/_torch/pyexecutor/model_loader.py`**

Core changes to `ModelLoader.load()`:

- **PRESHARDED LoadFormat**: new elif branch that sets `_weights_presharded=True`
  on all Linear modules, calls `checkpoint_loader.load_weights(model=model)`,
  and skips `model.load_weights()` when the loader returns empty dict (P2P target).

- **Source publish hook**: after all load format branches but BEFORE
  `post_load_weights()`, checks `MODEL_EXPRESS_SOURCE` env var and calls
  `publish_model_params(model)` to publish pre-processed weights.

- **Full post_load_weights**: both source and target run the standard
  `post_load_weights()` loop — no conditional skip.

**`tensorrt_llm/executor/worker.py`**

- Adds `publish_from_worker()` call in `worker_main()` as a fallback
- Checks `model._mx_source_published` flag to skip if already published
  from model_loader (prevents double-publish)

**`tensorrt_llm/llmapi/llm_args.py`** (via patch)

- Adds `PRESHARDED = 3` to `LoadFormat` enum

**`tensorrt_llm/_torch/modules/linear.py`** (via patch)

- In `load_weights_vanilla_helper`, `load_weights_fused_qkv_helper`,
  `load_weights_fused_gate_up_helper`: when `_weights_presharded=True`,
  forces `tp_size=1` so `load_weight_shard()` skips TP slicing

### Build Process

All three repos are combined into a single Docker image:

```bash
cd modelexpress
docker buildx build --platform linux/arm64 --no-cache \
    -f examples/p2p_transfer_trtllm/Dockerfile.ph3-gcp-gb200 \
    --build-context dynamo=../dynamo \
    -t nvcr.io/nvidian/dynamo-dev/kavink:dynamo-trtllm-mx-v1.8.0 \
    --push .
```

The Dockerfile:
1. Starts from the Dynamo TRT-LLM base image (TRT-LLM 1.3.0rc5 + NIXL)
2. Installs ModelExpress Python client
3. Copies Dynamo engine/worker files from the `dynamo` repo (via `--build-context`)
4. Applies TRT-LLM patches (PRESHARDED, publish hook, allgather fix)

---

## Key Debugging Findings

### RoCE Configuration (GCP GB200)

- `UCX_IB_GID_INDEX=3` is required for RoCEv2 on GCP — without it, UCX picks
  the wrong GID and connections fail silently
- `TRTLLM_UCX_INTERFACE=eth0` prevents TRT-LLM's internal NIXL agent from binding
  to `169.254.x.x` link-local addresses (the C++ `getAvailableIP()` function
  iterates interfaces and picks the first IPv4 without skipping link-local)
- `privileged: true` is required for RDMA memory registration (`ibv_reg_mr`)
- `OMPI_MCA_pml=ob1` avoids UCX UD transport hangs during MPI bootstrap

### Weight Format Mismatch (Root Cause of Garbage Output)

TRT-LLM's `post_load_weights()` performs critical transforms that create
NEW parameter tensors:

| Module | Transform | Effect |
|--------|-----------|--------|
| NVFP4 Linear | Pad weight/scale for GEMM alignment (32×16) | New `nn.Parameter` with different shape |
| FP8 Linear | Resmooth to E8M0 + scale layout transform | New scale `nn.Parameter` |
| DeepSeek Attention | Fuse `indexer.wk` into `kv_a_proj_with_mqa` | In-place copy + set indexer.wk = None |
| FP8 MoE | Resmooth all expert weights/scales | New weight/scale `nn.Parameter`s |

Publishing AFTER these transforms meant source parameters had different
sizes/layouts than the target's meta-init buffers — causing silent size
mismatches and garbage output. Moving the publish point to BEFORE
`post_load_weights()` resolved this completely.

### MPI Allgather Truncation

The ob1 PML's TCP BTL has a ~64KB limit on allgather messages. TRT-LLM's
`tp_gather` sends large tensors (>64KB) via `comm.allgather()`, which
fails with `MPI_ERR_TRUNCATE`. Fixed by chunking allgather into 64KB
segments in `tensorrt_llm/distributed/dist.py`.

---

## Performance

| Metric | Value |
|--------|-------|
| P2P transfer speed | 458 Gbps (RoCE) |
| Data per rank | 90.75 GB |
| Transfer time | 1.6 seconds |
| Parameters matched | 1815/1815 (100%) |
| Source load time | ~15 minutes (PVC) |
| Target load time | ~45 seconds (P2P + post_load_weights + warmup) |
| TTFT (first request) | 2.7 seconds |
| TTFT (subsequent) | 1.7 seconds |

## Repository Branches

| Repo | Branch | Base |
|------|--------|------|
| `modelexpress` | `kavink/trtllm` | main |
| `dynamo` | `kavink/trtllm-p2p` | v0.9.0 |
| `TensorRT-LLM` | `kavink/presharded-weight-loading` | main (93b0dc7af) |
