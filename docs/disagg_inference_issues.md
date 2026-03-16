# Disagg Multinode Inference Issues — GB200

**Date**: March 11, 2026 (updated)
**Cluster**: `dynamo-gcp-dev-01` (GCP GB200 NVL36, ARM64)

---

## Overview: What We've Achieved and What's Left

### Fully validated (ModelExpress P2P)
- **Aggregated TP=4**: P2P at 369 Gbps, end-to-end inference with coherent output (TTFT 3.4s)
- **DGDSA scaling**: second worker loads via P2P at 371-390 Gbps, ready in ~7.5 min
- **Disagg same-TP (TP=4+TP=4)**: both prefill and decode load concurrently at 234-538 Gbps
- **Mixed TP P2P (TP=4+TP=8)**: prefill 360-538 Gbps, decode 345-610 Gbps (8 ranks, 2 nodes)
- **Multinode source publish**: TP=8 source publishes all 8 workers correctly across 2 nodes
- **Cross-clique RoCE**: works at 345-379 Gbps (o7v→w0e)

### Infrastructure issues resolved
- **`MPI_ERR_TRUNCATE`**: Fixed with `safe_allgather` patch (chunks large MPI messages to 64KB)
- **UCX UD timeout**: Worked around with `ob1` PML
- **Multinode SSH**: Fixed with `HOME=/root`, `/run/sshd`, `HF_MODULES_CACHE`
- **Multinode rank mapping**: Fixed MPI rank vs `torch.cuda.current_device()` for source + target
- **NFS file lock (ESTALE)**: Fixed with `HF_MODULES_CACHE=/tmp/hf_modules`
- **KV cache transceiver backend**: Changed from `UCX` to `DEFAULT` (NIXL)

### Current issue: Garbage output from mixed TP disagg inference

Disagg inference pipeline runs end-to-end (prefill→KV transfer→decode→tokens), but
output is nonsensical (e.g., `"0000000000 ".format`). Multiple requests succeed
without crashing (no MPI errors, no hangs). Decode generates tokens at full speed
(~16ms/step). The issue is purely **output quality**.

**What works**: TP=4 aggregated inference produces coherent output from the same model
and same source weights. This confirms the weights are correct for TP=4.

**What's new**: This is our first time running TP=8 decode across 2 nodes. The TP=8
decode source publishes 8 workers × 90.75 GB each. The TP=8 decode target receives
all 8 shards correctly via NIXL RDMA at 365-376 Gbps.

### Root Cause Analysis (updated 2026-03-16)

**Status**: RoCE P2P validated at 363-479 Gbps. Checksums match between source and
target. Aggregated TP=8 target inference still produces garbage output ("0000 a00...").

**Disproven hypotheses**:
- ~~Weight corruption during RDMA~~: Checksums are byte-identical (verified via sum + nonzero)
- ~~MoE backend mismatch~~: Both source and target use `TRTLLM` backend
- ~~`model.load_weights({})` overwriting RDMA data~~: Patched to skip when empty dict
- ~~`post_load_weights()` recomputing quant scales~~: Patched to skip entirely for P2P

**True root cause: Skipping module-level `post_load_weights()` drops critical transforms**

Deep analysis of TRT-LLM codebase (`tensorrt_llm/_torch/`) reveals that the current
approach — transferring the source's FINAL parameter state and skipping all module
`post_load_weights()` — is **fundamentally flawed**. The problem is NOT that
`post_load_weights` corrupts data; it's that certain modules create NEW parameter
tensors (allocating fresh `nn.Parameter` objects), which means the P2P-written data
is in the OLD tensor that gets replaced.

#### Key `post_load_weights()` transforms we're skipping:

| Module | File | What it does | Impact of skipping |
|--------|------|-------------|-------------------|
| **NVFP4 Linear** | `linear.py:1545` | Pads weight/scale for GEMM alignment (32×16), allocates NEW Parameter | Target uses unpadded buffers with P2P data, but shapes may be misaligned for GEMM kernels |
| **FP8 Linear** | `linear.py:1184` | `resmooth_to_fp8_e8m0()` + `transform_sf_into_required_layout()`, allocates NEW scale Parameter | Scale layout wrong for TMA on SM100/SM120 |
| **DeepSeekV3 Attention** | `modeling_deepseekv3.py:812` | Copies `indexer.wk` into fused `kv_a_proj_with_mqa`, sets `indexer.wk = None` | P2P transfers fused weight correctly, but `indexer.wk` remains non-None on target |
| **FP8 MoE** | `quantization.py:1112` | Resmooths FP8 to E8M0 for all experts, allocates NEW weight/scale Parameters | Expert weights in wrong FP8 variant |
| **FP8 MoE** | `quantization.py:873` | Pads w3_w1/w2 for CUTLASS alignment (32×16) | Misaligned for CUTLASS kernels |
| **MoE shared experts** | `quantization.py:497` | `setup_quant_scales()`, finalize shared expert weight distribution | Quant scales never set up |

#### The Parameter replacement problem:

Several `post_load_weights()` implementations do this pattern:
```python
module.weight = Parameter(F.pad(module.weight, ...), requires_grad=False)
module.weight_scale = Parameter(transform(module.weight_scale), ...)
```

This creates a **new** `nn.Parameter` tensor and assigns it to the module. The old
tensor (which P2P wrote data into) is orphaned. Even if we ran `post_load_weights()`,
it would process the P2P-transferred data — but the key issue is that these
transforms MUST run because the GEMM kernels expect the transformed layout.

#### Why skipping doesn't work:

Our current approach: "source runs all transforms → P2P final state → target skips transforms"

This fails because:
1. Source's `post_load_weights` allocates NEW padded parameters (e.g., 4128×3584 instead of 4096×3584)
2. Target's meta init creates UNPADDED parameters (4096×3584)
3. P2P matches by name — if source tensor is LARGER than target tensor, they go into
   "size mismatch" and are skipped (logged as warning, but we missed it in the noise)
4. Even for tensors that match size, the scale layout transforms (interleave, resmooth)
   mean the source's scale data is in the FINAL kernel-expected format, but the target
   buffers were never allocated with the right shape

### Fix Strategy

**Option 1: RUN `post_load_weights()` on P2P target (recommended)**

Instead of skipping module post_load_weights, we should:
1. Transfer weights from source BEFORE post_load_weights runs (i.e., transfer the
   raw checkpoint-loaded state, not the final kernel-ready state)
2. OR: Transfer the final state AND also run post_load_weights on target

Option 1a — "Transfer pre-processed weights":
- Source: after `model.load_weights()` but BEFORE `post_load_weights()`, publish params
- Target: receive via P2P into unpadded buffers, then run full `post_load_weights()`
- Pro: Clean separation, no format mismatch
- Con: Requires source to publish at a different point in the loading pipeline

Option 1b — "Let post_load_weights run on P2P data":
- Remove the `_mx_p2p_weights_loaded` check so module post_load_weights DOES run
- This means P2P transfers raw (pre-transform) data, and target processes it normally
- Pro: Simplest change — just remove our patches
- Con: Requires understanding which transforms are idempotent vs. destructive

**Option 2: Match source parameter shapes exactly**

- After model init on target but BEFORE P2P, run a "dry run" of post_load_weights
  to get the final parameter shapes (with padding)
- Reallocate target parameters to match source shapes
- Transfer via P2P
- Skip post_load_weights since data is already in final form
- Pro: True zero-transform P2P
- Con: Complex, fragile, tightly coupled to TRT-LLM internals

**Option 3 (simplest): Use source's `load_weights()` format**

- Source publishes weights from AFTER `load_weights()` but BEFORE `post_load_weights()`
- On source side, hook into the model loader between lines 394 and 471 of model_loader.py
- Target receives pre-transform weights, then runs normal `post_load_weights()`
- This is the cleanest approach and avoids ALL format mismatch issues

### Immediate next step

Test **Option 1b** first (remove the skip): In `model_loader.py`, remove the
`_mx_p2p_weights_loaded` conditional so target runs all module `post_load_weights()`.
The P2P data will be processed by the same transforms that would run on PVC-loaded data.
If NVFP4 padding produces the wrong result because P2P data is already padded, we'll
see it. If the dimensions happen to match (Kimi K2.5 may have aligned dimensions),
this could "just work".

If Option 1b fails, implement **Option 3**: change publish point to AFTER load_weights
but BEFORE post_load_weights.

---

## Resolved Issues

### Issue 1 (RESOLVED): MPI_ERR_TRUNCATE with ob1 PML

**Symptom**: During inference, `tp_gather` → `comm.allgather` fails with `MPI_ERR_TRUNCATE: message truncated` on the decode TP=8 multinode worker.

**Call stack**:
```
py_executor._executor_loop
  → _handle_first_token_response
    → _enqueue_responses
      → dist.tp_gather
        → safe_gather → comm.allgather
          → MPI_ERR_TRUNCATE
```

**Root cause**: `OMPI_MCA_pml=ob1` + `OMPI_MCA_btl=tcp,self,vader` forces MPI to use TCP BTL for cross-node communication. TCP BTL has a fixed max message size (~128KB default). TRT-LLM's `allgather` sends pickled Python response objects that exceed this limit.

**Why we need ob1**: Without it, UCX PML discovers IB/UD devices via `privileged: true` and hits a **UCX UD endpoint timeout** during MPI bootstrap on GB200 — a known hardware/driver issue.

**What we tried**:

| Config | Result |
|--------|--------|
| `ob1` + `tcp,self,vader` | P2P works (370+ Gbps), inference truncates |
| `ob1` + `tcp,self,vader` + `btl_tcp_max_send_size=1MB` | MPI session hung |
| No OMPI overrides + `privileged: true` | MPI init hangs (UCX UD timeout) |
| No OMPI overrides + `runAsUser: 0` | MPI works, NIXL TCP only (~3 Gbps) |
| `UCX_TLS=tcp,self,sm` + `privileged: true` | MpiSession hangs |
| `UCX_TLS=tcp,cuda_ipc,cuda_copy,self,sm` + `privileged: true` | MpiSession hangs |
| No env overrides, let operator handle MPI | `btl_tcp_endpoint` process ID mismatch |

**How Karen avoids this**: Uses `runAsUser: 0` (not privileged) — UCX never sees IB devices, NCCL handles TP collectives via NVIDIA driver, MPI only does small bootstrap messages.

---

### Issue 2 (RESOLVED): UCX_TLS conflict between MPI and NIXL

**Symptom**: Setting `UCX_TLS=tcp` globally (to fix MPI) also restricts NIXL to TCP, dropping P2P from 370 Gbps to ~3 Gbps.

**Root cause**: `UCX_TLS` is a process-wide env var. Both MPI's UCX PML and NIXL's UCX agent read it. We need MPI on TCP but NIXL on RoCE.

**Fix implemented (v1.6.0)**: In `nixl_transfer.py`, NIXL temporarily removes `UCX_TLS=tcp` before creating its agent, then restores it. Supports `NIXL_UCX_TLS` env var for explicit override.

```python
# In NixlTransferManager.initialize():
saved_ucx_tls = os.environ.get("UCX_TLS")
if saved_ucx_tls == "tcp":
    os.environ.pop("UCX_TLS", None)  # let NIXL auto-detect RoCE
try:
    self._agent = NixlAgent(self._agent_name, config)
finally:
    os.environ["UCX_TLS"] = saved_ucx_tls  # restore for MPI
```

**Status**: Built in v1.6.0 image, deployed but pending cluster scheduling. Untested end-to-end.

---

### Issue 3 (RESOLVED): KV Cache Transceiver UCX Connection

**Symptom**: `CacheReceiver` can't connect to prefill's UCX port for KV cache transfer:
```
Error in addConnection(ip) for rank 1 ip: 10.0.15.220 port: 33879
Error in UcxConnection constructor: Request canceled
```

**Root cause**: The `cache_transceiver_config: backend: UCX` creates its own UCX connections between prefill and decode pods for KV cache transfer. With our config, UCX connectivity between the prefill (w0e) and decode (w0e) pods isn't establishing properly.

**Karen's working config**: Both prefill and decode in the same DGD, same clique, `runAsUser: 0`, no UCX_TLS overrides. UCX auto-detects and connects.

**Status**: Not investigated in depth. Secondary to MPI issue.

---

### Issue 4 (RESOLVED): Compute Domain / DRA Scheduling

**Symptom**: Target pods stuck in `Pending` — "cannot allocate all claims" or "node is unschedulable".

**Root cause**: Each node's IMEX channel can only serve one compute domain (`allocationMode: Single`). With Karen's pods, Sara's NCCL test, and our source pods consuming channels, not enough free nodes remain in the clique.

**Workarounds**:
- Move targets to w0e pool (partially worked — 4 free nodes)
- Remove clique affinity between source and target (cross-clique RoCE works at 345-379 Gbps)
- Ask other users to clean up idle compute domains

---

### Issue 5 (RESOLVED): NFS File Lock (ESTALE) on Multinode

**Symptom**: Ranks 4-7 on node B crash with `[Errno 116] Stale file handle` during model config loading.

**Root cause**: TRT-LLM's `config_file_lock()` uses Python `filelock` on the shared NFS PVC. Cross-node NFS locking (NFSv3 NLM) is unreliable.

**Fix**: `HF_MODULES_CACHE=/tmp/hf_modules` moves the lock file to local disk. From coworker's investigation — 100% failure rate with filelock on NFS, 0% on local disk.

**Status**: Fixed.

---

### Issue 6 (RESOLVED): Multinode SSH / sshd

**Symptom**: Worker pod CrashLoopBackOff — `sshd: no hostkeys available` or `Missing privilege separation directory`.

**Root cause**: DGD operator's SSH key gen script uses `~` which resolves to `/home/dynamo` (image USER), but sshd runs as root and looks in `/root/.ssh/`.

**Fixes**:
- `HOME=/root` env var
- `/run/sshd` directory with `chmod 0755` + `chown root:root` in Dockerfile

**Status**: Fixed in v1.6.0 image.

---

## What Works

| Scenario | P2P Speed | Inference | Status |
|----------|-----------|-----------|--------|
| Aggregated TP=4 (single node) | 369 Gbps | Works | **VALIDATED** |
| DGDSA scale 1→2 (single node) | 371-390 Gbps | Works | **VALIDATED** |
| Disagg same-TP TP=4+TP=4 | 234-538 Gbps | KV transceiver issue | P2P validated |
| Mixed TP prefill TP=4 (single node) | 360-393 Gbps | Works (standalone) | **VALIDATED** |
| Mixed TP decode TP=8 (multinode) | 345-479 Gbps | MPI_ERR_TRUNCATE | P2P validated |
| Cross-clique RoCE (o7v→w0e) | 345-379 Gbps | — | **Works** |
| TP=8 source multinode publish | 8 workers published | — | **VALIDATED** |

---

## Approaches to Fix Multinode Inference

### Option A: `runAsUser: 0` for decode (immediate workaround)
- Use Karen's config: `runAsUser: 0`, no OMPI/UCX overrides
- MPI works, NCCL handles TP via NVIDIA driver, KV cache NIXL works
- P2P falls back to TCP (~3 Gbps) — 90.75 GB in ~4 min (still 8x faster than disk)
- **Pros**: Zero code changes, proven by Karen
- **Cons**: No RoCE for decode P2P. Prefill (single-node) still gets RoCE.
- **Status**: Ready to deploy

### Option B: Add `safe_allgather` to TRT-LLM (recommended upstream fix)
- **The actual bug**: `tp_allgather` uses raw `comm.allgather(obj)` without chunking.
  `safe_gather` already exists with `chunk_size=4MB`, but `tp_allgather` bypasses it.
- **File**: `tensorrt_llm/_torch/distributed/communicator.py` line 545
- **Current code**: `return self.tp_comm.allgather(obj)` — no size limit
- **Fix**: Route through chunked allgather (like `safe_gather`), or use
  `torch.distributed.all_gather_object` which handles large messages natively
- **Pros**: Minimal change, fixes ob1 truncation without changing executor
- **Cons**: Requires TRT-LLM upstream PR
- **Status**: Not implemented. Filed as suggestion.

```python
# Suggested fix in MPIDist (communicator.py):
def tp_allgather(self, obj):
    # Use safe chunked allgather instead of raw comm.allgather
    return safe_allgather(self.tp_comm, obj, chunk_size=64 * 1024)
```

### Option C: `TLLM_DISABLE_MPI=1` (needs Ray)
- TRT-LLM has `TLLM_DISABLE_MPI=1` → switches from `MPIDist` to `TorchDist` (NCCL)
- **Problem**: Also switches executor from MPI-based to **Ray-based**
- Requires Ray installed + Ray-based launch (not compatible with operator's `mpirun`)
- **Tested**: Crashes with `ModuleNotFoundError: Cannot import Ray`
- **Pros**: Would completely solve MPI issues
- **Cons**: Different launch paradigm, needs Ray in image + operator support
- **Status**: Tested, not viable with current operator

### Option D: NIXL UCX_TLS override (implemented)
- `UCX_TLS=tcp` globally for MPI
- NIXL temporarily removes it during agent creation → auto-detects RoCE
- `IPC_LOCK` + `SYS_RESOURCE` capabilities (not privileged)
- **Problem**: operator's `mpirun -x UCX_TLS=tcp` re-sets env in worker processes,
  so NIXL override doesn't take effect in multinode workers
- **Pros**: Clean concept, works for single-node
- **Cons**: Doesn't work with `mpirun -x` env propagation
- **Status**: Built in v1.6.0, doesn't work for multinode

### Option E: NIXL per-context UCX config (NIXL team)
- NIXL C++ backend supports `engine_config` with `TLS=rc_v,rc_x,...`
- Python API (`nixl_agent_config`) doesn't expose this yet
- **Fix**: Add `ucx_engine_config` param to Python `nixl_agent_config`:
  ```python
  config = nixl_agent_config(
      backends=["UCX"],
      ucx_engine_config="TLS=rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"
  )
  ```
- This would override UCX_TLS at the NIXL context level, immune to `mpirun -x`
- **Pros**: Proper isolation, works with any MPI config
- **Cons**: Requires NIXL Python API change (C++ already supports it)
- **Status**: Not implemented. File as NIXL feature request.

### Option F: Fix UCX UD timeout on GB200
- Root cause fix in UCX/driver for IB UD transport timeout
- Would allow default UCX PML to work with `privileged: true`
- **Pros**: Fixes everything at the source
- **Cons**: Hardware/driver issue, not in our control
- **Status**: Known issue, no ETA

### Option G: Sara's pattern — `IPC_LOCK` + `UCX_TLS=tcp` + NCCL built-in IB
- From [k8s-nccl-test B200](https://github.com/sara4dev/k8s-nccl-test/tree/b200)
- `UCX_TLS=tcp` for MPI, NCCL uses built-in IB transport (not UCX)
- `IPC_LOCK` + `SYS_RESOURCE` capabilities for RDMA memory registration
- **Tested**: MPI works, but NIXL also gets `UCX_TLS=tcp` (no RoCE)
- Same issue as Option D — global UCX_TLS affects NIXL
- **Status**: Tested, NIXL falls back to TCP

---

## Recommendation Priority

1. **Immediate**: Option A (`runAsUser: 0`) — get end-to-end inference working today
2. **Short-term**: Option B (TRT-LLM `safe_allgather`) — one PR to TRT-LLM, fixes everything
3. **Medium-term**: Option E (NIXL `ucx_engine_config`) — proper RoCE isolation for NIXL
4. **Long-term**: Option F (fix UCX UD on GB200) — root cause, eliminates all workarounds

---

## Configuration Reference

### What works for P2P weight transfer (all scenarios):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_ipc,cuda_copy,tcp"
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
```

### What works for single-node inference (TP=4):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_copy,tcp"  # NO cuda_ipc
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
```

### What works for multinode inference (Karen's approach, no RoCE P2P):
```yaml
securityContext:
  runAsUser: 0
# No UCX_TLS, no OMPI_MCA overrides
# NCCL handles TP collectives via NVIDIA driver
# P2P falls back to TCP (~3 Gbps)
```

### What works for multinode P2P + startup (not inference):
```yaml
securityContext:
  privileged: true
env:
  UCX_TLS: "rc_v,rc_x,rc,dc_x,dc,cuda_ipc,cuda_copy,tcp"
  OMPI_MCA_pml: "ob1"
  OMPI_MCA_btl: "tcp,self,vader"
# P2P at 370-610 Gbps, autotuning works, inference hits MPI_ERR_TRUNCATE
```

### What we tried and failed:

| Config | P2P | MPI Init | Inference | Why it failed |
|--------|-----|----------|-----------|---------------|
| `privileged` + `ob1` + TCP BTL | RoCE 370+ Gbps | Works | `MPI_ERR_TRUNCATE` | TCP BTL can't handle large allgather |
| `privileged` + no OMPI overrides | N/A | Hangs | N/A | UCX UD timeout on GB200 |
| `runAsUser: 0` + no overrides | TCP ~3 Gbps | Works | Works (Karen) | No RoCE without IB devices |
| `IPC_LOCK` + `UCX_TLS=tcp` | TCP ~3 Gbps | Works | Untested | NIXL also gets TCP |
| `TLLM_DISABLE_MPI=1` | N/A | N/A | Ray not installed | Switches to Ray executor |
| `UCX_TLS=tcp,cuda_ipc,...` + `privileged` | N/A | Hangs | N/A | UCX confused by visible IB |
| No env overrides + operator mpirun | N/A | `btl_tcp` errors | N/A | Process ID mismatch |
| `runAsUser:0` + `IPC_LOCK` + no OMPI env | N/A | Inner MpiSession hangs | N/A | TRT-LLM's internal MPI picks `169.254.x.x` link-local |
| `runAsUser:0` + `OMPI_MCA_*` env + no UCX fixes | Weights matched 1154/1815 | MPI works | `NIXL_ERR_BACKEND` | NIXL `loadRemoteMD` connects to `169.254.4.6` |
| `runAsUser:0` + `OMPI_MCA_*` + `UCX_NET_DEVICES=eth0` | Same | MPI works | `NIXL_ERR_BACKEND` | Source published metadata with `169.254.4.6` baked in |

### Issue #7: NIXL source address bound to link-local `169.254.x.x`
**Status**: INVESTIGATING — needs source redeployment with `UCX_NET_DEVICES=eth0`

The source's `nixl_transfer.py` temporarily removes `UCX_TLS` for NIXL agent creation to auto-detect
RoCE. However, without `UCX_NET_DEVICES=eth0`, the NIXL agent binds to ALL interfaces including
`169.254.x.x` link-local addresses. This address gets baked into the metadata published to Redis.
When target pods try to `add_remote_agent(source_metadata)`, NIXL/UCX attempts to connect to
`169.254.4.6` which is unreachable from other pods.

**Fix**: Add `UCX_NET_DEVICES=eth0` to source DGD YAML, flush Redis, and redeploy.

**Key learnings from this debugging session (multinode disagg + MX P2P)**:
1. Inner MPI session (`start MpiSession with 8 workers`) needs `OMPI_MCA_pml=ob1` + `btl_tcp_if_include=eth0` as **env vars** (not just mpirun flags)
2. `UCX_TCP_IF_INCLUDE` is NOT a valid UCX env var (UCX warns "unused")
3. `UCX_NET_DEVICES=eth0,eth2,eth3,eth4,eth5` is the correct way to restrict UCX interface selection (exclude 169.254.x.x link-local)
4. Source NIXL metadata contains the transport address — must be routable from target pods
5. Karen's disagg recipe works without MX P2P because it loads from shared PVC directly
6. DGD operator exports env vars via mpirun `-x` from `collectAllEnvVars()` — includes all vars set in container env spec
7. MoE backend must match between source and target (TRTLLM vs WIDEEP causes garbage output)
8. `MX_TRANSFER_TIMEOUT` env var added (default 900s) — TCP transfers of 90 GB can exceed the old 300s hardcoded timeout

### Issue #8: RoCE not activating despite correct config
**Status**: RESOLVED — UCX_IB_GID_INDEX=3 was the missing piece

**Fix**: `UCX_TLS=self,sm,rc,cuda_copy,gdr_copy,tcp` + `UCX_IB_GID_INDEX=3` + `TRTLLM_UCX_INTERFACE=eth0`
**Result**: 360-430 Gbps RoCE across all 8 ranks (90.75 GB in 1.69-2.02s per rank)

UCX selects TCP even with all correct config:
- `privileged: true`, `ulimit -l unlimited`
- `UCX_TLS=rc_v,rc_x,rc,dc_x,dc,cuda_copy,cuda_ipc,tcp` propagated to MPI workers
- `UCX_NET_DEVICES=eth0,eth2,eth3,eth4,eth5`
- IB devices active (`ibv_devinfo` shows `PORT_ACTIVE`)
- UCX loads `libuct_ib_mlx5.so` and `libuct_rdmacm.so`
- All 4 mlx5 devices visible

UCX wireup logs show `tcp/eth4` on all lanes with `software emulation`.
Previous TP=4 aggregated test (plain Deployment, not DGD) achieved 369 Gbps RoCE on same nodes.

## RoCE Debug Plan

### Problem Summary
Two NIXL users in the same process need different UCX transport configs:
1. **MX P2P weight transfer** (modelexpress `nixl_transfer.py`) — wants RoCE for fast weight transfer
2. **TRT-LLM KV cache transceiver** (C++ `NixlTransferAgent`) — needs TCP-safe config for prefill↔decode

Setting `UCX_TLS` with RoCE globally breaks TRT-LLM's KV cache NIXL (hangs on inference).
Setting `UCX_TLS=tcp` globally limits weight transfer to ~3-4 Gbps.

### Root Causes Identified

#### RC1: TRT-LLM `getAvailableIP()` picks `169.254.x.x`
**Repo**: `TensorRT-LLM`
**File**: `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp:119-158`

`getAvailableIP()` iterates interfaces and returns the FIRST IPv4 address, only skipping
`docker*` and `lo`. Does NOT skip link-local `169.254.x.x`. With `hostNetwork: true` on
GB200 nodes, `169.254.x.x` appears before `eth0` and gets baked into the NIXL agent address.

**Fix**: Set `TRTLLM_UCX_INTERFACE=eth0` env var (checked at line 125-129 via `getEnvNixlInterface()`).
This forces TRT-LLM's KV cache NIXL to bind to `eth0` specifically.

#### RC2: UCX selects TCP even with RC transports requested
**Repo**: UCX / NIXL
**Evidence**: `UCX_TLS=rc_v,rc_x,...,tcp`, `privileged: true`, `ulimit -l unlimited`,
`ibv_devinfo` shows `PORT_ACTIVE`, UCX loads `libuct_ib_mlx5.so` — yet wireup selects `tcp/ethX`.

**Hypotheses**:
- (a) UCX `rc_v`/`rc_x` requires explicit `rdma/ib` Kubernetes resource requests
  (Sara's NCCL test uses `rdma/ib: "8"` resource limit)
- (b) NIXL's UCX agent config overrides transport selection
  (`nixlAgentConfig` may force TCP internally)
- (c) GKE container runtime blocks IB verbs even with `privileged: true`
  (need to test `ibv_rc_pingpong` inside the pod)
- (d) Missing GDRCopy or CUDA IPC configuration for GPU-direct RDMA

#### RC3: Two NIXL agents share same UCX process context
**Repos**: `modelexpress` + `TensorRT-LLM`

Our `nixl_transfer.py` temporarily overrides `UCX_TLS` before creating the MX NIXL agent,
then restores it. But UCX caches transport resources per-process — the first agent's context
affects the second. TRT-LLM creates its KV cache NIXL agent later in the same process.

### Debug Plan

#### Phase 1: Isolate UCX RC transport (standalone test in DGD pod)
**Goal**: Determine if UCX RC verbs work at all inside a DGD-managed pod
**Repos**: None (manual test)

```bash
# Exec into decode leader pod
kubectl exec -it <decode-ldr-pod> -c main -- bash

# Test 1: IB verbs directly
ibv_rc_pingpong -d mlx5_0 -g 0  # Run on source node
ibv_rc_pingpong -d mlx5_0 -g 0 <source-ip>  # Run on target node

# Test 2: UCX perftest with RC
ucx_perftest -t tag_bw -m cuda -s 1048576  # Server on source
ucx_perftest <source-ip> -t tag_bw -m cuda -s 1048576  # Client on target

# Test 3: Check UCX transport availability
ucx_info -d | grep -i "rc_mlx5\|rc_v\|Transport"
```

#### Phase 2: Fix TRT-LLM KV cache NIXL interface binding
**Goal**: TRT-LLM KV cache NIXL binds to `eth0`, not `169.254.x.x`
**Repo**: `TensorRT-LLM`
**File**: `cpp/tensorrt_llm/executor/cache_transmission/nixl_utils/transferAgent.cpp`

- **Quick fix**: Add `TRTLLM_UCX_INTERFACE=eth0` to DGD YAML env vars
- **Proper fix**: `getAvailableIP()` should skip link-local addresses (`169.254.x.x`)
  Add check: `if (strncmp(address_buffer, "169.254.", 8) == 0) continue;`

#### Phase 3: Add `rdma/ib` resource requests
**Goal**: Ensure RDMA device plugin properly allocates IB devices to pods
**Repo**: `modelexpress` (YAML configs) + `dynamo` (operator)

Sara's NCCL test uses `rdma/ib: "8"` in resource limits. Our DGD YAMLs don't request
this resource. The RDMA shared device plugin may not expose IB devices properly without it.

```yaml
resources:
  limits:
    gpu: "4"
    rdma/ib: "4"   # <-- add this
```

Check if `rdma/ib` resource is available: `kubectl describe node <gpu-node> | grep rdma`

#### Phase 4: NIXL agent UCX context isolation
**Goal**: MX P2P NIXL uses RoCE, TRT-LLM KV cache NIXL uses TCP, independently
**Repos**: `modelexpress` + `nixl`

Options:
- (a) **NIXL agent-level transport config**: Check if `nixlAgentConfig` supports
  per-agent `UCX_TLS` override (not env var based). Look at NIXL's `_api.py` and
  `nixl_agent.cpp` for backend config params.
- (b) **Process isolation**: Run MX P2P transfer in a subprocess with different
  UCX env vars, then exit. TRT-LLM's NIXL agent creates in the main process
  with TCP config afterward.
- (c) **Sequential override**: Ensure MX NIXL agent is created and transfer
  completes BEFORE TRT-LLM creates its KV cache NIXL agent. Currently both
  happen inside `LLM()` constructor — weight load (our NIXL) then executor
  setup (TRT-LLM NIXL).

#### Phase 5: Dynamo operator env var propagation audit
**Goal**: Ensure all UCX/NIXL env vars reach MPI workers
**Repo**: `dynamo`
**File**: `deploy/operator/internal/dynamo/backend_trtllm.go:251-278`

`collectAllEnvVars()` already includes all container env vars in mpirun `-x` flags.
Verified working. Add `TRTLLM_UCX_INTERFACE` to the common vars list for convenience:

```go
func getCommonTRTLLMEnvVars() map[string]bool {
    return map[string]bool{
        // ... existing vars ...
        "TRTLLM_USE_UCX_KVCACHE": true,
        "TRTLLM_UCX_INTERFACE": true,  // <-- add
        "UCX_TLS": true,               // <-- add
        "UCX_NET_DEVICES": true,       // <-- add
    }
}
```

### Immediate Next Steps (priority order)

1. **Set `TRTLLM_UCX_INTERFACE=eth0`** in DGD YAML → fixes KV cache NIXL `169.254.x.x` → allows RoCE `UCX_TLS` without breaking inference
2. **Revert to `UCX_TLS=tcp` + test inference** → validate MoE backend fix produces correct output
3. **Run `ibv_rc_pingpong` inside DGD pod** → confirm IB verbs work at all
4. **Check `rdma/ib` resource** → may be required for proper RDMA device exposure
5. **Test with `TRTLLM_UCX_INTERFACE=eth0` + RoCE `UCX_TLS`** → should fix both issues simultaneously
