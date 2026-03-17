# Disaggregated Inference with TRT-LLM on GCP GB200

**Status**: Working (validated March 17 2026)
**Model**: Kimi K2.5 (DeepSeek V3, NVFP4, 256 experts)
**Config**: Prefill TP=8 (2 nodes) + Decode TP=8 (2 nodes), GCP GB200 NVL36

---

## Required Fixes for GCP GB200

Six fixes are needed to get disaggregated inference working on this cluster.
These apply to both PVC-loaded and MX P2P-loaded deployments.

### 1. SSH Host Key Path Mismatch (`HOME=/root`)

**Symptom**: Worker pods crash with `sshd: no hostkeys available -- exiting`

**Root cause**: The Dynamo operator injects a shell command that generates SSH
host keys at `~/.ssh/host_keys/` using `ssh-keygen`. The base image sets
`HOME=/home/dynamo`, so keys land at `/home/dynamo/.ssh/host_keys/`. But `sshd`
resolves `~` via `getpwuid(0)` → `/root`, so it looks for keys at
`/root/.ssh/host_keys/` and fails.

**Fix**: Set `HOME=/root` as an env var on all worker pods.

```yaml
env:
  - name: HOME
    value: /root
```

### 2. Autotuner Warmup MPI Desync (`enable_autotuner: false`)

**Symptom**: Hang detected after 300s, then `TypeError: '<' not supported
between instances of 'list' and 'int'` in `calculate_max_num_blocks`

**Root cause**: The warmup request during autotuning completes on some ranks
before others. Ranks that finish warmup enter the executor loop and call
`allgather`, while other ranks are still in `configure_kv_cache_capacity`
calling `allreduce`. Different MPI collectives on different ranks = deadlock.

**Fix**: Disable autotuner in both prefill and decode engine configs.

```yaml
enable_autotuner: false
```

### 3. KV Cache Transceiver UCX UD Timeout (`backend: DEFAULT`)

**Symptom**: Prefill step takes ~102 seconds (the KV transfer timeout),
`storeContextBlocks: Can not find sequence for request` warnings

**Root cause**: TRT-LLM's UCX-based KV cache transceiver uses UCX UD
(Unreliable Datagram) for initial connection establishment between prefill
and decode NIXL agents. On this GCP GB200 cluster, UD packets are dropped
and never ACK'd. After 30 seconds, endpoints time out:
```
ud_ep.c:392 UCX DEBUG ep: timeout of 30.99 sec, config::peer_timeout - 30.00 sec
ucp_worker.c:545 UCX DEBUG worker: error handler called for UCT EP: Endpoint timeout
```

**Fix**: Use `DEFAULT` backend instead of `UCX` for the cache transceiver.
The DEFAULT backend uses NIXL's built-in transceiver which handles peer
discovery differently, avoiding the UCX UD bootstrap.

```yaml
cache_transceiver_config:
  backend: DEFAULT
  max_tokens_in_buffer: 240000
```

### 4. NATS/ETCD Environment Variables

**Symptom**: Frontend crashes with `Failed to connect to NATS: timed out`

**Root cause**: The Dynamo operator doesn't inject NATS/ETCD env vars into
custom DGD pod specs. They must be set explicitly.

**Fix**: Add to all pods (frontend, prefill, decode):

```yaml
env:
  - name: NATS_SERVER
    value: "nats://dynamo-platform-nats.<namespace>.svc.cluster.local:4222"
  - name: ETCD_ENDPOINTS
    value: "dynamo-platform-etcd.<namespace>.svc.cluster.local:2379"
```

### 5. MPI Bootstrap UCX UD Workaround (`ob1` PML)

**Symptom**: MPI session hangs during multinode bootstrap, trying to
connect to `169.254.x.x` link-local addresses

**Root cause**: OpenMPI's UCX PML uses UD transport for session setup,
which hits the same UD timeout as fix #3.

**Fix**: Force TCP-based MPI transport:

```yaml
env:
  - name: OMPI_MCA_pml
    value: "ob1"
  - name: OMPI_MCA_btl
    value: "tcp,self,vader"
  - name: OMPI_MCA_btl_tcp_if_include
    value: "eth0"
  - name: OMPI_MCA_oob_tcp_if_include
    value: "eth0"
```

### 6. RoCE Configuration for RDMA

**Symptom**: NIXL transfers fail or fall back to TCP

**Fix**: Configure UCX for RoCE and prevent link-local binding:

```yaml
env:
  - name: UCX_TLS
    value: "self,sm,rc,cuda_copy,gdr_copy,tcp"
  - name: UCX_IB_GID_INDEX
    value: "3"              # Critical for GCP RoCEv2
  - name: TRTLLM_UCX_INTERFACE
    value: eth0              # Prevents 169.254.x.x binding
securityContext:
  privileged: true           # Required for RDMA mem registration
  runAsUser: 0               # Fixes SSH key path (see fix #1)
```

---

## PVC Load Time Baseline (TP=8, 2 nodes per service)

| Phase | Prefill | Decode |
|-------|---------|--------|
| Model init (meta + CUDA alloc) | ~25s | ~25s |
| Weight loading from PVC | ~1200s (~20 min) | ~1200s (~20 min) |
| post_load_weights | ~120s | ~120s |
| NCCL init + CUDA graphs | ~110s | ~100s |
| **Total (Model init total)** | **~1453s (~24 min)** | **~821-1019s (~14-17 min)** |

---

## Deployment YAML

See `examples/p2p_transfer_trtllm/deploy/gcp/kimi-disagg-baseline-dgd.yaml`

Key config differences from Karen's recipe:
- `cache_transceiver_config.backend: DEFAULT` (not UCX)
- `enable_autotuner: false`
- `HOME=/root` env var
- `NATS_SERVER` / `ETCD_ENDPOINTS` explicit env vars
- Uses our `v1.8.0` image with MPI allgather fix

---

## Inference Results

```
TTFT: 640ms (prefill) → 480ms (subsequent)
Total: 2259ms (50 tokens) → 2877ms (100 tokens)

Prefill DP rank: varies (load balanced across 8 attention DP ranks)
Decode DP rank: varies (load balanced across 8 attention DP ranks)
KV hit rate: 0.0 (no cache reuse for first requests)
```
