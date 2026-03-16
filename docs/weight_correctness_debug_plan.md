# P2P Weight Transfer — Correctness Debug Plan

## Problem
RoCE P2P at 430 Gbps is validated. 1754/1815 tensors transfer by name match.
PVC-loaded model produces correct output. P2P-loaded model produces garbage.

## What We've Eliminated
| Hypothesis | Status | Evidence |
|---|---|---|
| `model.load_weights({})` overwrites RDMA weights | DISPROVEN | Skipped via patch, still garbage |
| `post_load_weights()` recalculates quant scales | DISPROVEN | Skipped via patch, still garbage |
| MoE backend mismatch (TRTLLM vs WIDEEP) | DISPROVEN | Both use TRTLLM |
| Unmatched tensors cause missing weights | DISPROVEN | 61 unmatched are aliases (next_layer_layernorm) |
| Size/dtype mismatch (e_score_correction_bias) | DISPROVEN | Now matches (768 bytes bf16 both sides) |
| UCX transport corruption | UNLIKELY | 430 Gbps RoCE, all transfers complete without error |

## Active Investigation (v1.7.7)
Checksum logging added to both source publish and target receive.
Compare `sum()` and `nonzero()` for first 5 params on rank 0.

## Debug Steps (in priority order)

### Step 1: Verify RDMA Data Correctness
**Image v1.7.7 has this diagnostic.**

Compare source checksums (logged during `publish_from_worker`) with target checksums
(logged after `receive_from_source`). Check `/tmp/mx_logs/rank0.log` on target and
source pod stdout.

- **If checksums match**: RDMA is correct. Bug is in model structure/forward pass.
- **If checksums differ**: RDMA writes to wrong addresses or data corrupted.
- **If target shows all zeros**: RDMA didn't actually write (async issue, wrong device).

### Step 2: Compare PVC-Loaded vs P2P-Loaded GPU Memory
Deploy BOTH PVC and P2P targets simultaneously (if GPU capacity allows) or sequentially.
Exec into each and run:

```python
# Dump checksums of ALL params to a file
import torch, json
checksums = {}
for name, param in model.named_parameters():
    if param.device.index == 0:
        checksums[name] = {
            'sum': param.to(torch.float32).sum().item(),
            'nonzero': (param != 0).sum().item(),
            'numel': param.numel(),
            'first3': param.flatten()[:3].to(torch.float32).tolist(),
        }
json.dump(checksums, open('/tmp/param_checksums.json', 'w'), indent=2)
```

Problem: model runs in MPI worker process, not accessible from `kubectl exec`.
**Fix**: Modify `publish_from_worker` (source) and `load_weights` (target) to dump
checksums to `/tmp/param_checksums_source.json` and `/tmp/param_checksums_target.json`.
Then `kubectl cp` the files out and diff.

### Step 3: Verify `named_parameters()` Returns Live Params
The model's `named_parameters()` might return nn.Parameter objects that are not the
actual buffers used during forward pass. TRT-LLM's MoE backend (`TRTLLMGenFusedMoE`)
might create internal CUDA buffers for kernels that are separate from nn.Parameters.

Check in `fused_moe_trtllm_gen.py`:
- Does `forward()` use `self.w3_w1_weight` (nn.Parameter) directly?
- Or does it copy to an internal buffer during init/first-forward?
- Does `load_weights()` on the MoE module set up kernel-specific buffers
  that `named_parameters()` doesn't capture?

**Files to check:**
- `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` — forward() weight access
- `tensorrt_llm/_torch/modules/fused_moe/quantization.py` — weight setup
- `tensorrt_llm/_torch/modules/fused_moe/interface.py` — load_weights()

### Step 4: Check if MoE `load_weights()` Creates Processed Copies
The MoE module's `load_weights([weights_dict])` might:
1. Take raw weights from the dict
2. Permute/rearrange them for the GEMM kernel (row reordering for FP4)
3. Store the processed version in a SEPARATE buffer
4. The nn.Parameter holds the raw version, the processed buffer is used in forward

If this is the case, our RDMA writes the source's PROCESSED weight (after permutation)
into the target's RAW parameter buffer. The target then doesn't re-process it (since we
skip load_weights), so forward() uses the parameter directly — but expects the RAW format.

**OR**: the source's `named_parameters()` returns the PROCESSED buffer, and the target's
`named_parameters()` returns the RAW buffer (same name, same size, different data layout).

This would explain everything:
- Same name, same size → matched
- RDMA transfers bytes correctly
- But the data layout in the tensor is wrong for the target

**Test**: Check `quantization.py:setup_quant_scales()` and any weight permutation code
in `load_weights()`. Look for `get_reorder_rows_for_gated_act_gemm_row_indices` or
`get_shuffle_matrix_a_row_indices` which permute FP4 weight rows.

### Step 5: TP=4 vs TP=8 Difference
The TP=4 aggregated test reportedly produced correct output with P2P. Verify:
1. Was the TP=4 test on the same base image (same TRT-LLM version)?
2. Did TP=4 use the same MoE backend (TRTLLM)?
3. Did TP=4 actually produce coherent output (not just non-crashing)?
4. Re-run TP=4 P2P test with v1.7.7 to confirm it still works

### Step 6: Minimal Reproduction
Deploy a smaller model (Qwen 0.5B, no MoE) with P2P to verify the basic
transfer mechanism works. If Qwen produces correct output, the bug is
MoE-specific weight handling.

## Key Files
| File | Role |
|---|---|
| `modelexpress/trtllm_live_transfer.py` | P2P source publish + target load |
| `modelexpress/nixl_transfer.py` | NIXL RDMA transfer |
| `TRT-LLM/model_loader.py:297-320` | PRESHARDED load path |
| `TRT-LLM/modeling_deepseekv3.py:152-600` | DeepseekV3WeightLoader |
| `TRT-LLM/fused_moe/quantization.py` | MoE weight quantization + setup |
| `TRT-LLM/fused_moe/fused_moe_trtllm_gen.py` | MoE forward pass + weight access |
| `TRT-LLM/fused_moe/interface.py` | MoE load_weights interface |

## Current Deployment
- Source: v1.7.7 with checksum logging
- Target: v1.7.7 with checksum logging + PRESHARDED skips
- Config: TP=8, EP=8, MoE backend=TRTLLM, 384 experts, 61 layers
- UCX: `self,sm,rc,cuda_copy,gdr_copy,tcp` + `UCX_IB_GID_INDEX=3`
