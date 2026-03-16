"""Patch model_loader.py for PRESHARDED P2P RDMA weight loading.

Two patches:
1. Skip model.load_weights({}) when PRESHARDED returns empty dict
2. Skip module post_load_weights() which recalculates quant scales and
   pads weights — corrupts pre-processed RDMA-transferred weights
"""
import os

target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py"

with open(target) as f:
    content = f.read()

# Patch 1: Skip model.load_weights({}) for PRESHARDED
old1 = """                self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                    model, config)
                self._call_load_weights(model.load_weights, weights,
                                        self.weight_mapper)"""

new1 = """                if weights:
                    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                        model, config)
                    self._call_load_weights(model.load_weights, weights,
                                            self.weight_mapper)
                else:
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")"""

if old1 in content:
    content = content.replace(old1, new1)
    print("patch_model_loader: patch 1 (skip load_weights) applied")
elif "PRESHARDED: weights injected directly" in content:
    print("patch_model_loader: patch 1 already applied")
else:
    print("patch_model_loader: WARNING — patch 1 target not found")

# Patch 2: REMOVED — post_load_weights must run for both source and target.
# Source needs it for MoE load balancer setup, next_layer_layernorm aliases, etc.
print("patch_model_loader: patch 2 skipped (post_load_weights must always run)")

with open(target, "w") as f:
    f.write(content)
print("patch_model_loader: done")
