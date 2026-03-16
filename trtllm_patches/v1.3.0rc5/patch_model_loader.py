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

# Patch 2: Skip module post_load_weights for PRESHARDED with empty weights
old2 = """            for module in model.modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()"""

new2 = """            skip_post_load = (load_format == LoadFormat.PRESHARDED and not weights)
            if not skip_post_load:
                for module in model.modules():
                    if hasattr(module, 'post_load_weights') and not getattr(
                            module, '_weights_removed', False):
                        module.post_load_weights()
            else:
                if hasattr(model, 'post_load_weights'):
                    model.post_load_weights()
                logger.info("PRESHARDED: skipping module post_load_weights (weights pre-processed via RDMA)")"""

if old2 in content:
    content = content.replace(old2, new2)
    print("patch_model_loader: patch 2 (skip post_load_weights) applied")
elif "skipping module post_load_weights" in content:
    print("patch_model_loader: patch 2 already applied")
else:
    print("patch_model_loader: WARNING — patch 2 target not found")

with open(target, "w") as f:
    f.write(content)
print("patch_model_loader: done")
