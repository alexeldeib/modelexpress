"""Patch model_loader.py for PRESHARDED P2P RDMA weight loading.

Two patches on the SECOND PRESHARDED block (the one that actually runs):
1. Skip model.load_weights({}) when empty + set _mx_p2p_weights_loaded flag
2. Conditional post_load_weights based on flag
"""
import os

target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py"

with open(target) as f:
    content = f.read()

# Patch 1: The SECOND PRESHARDED block (with tp_size check)
old1 = """            elif load_format == LoadFormat.PRESHARDED:
                for module in model.modules():
                    if hasattr(module, 'tp_size'):
                        module._weights_presharded = True
                weights = checkpoint_loader.load_weights(
                    checkpoint_dir, mapping=self.mapping, model=model)
                if weights:
                    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                        model, config)
                    self._call_load_weights(model.load_weights, weights,
                                            self.weight_mapper)"""

new1 = """            elif load_format == LoadFormat.PRESHARDED:
                for module in model.modules():
                    if hasattr(module, 'tp_size'):
                        module._weights_presharded = True
                weights = checkpoint_loader.load_weights(
                    checkpoint_dir, mapping=self.mapping, model=model)
                if weights:
                    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                        model, config)
                    self._call_load_weights(model.load_weights, weights,
                                            self.weight_mapper)
                else:
                    model._mx_p2p_weights_loaded = True
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")"""

if old1 in content:
    content = content.replace(old1, new1)
    print("patch_model_loader: patch 1 (PRESHARDED skip + flag) applied")
elif "_mx_p2p_weights_loaded" in content and "elif load_format == LoadFormat.PRESHARDED" in content:
    print("patch_model_loader: patch 1 may already be applied")
else:
    print("patch_model_loader: WARNING — patch 1 target not found")

# Patch 2: Conditional post_load_weights
old2 = """            for module in model.modules():
                if hasattr(module, 'post_load_weights') and not getattr(
                        module, '_weights_removed', False):
                    module.post_load_weights()"""

new2 = """            if getattr(model, '_mx_p2p_weights_loaded', False):
                if hasattr(model, 'post_load_weights'):
                    model.post_load_weights()
                logger.info("PRESHARDED P2P: skipping module post_load_weights")
            else:
                for module in model.modules():
                    if hasattr(module, 'post_load_weights') and not getattr(
                            module, '_weights_removed', False):
                        module.post_load_weights()"""

if old2 in content:
    content = content.replace(old2, new2)
    print("patch_model_loader: patch 2 (conditional post_load_weights) applied")
elif "PRESHARDED P2P: skipping module post_load_weights" in content:
    print("patch_model_loader: patch 2 already applied")
else:
    print("patch_model_loader: WARNING — patch 2 target not found")

with open(target, "w") as f:
    f.write(content)
print("patch_model_loader: done")
