"""Patch model_loader.py for PRESHARDED P2P — add flag + conditional post_load_weights.

apply_patches.py already adds the PRESHARDED block with load_weights skip.
This patch adds:
1. _mx_p2p_weights_loaded flag in the existing else branch
2. Conditional post_load_weights based on flag
"""
import os

target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py"

with open(target) as f:
    content = f.read()

# Patch 1: Add flag to existing PRESHARDED else branch (created by apply_patches.py)
old1 = '''                else:
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")'''

new1 = '''                else:
                    model._mx_p2p_weights_loaded = True
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")'''

if old1 in content and "_mx_p2p_weights_loaded" not in content:
    content = content.replace(old1, new1)
    print("patch_model_loader: patch 1 (add flag to existing else) applied")
elif "_mx_p2p_weights_loaded" in content:
    print("patch_model_loader: patch 1 already applied")
else:
    print("patch_model_loader: WARNING — patch 1 target not found")

# Patch 2: Conditional post_load_weights based on P2P flag
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
