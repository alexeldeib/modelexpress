"""Patch model_loader.py to skip model.load_weights() when PRESHARDED returns empty dict.

When P2P RDMA transfers weights directly into model params and returns {},
the original code still calls model.load_weights({}) which iterates all modules
and overwrites RDMA-transferred weights with empty data.
"""
import os

target = "/opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/_torch/pyexecutor/model_loader.py"

with open(target) as f:
    content = f.read()

old = """                self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                    model, config)
                self._call_load_weights(model.load_weights, weights,
                                        self.weight_mapper)"""

new = """                if weights:
                    self.weight_mapper = checkpoint_loader.get_initialized_weight_mapper(
                        model, config)
                    self._call_load_weights(model.load_weights, weights,
                                            self.weight_mapper)
                else:
                    logger.info("PRESHARDED: weights injected directly, skipping load_weights()")"""

if old in content:
    content = content.replace(old, new)
    with open(target, "w") as f:
        f.write(content)
    print("patch_model_loader: PRESHARDED skip applied")
else:
    if "PRESHARDED: weights injected directly" in content:
        print("patch_model_loader: already applied")
    else:
        print("patch_model_loader: WARNING — target code not found, patch not applied")
