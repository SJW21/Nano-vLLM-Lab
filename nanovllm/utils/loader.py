import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _remap_expert_weight(weight_name: str) -> tuple[str | None, int | None]:
    """Map an HF expert weight name to its stacked_*_tp parameter.

    The checkpoint names experts globally
    (experts.{0..E-1}.{gate,up,down}_proj.weight). Every rank holds all
    experts and one EP slice of the intermediate dimension, so the parameter
    name becomes stacked_gate_tp / stacked_up_tp / stacked_down_tp,
    expert_id stays the global id (the row in the stacked parameter), and the
    M-dim narrow happens in the parameter's weight_loader.

    Returns (param_name, expert_id); non-expert weights return
    (weight_name, None).
    """
    m = re.search(r"experts\.(\d+)\.", weight_name)
    if m is None:
        return weight_name, None  # Not an expert weight, load as-is.

    expert_id = int(m.group(1))

    if 'gate_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.gate_proj\.weight', 'stacked_gate_tp', weight_name)
    elif 'up_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.up_proj\.weight', 'stacked_up_tp', weight_name)
    elif 'down_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.down_proj\.weight', 'stacked_down_tp', weight_name)
    else:
        return weight_name, None
    return new_name, expert_id


def load_model(model: nn.Module, path: str, num_experts: int | None = None):
    """Load model weights from safetensors.

    Args:
        model: the model to load weights into.
        path: directory containing .safetensors files.
        num_experts: total number of experts for MoE models (expert weights
            are remapped to stacked_*_tp); pass None for dense models.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for hf_name in f.keys():
                # --- MoE expert remapping ---
                # hf_name: original key in the checkpoint (never mutated).
                # param_name: target parameter name (remapped for experts).
                # expert_id: expert row index for stacked params (or None).
                param_name = hf_name
                expert_id = None
                if num_experts is not None:
                    param_name, expert_id = _remap_expert_weight(hf_name)

                # --- Packed module mapping ---
                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = param_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(hf_name), shard_id)
                        break
                else:
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    if expert_id is not None:
                        weight_loader(param, f.get_tensor(hf_name), expert_id)
                    else:
                        weight_loader(param, f.get_tensor(hf_name))
