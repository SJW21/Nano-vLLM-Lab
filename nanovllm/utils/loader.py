import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _remap_expert_weight(weight_name: str, ep_size: int, ep_rank: int,
                         num_experts: int) -> str | None:
    """Remap expert weight name for EP filtering.

    In EP mode, each rank only loads a subset of experts. The HF checkpoint
    names experts by their global ID (0..num_experts-1); this function maps
    the global expert ID to the local expert ID for the current EP rank.

    Args:
        weight_name: HF weight name (e.g. "model.layers.0.mlp.experts.42.gate_proj.weight").
        ep_size: expert parallel size.
        ep_rank: rank within the EP group.
        num_experts: total number of experts in the model.

    Returns:
        Remapped weight name if the expert belongs to this rank, None if it
        should be skipped (belongs to another rank).
    """
    m = re.search(r"experts\.(\d+)\.", weight_name)
    if m is None:
        return weight_name  # Not an expert weight, load as-is.

    expert_id = int(m.group(1))
    experts_per_rank = num_experts // ep_size
    local_id = expert_id % experts_per_rank
    owner_rank = expert_id // experts_per_rank

    if owner_rank != ep_rank:
        return None  # Skip: this expert belongs to another rank.

    return weight_name.replace(f"experts.{expert_id}.", f"experts.{local_id}.")


def load_model(model: nn.Module, path: str,
               ep_size: int = 1, ep_rank: int = 0,
               num_experts: int | None = None):
    """Load model weights from safetensors with optional EP filtering.

    Args:
        model: the model to load weights into.
        path: directory containing .safetensors files.
        ep_size: expert parallel size (1 = no EP filtering).
        ep_rank: rank within the EP group.
        num_experts: total number of experts (needed for EP remapping).
                     If None, no EP filtering is applied.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for hf_name in f.keys():
                # --- EP expert filtering ---
                # hf_name: original key in the checkpoint (never mutated).
                # param_name: target parameter name (may be remapped for EP).
                param_name = hf_name
                if num_experts is not None and ep_size > 1:
                    param_name = _remap_expert_weight(
                        hf_name, ep_size, ep_rank, num_experts
                    )
                    if param_name is None:
                        continue  # Skip expert belonging to another rank

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
                    weight_loader(param, f.get_tensor(hf_name))
