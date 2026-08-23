import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _remap_expert_weight(weight_name: str, ep_size: int, ep_rank: int,
                         num_experts: int, moe_mode: str = "ep") -> tuple[str | None, int | None]:
    """Remap expert weight name for EP filtering with stacked parameters.

    In EP mode, each rank only loads a subset of experts.  The HF checkpoint
    names experts by global ID (0..num_experts-1).  This function maps the
    global expert ID to the local expert ID and a stacked-parameter name
    (``stacked_gate`` / ``stacked_up`` / ``stacked_down``) so the weight
    loader can write directly into the correct slot of the batched tensor.

    Args:
        weight_name: HF weight name (e.g. "model.layers.0.mlp.experts.42.gate_proj.weight").
        ep_size: expert parallel size.
        ep_rank: rank within the EP group.
        num_experts: total number of experts in the model.
        moe_mode: "ep"（默认）→ 按 owner 过滤专家；"tp" → 不跳过专家，映射
            stacked_*_tp，M 维 narrow 份数 = EP（= DP×TP），切片索引 = rank
            在 EP group 内位置。tp + ep_size==1 走下方既有逻辑（个体专家、
            全量加载）。

    Returns:
        ``(param_name, expert_id)`` if the expert belongs to this rank,
        ``(None, None)`` if it should be skipped (belongs to another rank).
        For non-expert weights, returns ``(weight_name, None)`` as-is.
    """
    m = re.search(r"experts\.(\d+)\.", weight_name)
    if m is None:
        return weight_name, None  # Not an expert weight, load as-is.

    expert_id = int(m.group(1))

    if moe_mode == "tp" and ep_size > 1:
        # tp 模式：全专家本地持有，不按 owner 过滤；权重沿 M 维切 EP 份，
        # 切片由 param 的 weight_loader（M 维 narrow，切片索引 = ep_rank）完成。
        # expert_id 保持全局 id（0..num_experts-1），对应 stacked_*_tp 的行号。
        if 'gate_proj' in weight_name:
            new_name = re.sub(r'experts\.\d+\.gate_proj\.weight', 'stacked_gate_tp', weight_name)
        elif 'up_proj' in weight_name:
            new_name = re.sub(r'experts\.\d+\.up_proj\.weight', 'stacked_up_tp', weight_name)
        elif 'down_proj' in weight_name:
            new_name = re.sub(r'experts\.\d+\.down_proj\.weight', 'stacked_down_tp', weight_name)
        else:
            return weight_name, None
        return new_name, expert_id

    experts_per_rank = num_experts // ep_size
    local_id = expert_id % experts_per_rank
    owner_rank = expert_id // experts_per_rank

    if owner_rank != ep_rank:
        return None, None  # Skip: this expert belongs to another rank.

    if ep_size == 1:
        # Keep individual expert names (no stacked params for ep_size=1).
        return weight_name.replace(
            f"experts.{expert_id}.", f"experts.{local_id}."), None

    # Map to stacked parameter name:
    #   experts.{global_id}.gate_proj.weight → stacked_gate
    #   experts.{global_id}.up_proj.weight   → stacked_up
    #   experts.{global_id}.down_proj.weight → stacked_down
    if 'gate_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.gate_proj\.weight', 'stacked_gate', weight_name)
    elif 'up_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.up_proj\.weight', 'stacked_up', weight_name)
    elif 'down_proj' in weight_name:
        new_name = re.sub(r'experts\.\d+\.down_proj\.weight', 'stacked_down', weight_name)
    else:
        return weight_name, None  # shouldn't happen for expert weights

    return new_name, local_id


def load_model(model: nn.Module, path: str,
               ep_size: int = 1, ep_rank: int = 0,
               num_experts: int | None = None,
               moe_mode: str = "ep"):
    """Load model weights from safetensors with optional EP filtering.

    Args:
        model: the model to load weights into.
        path: directory containing .safetensors files.
        ep_size: expert parallel size (1 = no EP filtering).
        ep_rank: rank within the EP group.
        num_experts: total number of experts (needed for EP remapping).
                     If None, no EP filtering is applied.
        moe_mode: "ep"（默认）→ 专家按 owner 过滤；"tp" → 映射 stacked_*_tp
                  并沿 M 维切片（详见 _remap_expert_weight）。
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for hf_name in f.keys():
                # --- EP expert filtering ---
                # hf_name: original key in the checkpoint (never mutated).
                # param_name: target parameter name (may be remapped for EP).
                # expert_id: local expert index for stacked params (or None).
                param_name = hf_name
                expert_id = None
                if num_experts is not None and ep_size > 1:
                    param_name, expert_id = _remap_expert_weight(
                        hf_name, ep_size, ep_rank, num_experts, moe_mode
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
                    if expert_id is not None:
                        weight_loader(param, f.get_tensor(hf_name), expert_id)
                    else:
                        weight_loader(param, f.get_tensor(hf_name))
