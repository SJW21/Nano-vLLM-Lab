import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

# Two checkpoint modules have no counterpart here and are skipped explicitly: the vision
# tower, and `mtp.*` (a training-time draft head that plain inference never executes).
# Everything else must resolve -- an unloaded parameter means an uninitialised model.
_IGNORED_PREFIXES = {
    "model.visual.": "vision-tower",
    "mtp.": "multi-token-prediction",
}
_TEXT_FRAGMENTS = ("language_model.",)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _parameter_name(params: dict, weight_name: str) -> str | None:
    """Map a checkpoint tensor name onto one of the model's parameter names."""
    candidates = [weight_name]
    for fragment in _TEXT_FRAGMENTS:            # model.language_model.x -> model.x
        if fragment in weight_name:
            candidates.append(weight_name.replace(fragment, "", 1))
    for candidate in candidates:
        if candidate in params:
            return candidate
    return None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    params = dict(model.named_parameters())
    loaded, skipped, unresolved = set(), [], []
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                if any(weight_name.startswith(p) for p in _IGNORED_PREFIXES):
                    skipped.append(weight_name)
                    continue
                name, shard_id = weight_name, None
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        name = weight_name.replace(k, v)
                        break
                param_name = _parameter_name(params, name)
                if param_name is None:
                    unresolved.append(weight_name)
                    continue
                param = params[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is None:
                    weight_loader(param, f.get_tensor(weight_name))
                else:
                    weight_loader(param, f.get_tensor(weight_name), shard_id)
                loaded.add(param_name)

    if unresolved:
        raise RuntimeError(
            f"{len(unresolved)} checkpoint weight(s) match no parameter "
            f"(first: {unresolved[:5]}). A new architecture or a renamed module "
            f"would land here -- do not load a model with weights missing. If the "
            f"weights belong to a module this engine does not implement, add its "
            f"prefix to _IGNORED_PREFIXES."
        )
    for prefix, reason in _IGNORED_PREFIXES.items():
        names = [w for w in skipped if w.startswith(prefix)]
        if names:
            print(f"[loader] skipped {len(names)} {reason} weight(s), "
                  f"e.g. {names[0]}")
    # tied parameters (lm_head sharing embed_tokens) never appear in the checkpoint.
    # NOTE: this catches a parameter that received *nothing* -- a fused parameter missing
    # only one shard (q_proj and v_proj present, k_proj absent) still counts as loaded.
    missing = [n for n in params if n not in loaded
               and not any(params[n].data_ptr() == params[m].data_ptr() for m in loaded)]
    if missing:
        raise RuntimeError(
            f"{len(missing)} parameter(s) received no weight: {missing[:8]}"
        )
