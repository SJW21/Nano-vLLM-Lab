import torch
from torch.distributed import ProcessGroup
import torch.distributed as dist

_group_name_counter: dict[str, int] = {}

def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


class GroupCoordinator:
    """PyTorch ProcessGroup wrapper for a group of processes (as in vLLM). Only CUDA
    is supported, and only one node: rank == local_rank (see ``device_group``).
    """

    # supported attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    rank_in_group: int  # rank inside the group
    device_group: ProcessGroup

    def __init__(self,
                 group_ranks: list[list[int]],
                 dist_backend: str,
                 group_name: str | None = None):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)

        self.rank = dist.get_rank()
        for ranks in group_ranks:
            device_group = dist.new_group(ranks, backend=dist_backend)
            # set specific commu group to exact rank
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)  # world size of exact PG
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group

        assert self.device_group is not None

    @property
    def local_rank(self):
        # rank in the group
        return self.rank_in_group

    def all_reduce(self, input_:torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        dist.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

_TP: GroupCoordinator | None = None
_WORLD: GroupCoordinator | None = None


def get_tcp_uri(ip: str, port: int) -> str:
    return f"tcp://{ip}:{port}"


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    backend: str = "nccl"
):
    # init the WORLD Group 
    dist_init_method = get_tcp_uri("localhost", 2333)
    dist.init_process_group(backend=backend,
                            init_method=dist_init_method,
                            world_size=world_size,
                            rank=rank,
                            device_id=torch.device('cuda', rank))
    # set World Group
    global _WORLD
    assert torch.distributed.get_world_size() == world_size, "ProcessGroup init failed."
    ranks = list(range(world_size))
    _WORLD = GroupCoordinator([ranks], backend, group_name="world")


def initialize_model_parallel(
    world_size: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    backend: str | None = None
):
    """Build the tensor-parallel groups (DP is done over ZMQ, see the DP controller)."""
    assert torch.distributed.is_initialized()
    assert world_size == tensor_parallel_size * data_parallel_size, \
        "World Size must satisfy DP x TP now."

    all_ranks = torch.arange(world_size).reshape(
        data_parallel_size,
        tensor_parallel_size
    )
    backend = backend or torch.distributed.get_backend(
            get_world_group().device_group
    )

    # Build the tensor model-parallel groups.
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = all_ranks.view(-1, tensor_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    _TP = GroupCoordinator(group_ranks=group_ranks,
                           dist_backend=backend,
                           group_name="tp")


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor parallel group is not initialized"
    return _TP