import torch
import ipaddress
from torch.distributed import ProcessGroup
import torch.distributed as dist
from nanovllm.utils.context import suppress_stdout

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
    """
    Refer to https://github.com/vllm-project/vllm
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It manages both CPU and device
        communication.    
    """
    # supported attributes:
    rank: int  # global rank
    ranks: list[int]  # global ranks in the group
    world_size: int  # size of the group
    rank_in_group: int  # rank inside the group
    # [NOTE] Assume only have one node | ie. rank == local_rank in vLLM v1
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    # device communicator (if use_device_communicator=True) 
    # [NOTE] device specific communicator
    device_communicator: None 

    def __init__(self,
                 group_ranks: list[list[int]],
                 dist_backend: str,
                 use_device_communicator: bool = False,
                 group_name: str | None = None):
        #[NOTE] Support device communicator if necessary
        assert not use_device_communicator, "Device communicator hasn't been supported yet."
        
        # set group Indentify ID
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)

        # set cpu / device group
        self.rank = dist.get_rank()
        for ranks in group_ranks:
            device_group = dist.new_group(
                ranks, backend=dist_backend
            )
            with suppress_stdout(): # suppress C lib print
                cpu_group = dist.new_group(ranks, backend='gloo')
            # set specific commu group to exact rank
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks) # world size of exact PG
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group
        
        assert self.cpu_group is not None
        assert self.device_group is not None
        #[NOTE] Only support CUDA Platform Now
        self.device = torch.device(f"cuda:{self.rank}")
    
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

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()
        gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(gather_list, input_, group=self.device_group)
        return torch.cat(gather_list, dim=dim)

    def send(self, tensor: torch.Tensor, dst: int| None = None) -> None:
        raise NotImplementedError

    def recv(self, tensor: torch.Tensor, dtype: torch.dtype, src: int| None = None) -> None:
        raise NotImplementedError
    

_TP: GroupCoordinator | None = None
_DP: GroupCoordinator | None = None
_EP: GroupCoordinator | None = None
_WORLD: GroupCoordinator | None = None

def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"
    else:
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
    #[NOTE] For Simplify, whether world group can be abandoned.
    global _WORLD
    #[BUG] Test whether initialization successful
    assert torch.distributed.get_world_size() == world_size, "ProcessGroup init failed."
    ranks = list(range(world_size))
    _WORLD = GroupCoordinator([ranks], backend, group_name="world")


def initialize_model_parallel(
    world_size: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    backend: str | None = None
):
    """
    Initialize model parallel groups.
    Arguments:
        tensor_parallel_size: number of GPUs used for tensor_parallel_size,
        data_parallel_size: number of GPUs for A Single DP Group,
        expert_parallel_size: number of GPUs for expert parallelism (EP=全体rank).
    """
    assert torch.distributed.is_initialized()
    assert world_size == tensor_parallel_size * data_parallel_size, \
        "World Size must satisfy DP x TP now."
    if expert_parallel_size > 1:
        assert expert_parallel_size == world_size, \
            "EP size must equal world_size (EP overlays all ranks)."

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

    # Build the data model-parallel groups.
    global _DP
    assert _DP is None, "data parallel group is already initialized"
    if data_parallel_size > 1:
        group_ranks = all_ranks.transpose(1, 0).reshape(-1, data_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        _DP = GroupCoordinator(group_ranks=group_ranks,
                               dist_backend=backend,
                               group_name="dp")

    # Build the expert model-parallel group.
    global _EP
    assert _EP is None, "expert parallel group is already initialized"
    if expert_parallel_size > 1:
        group_ranks = [list(range(world_size))]
        _EP = GroupCoordinator(group_ranks=group_ranks,
                               dist_backend=backend,
                               group_name="ep")



def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP


def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor parallel group is not initialized"
    return _TP


def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, "expert parallel group is not initialized"
    return _EP


def destroy_distributed_environment():
    global _WORLD, _TP, _DP, _EP
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _TP = None
    _DP = None
    _EP = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def reset_parallel_groups():
    """Reset all parallel group globals (for testing)."""
    global _TP, _DP, _EP, _WORLD
    _TP = None
    _DP = None
    _EP = None
    _WORLD = None