import multiprocessing as mp
import uuid
import zmq
import torch
import time
import json
import pickle
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, TokenizedInput
from nanovllm.utils.network import get_zmq_socket, NetworkConfig, send_with_backpressure


class DPClient:
    """
    The Client for a single DP instance
        - scheduler: schedule reqs to inference in DP instance until finished;
        - ModelRunner: execute model to decode the next token
        - 
    """
    def __init__(self,
                 config: Config,
                 net_config: NetworkConfig,
                 target_rank: int,
        ):
        torch.cuda.set_device(target_rank)
        self.rank = target_rank
        self.tp_size = config.tensor_parallel_size
        self.dp_size = config.data_parallel_size

        # set tp worker
        self.ps = []
        self.events = []
        ctx = mp.get_context('spawn')
        for i in range(target_rank + 1, target_rank + self.tp_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, target_rank, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, target_rank, target_rank, self.events)

        # init scheduler
        self.scheduler = Scheduler(config)
        # init tcp socket
        self.init_tcp_socket(net_config)
        self.handling = True

        self.event_loop()

    def init_tcp_socket(self, config: NetworkConfig):
        context = zmq.Context(io_threads=2)
        # Router socket to receive request
        self.recv_from_router = get_zmq_socket(
            context, zmq.DEALER, config.recv_endpoint, False, config.identity
        )

        # Results send socket
        self.send_to_control = get_zmq_socket(
            context, zmq.PUSH, config.send_endpoint, False
        )

        # report successful registration
        registration = {
            "command": "register",
            "rank": self.rank
        }
        self.send_to_control.send_json(registration)

    def req_dispatcher(self, frame):
        req = pickle.loads(frame)
        if isinstance(req, TokenizedInput):
            seq = Sequence(req.token_ids, req.seq_id, req.sampling_params)
            self.add_request(seq)
        elif isinstance(req, dict):
            if req.get("command") == "exit":
                self.handling = False
        else:
            raise ValueError("Unexpected Input Value.")

    def recv_request(self):
        while True:
            try:
                recv_req = self.recv_from_router.recv_multipart(zmq.NOBLOCK)
                self.req_dispatcher(recv_req[-1])
            except zmq.ZMQError:
                break

        if not self.handling:
            self.exit()

    def event_loop(self):
        #[TODO] Async recv/execute/send in future
        while True:
            # recv req
            self.recv_request()
            if not self.handling:
                break
            outputs = self.step()
            # return result once inference has finished
            if outputs:
                send_with_backpressure(self.send_to_control, outputs)

    def add_request(self, seq: Sequence):
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if seqs:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        else:
            #[TODO] add IDLE sync
            outputs = []
        return outputs

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        self.recv_from_router.close()
        self.send_to_control.close()
        for p in self.ps:
            p.join()


class DataParallelController:
    """
    The controller that dispatches requests to multiple data parallel workers.
    For a Light LLM Inference Framework, the ideal state is efficient, which is
    the core target to development. Therefore, we should set some principles:
    1. All DP Groups are in the same node;
    2. Controller is not only responsible for scheduling, but also execute;
    3. TP Rank is set in order, such as DP2TP4,
        | 0 1 2 3 | 4 5 6 7 |
        |-- DP1 --|-- DP2 --|
        Rank 0~3 is equal to tp 1~4

    4. ZMQ connections
        main_process
        |- Router socket: reqs router
        |- Pull socket: results collection
        |- PUB socket: life cycle
    """
    def __init__(self,
                 config: Config
        ):
        self.dp_size = config.data_parallel_size
        self.tp_size = config.tensor_parallel_size
        # 1. build communication socket
        # 1 thread for result receive / dp_size thread for request router
        context = zmq.Context(io_threads=2)
        # 1.1 router socket
        router_port, self.router_socket = get_zmq_socket(context, zmq.ROUTER)
        router_endpoint = f"tcp://127.0.0.1:{router_port}"

        # 1.2 collect socket
        result_port, self.result_socket = get_zmq_socket(context, zmq.PULL)
        result_endpoint = f"tcp://127.0.0.1:{result_port}"

        net_config = NetworkConfig(recv_endpoint=router_endpoint,
                                   send_endpoint=result_endpoint)
        # 2. make DP Client
        self.dp_target_ranks = range(0,
                                     self.dp_size * self.tp_size,
                                     self.tp_size)
        self.dp_ps: list[mp.Process] = []
        self.dp_ids: list[str] = []
        ctx = mp.get_context('spawn')
        for target_rank in self.dp_target_ranks:
            identity = f"Client_{target_rank}_{uuid.uuid4().hex[:6]}".encode()
            net_config.identity = identity
            process = ctx.Process(target=DPClient, args=(config,
                                                         net_config,
                                                         target_rank))
            process.start()
            self.dp_ps.append(process)
            self.dp_ids.append(identity)

        self.wait_for_client_registration()
        self.init_req_dispatcher(config.dplb_method)
        self.Res_Budget = {}

    def wait_for_client_registration(self, timeout: int = 90):
        registered_clients = 0
        total_clients = self.dp_size
        start_time = time.time()
        while registered_clients < total_clients and (time.time() - start_time) < timeout:
            try:
                flag = self.result_socket.recv_json(zmq.NOBLOCK)
                if flag.get('command') == 'register':
                    assert flag.get('rank') in self.dp_target_ranks, 'Unexpected device id mapping ERROR!'
                    registered_clients += 1
            except zmq.Again:
                pass
        
        if registered_clients < total_clients:
            raise TimeoutError(f'DPClent Registration Timeout, Only Find {registered_clients} Ranks.')
        print("[Logger] All DPClients have registered successfully.")

    def init_req_dispatcher(self, method: str):
        function_map = {"ROUND_ROBIN": self.round_robin_scheduler}
        assert method in function_map, f'Dispatch method {method} has not been supported yet.'
        self.dispatcher = function_map.get(method)
        if method == "ROUND_ROBIN":
            self.robin_counter = 0

    def round_robin_scheduler(self, seq: TokenizedInput):
        target_ids = self.dp_ids[self.robin_counter]
        seq_bytes = pickle.dumps(seq)
        self.router_socket.send_multipart([target_ids, seq_bytes])
        self.robin_counter = (self.robin_counter + 1) % self.dp_size

    def res_recv_from_client(self):
        num_tokens = 0
        while True:
            try:
                generate_res = self.result_socket.recv_pyobj(zmq.NOBLOCK)
                for seq_id, token_ids in generate_res:
                    self.Res_Budget[seq_id] = token_ids
                    num_tokens += len(token_ids)
            except zmq.ZMQError:
                break
        return num_tokens

    def call(self, method_name: str, *args):
        method = getattr(self, method_name, None)
        return method(*args)

    def exit(self):
        messg = pickle.dumps(
            {
                "command": "exit"
        })
        for i in range(self.dp_size):
            self.router_socket.send_multipart([self.dp_ids[i], messg])
        self.result_socket.close()
        for ps in self.dp_ps:
            ps.join()