from dataclasses import dataclass
import zmq
import time
import socket
import psutil
from typing import Optional, Tuple, Union

@dataclass
class NetworkConfig:
    """
    Set Net Connection addr
     - recv_endpoint for results collection;
     - send_endpoint for requests dispatch;
    """
    recv_endpoint: str
    send_endpoint: str
    identity: str | None = None


def config_socket(socket, socket_type: zmq.SocketType):
    mem = psutil.virtual_memory()
    total_mem = mem.total / 1024**3
    available_mem = mem.available / 1024**3
    if total_mem > 32 and available_mem > 16:
        buf_size = int(0.5 * 1024**3)
    else:
        buf_size = -1
    
    def set_send_opt():
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, buf_size)
    
    def set_recv_opt():
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF,buf_size)
    
    if socket_type in [zmq.PUSH, zmq.PUB]:
        set_send_opt()
    elif socket_type in [zmq.PULL, zmq.SUB]:
        set_recv_opt()
    elif socket_type in [zmq.DEALER, zmq.ROUTER]:
        set_send_opt()
        set_recv_opt()
    else:
        raise ValueError(f"Unsupported socket type: {socket_type}")
    

def get_free_port() -> int:
    """Get available port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]
    

def get_zmq_socket(
    context: zmq.Context,
    socket_type: zmq.SocketType,
    endpoint: Optional[str] = None,
    bind: bool = True,
    identity: Optional[bytes] = None
) -> Union[zmq.Socket, Tuple[int, zmq.Socket]]:
    """
    Create and configure a ZeroMQ socket.
    """
    socket = context.socket(socket_type)
    # set identity for ROUTER OR DEALER
    if identity is not None:
        socket.setsockopt(zmq.IDENTITY, identity)
    
    if endpoint is None:
        # bind to random TCP Port
        config_socket(socket, socket_type)
        try:
            port = socket.bind_to_random_port("tcp://*")
            return port, socket
        except zmq.ZMQError:
            port = get_free_port()
            socket.bind(f"tcp://*:{port}")
            return port, socket
    else:
        config_socket(socket, socket_type)

        if bind:
            socket.bind(endpoint)
        else:
            socket.connect(endpoint)

        return socket


def send_with_backpressure(socket, data, retry_time: int = 0.1):
    """Conservative sending: back off and retry rather than let a backlog fail the send."""
    while True:
        try: 
            socket.send_pyobj(data, zmq.NOBLOCK)
            break
        except zmq.Again:
            print("Backpressure: slowing down...")
            time.sleep(retry_time)
    