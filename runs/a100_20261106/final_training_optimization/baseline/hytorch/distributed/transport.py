"""Length-framed TCP/JSON and gRPC/protobuf with the same PS operations."""
import socket
import socketserver
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .protocol import Codec, METHOD, SERVICE


def receive_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        block = sock.recv(min(size - len(result), 1024 * 1024))
        if not block:
            raise ConnectionError("Peer disconnected during message reception")
        result.extend(block)
    return bytes(result)


def receive_frame(sock, limit):
    size, = struct.unpack("!Q", receive_exact(sock, 8))
    if size > limit:
        raise ValueError("Incoming frame exceeds max_message_mb")
    return receive_exact(sock, size)


def send_frame(sock, payload):
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


class Client:
    def __init__(self, config, address, token, rank):
        self.config, self.token, self.rank = config, token, rank
        self.codec = Codec(config.transport, config.max_message_mb * 1024**2)
        self.channel = None
        self.sock = None
        if config.transport == "socket_json":
            self.sock = socket.create_connection(address, timeout=config.rpc_timeout_s)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            import grpc
            self.channel = grpc.insecure_channel(f"{address[0]}:{address[1]}", options=[
                ("grpc.max_send_message_length", self.codec.limit),
                ("grpc.max_receive_message_length", self.codec.limit)])
            grpc.channel_ready_future(self.channel).result(timeout=config.rpc_timeout_s)
            self.exchange = self.channel.unary_unary(METHOD)

    def call(self, operation, tensors=None, **values):
        started = time.perf_counter()
        metadata = dict(values, op=operation, token=self.token, rank=self.rank)
        payload = self.codec.encode(metadata, tensors)
        encoded = time.perf_counter()
        if self.sock is not None:
            send_frame(self.sock, payload)
            response = receive_frame(self.sock, self.codec.limit)
        else:
            response = self.exchange(payload, timeout=self.config.rpc_timeout_s)
        received = time.perf_counter()
        result, arrays = self.codec.decode(response)
        finished = time.perf_counter()
        timing = {"rpc_s": finished - started, "encode_s": encoded - started,
                  "transport_wait_s": received - encoded, "decode_s": finished - received,
                  "sent_bytes": len(payload), "received_bytes": len(response)}
        if not result.get("ok"):
            raise RuntimeError(f"PS {operation} failed: {result.get('error', 'unknown error')}")
        return result, arrays, timing

    def close(self):
        if self.sock is not None:
            self.sock.close()
        if self.channel is not None:
            self.channel.close()


def start_transport(config, dispatch, on_disconnect):
    codec = Codec(config.transport, config.max_message_mb * 1024**2)

    def exchange(payload):
        try:
            metadata, arrays = codec.decode(payload)
            response, result_arrays = dispatch(metadata, arrays)
            return codec.encode(response, result_arrays)
        except Exception as exc:
            on_disconnect(str(exc))
            response, result_arrays = {"ok": False, "error": str(exc)}, {}
        return codec.encode(response, result_arrays)

    if config.transport == "socket_json":
        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                self.request.settimeout(config.rpc_timeout_s)
                self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    while True:
                        payload = receive_frame(self.request, codec.limit)
                        send_frame(self.request, exchange(payload))
                except (ConnectionError, OSError, ValueError) as exc:
                    on_disconnect(str(exc))

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = False

        server = Server((config.host, config.port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        return server.server_address, stop

    import grpc
    server = grpc.server(ThreadPoolExecutor(max_workers=config.num_workers + 4), options=[
        ("grpc.max_send_message_length", codec.limit),
        ("grpc.max_receive_message_length", codec.limit)])

    def grpc_exchange(payload, context):
        try:
            return exchange(payload)
        except Exception as exc:
            on_disconnect(str(exc))
            context.abort(grpc.StatusCode.INTERNAL, str(exc))

    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, {
        "Exchange": grpc.unary_unary_rpc_method_handler(grpc_exchange)}),))
    port = server.add_insecure_port(f"{config.host}:{config.port}")
    if not port:
        raise OSError("Could not bind PS gRPC port")
    server.start()
    return (config.host, port), lambda: server.stop(3).wait(timeout=5)
