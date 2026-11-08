"""Safe numeric array codecs; no pickle and no execution of received objects."""
import json
import math
from functools import lru_cache

import numpy as np

SERVICE = "hytorch.distributed.ParameterService"
METHOD = f"/{SERVICE}/Exchange"


@lru_cache(maxsize=1)
def envelope_class():
    # Runtime descriptors exactly mirror parameter_service.proto. No protoc step.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    file = descriptor_pb2.FileDescriptorProto()
    file.name = "parameter_service.proto"
    file.package = "hytorch.distributed"
    file.syntax = "proto3"
    tensor = file.message_type.add(name="Tensor")
    for name, number, kind, label in (("name", 1, 9, 1), ("dtype", 2, 9, 1),
                                     ("shape", 3, 3, 3), ("data", 4, 12, 1)):
        tensor.field.add(name=name, number=number, type=kind, label=label)
    envelope = file.message_type.add(name="Envelope")
    envelope.field.add(name="metadata_json", number=1, type=9, label=1)
    envelope.field.add(name="tensors", number=2, type=11, label=3,
                       type_name=".hytorch.distributed.Tensor")
    service = file.service.add(name="ParameterService")
    service.method.add(name="Exchange", input_type=".hytorch.distributed.Envelope",
                       output_type=".hytorch.distributed.Envelope")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file)
    descriptor = pool.FindMessageTypeByName("hytorch.distributed.Envelope")
    return message_factory.GetMessageClass(descriptor)


def checked_layout(dtype, shape, limit):
    dtype = np.dtype(dtype)
    if dtype.kind != "f" or dtype.itemsize not in (4, 8):
        raise ValueError("Only float32/float64 tensors are supported")
    shape = tuple(shape)
    if len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Invalid tensor shape")
    size = math.prod(shape) * dtype.itemsize
    if size > limit:
        raise ValueError("Tensor exceeds message limit")
    return dtype, shape, size


class Codec:
    def __init__(self, transport, limit):
        self.transport = transport
        self.limit = limit

    def encode(self, metadata, tensors=None):
        tensors = tensors or {}
        if self.transport == "socket_json":
            packet = {"metadata": metadata, "tensors": [
                {"name": name, "dtype": array.dtype.str, "shape": list(array.shape),
                 "data": array.reshape(-1).tolist()} for name, array in sorted(tensors.items())]}
            payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
        else:
            packet = envelope_class()()
            packet.metadata_json = json.dumps(metadata, separators=(",", ":"), allow_nan=False)
            for name, array in sorted(tensors.items()):
                item = packet.tensors.add()
                item.name, item.dtype = name, array.dtype.str
                item.shape.extend(array.shape)
                item.data = np.ascontiguousarray(array).tobytes()
            payload = packet.SerializeToString()
        if len(payload) > self.limit:
            raise ValueError("Encoded message exceeds max_message_mb")
        return payload

    def decode(self, payload):
        if len(payload) > self.limit:
            raise ValueError("Message exceeds max_message_mb")
        if self.transport == "socket_json":
            packet = json.loads(payload)
            metadata, records = packet["metadata"], packet["tensors"]
        else:
            packet = envelope_class()()
            packet.ParseFromString(payload)
            metadata = json.loads(packet.metadata_json)
            records = [{"name": t.name, "dtype": t.dtype, "shape": list(t.shape), "data": t.data}
                       for t in packet.tensors]
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be an object")
        tensors, total = {}, 0
        for item in records:
            name = item["name"]
            if not isinstance(name, str) or not name or name in tensors:
                raise ValueError("Invalid or duplicate tensor name")
            dtype, shape, size = checked_layout(item["dtype"], item["shape"], self.limit)
            total += size
            if total > self.limit:
                raise ValueError("Decoded tensors exceed message limit")
            if self.transport == "socket_json":
                array = np.asarray(item["data"], dtype=dtype)
            else:
                if len(item["data"]) != size:
                    raise ValueError("Invalid tensor byte count")
                array = np.frombuffer(item["data"], dtype=dtype).copy()
            if array.nbytes != size or not np.isfinite(array).all():
                raise ValueError("Invalid tensor size or non-finite tensor")
            tensors[name] = array.reshape(shape)
        return metadata, tensors
