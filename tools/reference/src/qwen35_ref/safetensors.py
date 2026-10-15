"""Minimal safetensors reader and writer on top of NumPy (no torch).

Reading memory-maps the file; BF16 tensors are widened to float32 on access
(bf16 is the top half of an f32, so `uint16 << 16` reinterpreted is exact).
"""

import json
import struct
from pathlib import Path

import numpy as np

_NP_DTYPES = {"F32": np.float32, "F16": np.float16, "I32": np.int32, "I64": np.int64}


class SafetensorsFile:
    def __init__(self, path: Path):
        self._buf = np.memmap(path, dtype=np.uint8, mode="r")
        (header_len,) = struct.unpack("<Q", self._buf[:8].tobytes())
        header = json.loads(self._buf[8 : 8 + header_len].tobytes())
        self.metadata: dict[str, str] = header.pop("__metadata__", {})
        self._entries = header
        self._data_start = 8 + header_len

    def keys(self) -> list[str]:
        return list(self._entries)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._entries[name]["shape"])

    def get(self, name: str) -> np.ndarray:
        """Returns the tensor as float32 (BF16) or its native dtype, copied out of the map."""
        e = self._entries[name]
        begin, end = e["data_offsets"]
        raw = self._buf[self._data_start + begin : self._data_start + end]
        shape = tuple(e["shape"])
        if e["dtype"] == "BF16":
            return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return np.array(raw.view(_NP_DTYPES[e["dtype"]]).reshape(shape))


class Checkpoint:
    """All shards of a Hugging Face checkpoint directory."""

    def __init__(self, directory: Path):
        index = directory / "model.safetensors.index.json"
        if index.exists():
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            shards = ["model.safetensors"]
        self._by_name: dict[str, SafetensorsFile] = {}
        for shard in shards:
            f = SafetensorsFile(directory / shard)
            for k in f.keys():
                self._by_name[k] = f

    def get(self, name: str) -> np.ndarray:
        return self._by_name[name].get(name)


def save(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """Writes float32 / int32 tensors; the header is padded to 8 bytes as the format requires."""
    header: dict = {"__metadata__": metadata}
    offset = 0
    blobs = []
    for name, t in tensors.items():
        t = np.ascontiguousarray(t)
        dtype = {np.dtype(np.float32): "F32", np.dtype(np.int32): "I32"}[t.dtype]
        blob = t.astype(t.dtype.newbyteorder("<"), copy=False).tobytes()
        header[name] = {"dtype": dtype, "shape": list(t.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)
