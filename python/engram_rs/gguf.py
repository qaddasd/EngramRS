"""Export PEFT / Engram safetensors LoRA adapters into GGUF LoRA format.

Compatible with llama.cpp (e.g. `llama-cli -m model.gguf --lora adapter.gguf`)
and Ollama (via Modelfile `ADAPTER adapter.gguf`).

Includes a robust, zero-dependency pure-Python GGUF v3 writer and reader.
"""

from __future__ import annotations

import enum
import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np


class GGUFValueType(enum.IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(enum.IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    BF16 = 16


def _pack_str(val: str) -> bytes:
    encoded = val.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class GGUFWriter:
    """Pure-Python writer for GGUF v3 files with 32-byte alignment."""

    def __init__(self, alignment: int = 32) -> None:
        self.alignment = alignment
        self.metadata: list[tuple[str, GGUFValueType, bytes]] = []
        self.tensors: list[dict[str, Any]] = []

    def add_uint8(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.UINT8, struct.pack("<B", val)))

    def add_int8(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.INT8, struct.pack("<b", val)))

    def add_uint16(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.UINT16, struct.pack("<H", val)))

    def add_int16(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.INT16, struct.pack("<h", val)))

    def add_uint32(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.UINT32, struct.pack("<I", val)))

    def add_int32(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.INT32, struct.pack("<i", val)))

    def add_uint64(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.UINT64, struct.pack("<Q", val)))

    def add_int64(self, key: str, val: int) -> None:
        self.metadata.append((key, GGUFValueType.INT64, struct.pack("<q", val)))

    def add_float32(self, key: str, val: float) -> None:
        self.metadata.append((key, GGUFValueType.FLOAT32, struct.pack("<f", val)))

    def add_float64(self, key: str, val: float) -> None:
        self.metadata.append((key, GGUFValueType.FLOAT64, struct.pack("<d", val)))

    def add_bool(self, key: str, val: bool) -> None:
        self.metadata.append((key, GGUFValueType.BOOL, struct.pack("<B", 1 if val else 0)))

    def add_string(self, key: str, val: str) -> None:
        self.metadata.append((key, GGUFValueType.STRING, _pack_str(val)))

    def add_array(self, key: str, elem_type: GGUFValueType, items: list[Any]) -> None:
        header = struct.pack("<IQ", elem_type.value, len(items))
        payload = bytearray(header)
        for item in items:
            if elem_type == GGUFValueType.STRING:
                payload.extend(_pack_str(str(item)))
            elif elem_type == GGUFValueType.UINT32:
                payload.extend(struct.pack("<I", int(item)))
            elif elem_type == GGUFValueType.INT32:
                payload.extend(struct.pack("<i", int(item)))
            elif elem_type == GGUFValueType.FLOAT32:
                payload.extend(struct.pack("<f", float(item)))
            elif elem_type == GGUFValueType.BOOL:
                payload.extend(struct.pack("<B", 1 if item else 0))
            else:
                raise NotImplementedError(f"Array of type {elem_type} not yet implemented")
        self.metadata.append((key, GGUFValueType.ARRAY, bytes(payload)))

    def add_tensor(
        self,
        name: str,
        tensor: np.ndarray,
        raw_dtype: GGMLType | None = None,
    ) -> None:
        """Add tensor to GGUF.

        GGML expects dimension order reversed compared to PyTorch/NumPy (i.e. innermost first).
        """
        if not tensor.flags.c_contiguous:
            tensor = np.ascontiguousarray(tensor)

        if raw_dtype is None or raw_dtype == GGMLType.F32:
            data_bytes = tensor.astype(np.float32).tobytes()
            ggml_dtype = GGMLType.F32
        elif raw_dtype == GGMLType.F16:
            data_bytes = tensor.astype(np.float16).tobytes()
            ggml_dtype = GGMLType.F16
        else:
            raise ValueError(f"Unsupported GGML type: {raw_dtype}")

        # In GGML, dimensions are innermost-first (ne[0] = columns, ne[1] = rows)
        ne_dims = list(reversed(tensor.shape))

        self.tensors.append({
            "name": name,
            "shape": ne_dims,
            "orig_shape": tensor.shape,
            "dtype": ggml_dtype,
            "data": data_bytes,
        })

    def write_to_file(self, filepath: str | Path) -> None:
        """Serialize complete GGUF v3 file."""
        target_path = Path(filepath).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Prepare Header
        # magic: "GGUF", version: 3, tensor_count: u64, metadata_kv_count: u64
        magic = b"GGUF"
        version = 3
        tensor_count = len(self.tensors)
        metadata_kv_count = len(self.metadata)

        header_bytes = bytearray()
        header_bytes.extend(magic)
        header_bytes.extend(struct.pack("<IQQ", version, tensor_count, metadata_kv_count))

        # 2. Prepare Metadata section
        for key, vtype, payload in self.metadata:
            header_bytes.extend(_pack_str(key))
            header_bytes.extend(struct.pack("<I", vtype.value))
            header_bytes.extend(payload)

        # 3. Compute Tensor Info offsets
        # Each tensor has:
        #   name: string
        #   n_dimensions: uint32
        #   dimensions: n_dimensions * uint64
        #   type: uint32
        #   offset: uint64 (relative to start of tensor data section)
        tensor_info_bytes = bytearray()
        current_data_offset = 0
        tensor_offsets: list[int] = []

        for t in self.tensors:
            # Align tensor offset to self.alignment
            rem = current_data_offset % self.alignment
            if rem != 0:
                current_data_offset += self.alignment - rem
            tensor_offsets.append(current_data_offset)
            current_data_offset += len(t["data"])

        for idx, t in enumerate(self.tensors):
            tensor_info_bytes.extend(_pack_str(t["name"]))
            tensor_info_bytes.extend(struct.pack("<I", len(t["shape"])))
            for dim in t["shape"]:
                tensor_info_bytes.extend(struct.pack("<Q", dim))
            tensor_info_bytes.extend(struct.pack("<I", t["dtype"].value))
            tensor_info_bytes.extend(struct.pack("<Q", tensor_offsets[idx]))

        # 4. Alignment padding before tensor data section
        pre_data_len = len(header_bytes) + len(tensor_info_bytes)
        pad_len = (self.alignment - (pre_data_len % self.alignment)) % self.alignment
        data_padding = b"\x00" * pad_len

        # 5. Write to disk
        with target_path.open("wb") as f:
            f.write(header_bytes)
            f.write(tensor_info_bytes)
            if pad_len > 0:
                f.write(data_padding)

            # Write tensor binary blobs with inter-tensor padding
            curr_pos = 0
            for idx, t in enumerate(self.tensors):
                target_offset = tensor_offsets[idx]
                gap = target_offset - curr_pos
                if gap > 0:
                    f.write(b"\x00" * gap)
                    curr_pos += gap
                f.write(t["data"])
                curr_pos += len(t["data"])


class GGUFReader:
    """Pure-Python reader to inspect and verify GGUF files."""

    def __init__(self, filepath: str | Path) -> None:
        self.path = Path(filepath).expanduser().resolve()
        with self.path.open("rb") as f:
            self._parse(f)

    def _read_str(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _parse(self, f) -> None:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Invalid GGUF magic bytes: {magic!r}")
        self.version, self.tensor_count, self.metadata_kv_count = struct.unpack("<IQQ", f.read(20))

        self.metadata: dict[str, Any] = {}
        for _ in range(self.metadata_kv_count):
            key = self._read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = self._read_val(f, GGUFValueType(vtype))
            self.metadata[key] = val

        self.tensors: dict[str, dict[str, Any]] = {}
        for _ in range(self.tensor_count):
            name = self._read_str(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = GGMLType(struct.unpack("<I", f.read(4))[0])
            offset = struct.unpack("<Q", f.read(8))[0]
            self.tensors[name] = {
                "name": name,
                "shape": dims,
                "dtype": dtype,
                "offset": offset,
            }

        # Calculate data section start
        curr = f.tell()
        alignment = self.metadata.get("general.alignment", 32)
        rem = curr % alignment
        pad = (alignment - rem) % alignment if rem != 0 else 0
        self.data_section_offset = curr + pad

    def _read_val(self, f, vtype: GGUFValueType) -> Any:
        if vtype == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif vtype == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif vtype == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUFValueType.BOOL:
            return struct.unpack("<B", f.read(1))[0] != 0
        elif vtype == GGUFValueType.STRING:
            return self._read_str(f)
        elif vtype == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif vtype == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif vtype == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif vtype == GGUFValueType.ARRAY:
            elem_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
            count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_val(f, elem_type) for _ in range(count)]
        raise ValueError(f"Unknown value type: {vtype}")

    def get_tensor_data(self, name: str) -> np.ndarray:
        """Extract tensor payload as NumPy array."""
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found")
        info = self.tensors[name]
        with self.path.open("rb") as f:
            f.seek(self.data_section_offset + info["offset"])
            # ne shape is innermost first; NumPy shape is reversed
            np_shape = tuple(reversed(info["shape"]))
            count = int(np.prod(info["shape"]))
            if info["dtype"] == GGMLType.F32:
                raw = f.read(count * 4)
                arr = np.frombuffer(raw, dtype="<f4").reshape(np_shape)
            elif info["dtype"] == GGMLType.F16:
                raw = f.read(count * 2)
                arr = np.frombuffer(raw, dtype="<f2").reshape(np_shape)
            else:
                raise NotImplementedError(f"Loading {info['dtype']} not implemented")
            return arr


def peft_to_gguf_tensor_name(peft_name: str, standard: bool = True) -> str:
    """Translate HuggingFace PEFT LoRA tensor names to llama.cpp GGUF LoRA naming.

    When standard=True:
        Maps e.g.:
        `base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight`
        to:
        `blk.0.attn_output.weight.lora_a`

    When standard=False:
        Preserves base_model path with lowercase .lora_a / .lora_b suffix.
    """
    is_lora_a = ".lora_A.weight" in peft_name or ".lora_A" in peft_name or peft_name.endswith(".lora_A")
    is_lora_b = ".lora_B.weight" in peft_name or ".lora_B" in peft_name or peft_name.endswith(".lora_B")
    suffix = ".lora_a" if is_lora_a else (".lora_b" if is_lora_b else "")

    if not standard:
        cleaned = peft_name.replace(".lora_A.weight", suffix).replace(".lora_B.weight", suffix)
        return cleaned

    # Extract layer number
    layer_match = re.search(r"layers\.(\d+)\.", peft_name)
    layer_idx = layer_match.group(1) if layer_match else "0"

    module_map = {
        "self_attn.o_proj": "attn_output.weight",
        "self_attn.q_proj": "attn_q.weight",
        "self_attn.k_proj": "attn_k.weight",
        "self_attn.v_proj": "attn_v.weight",
        "mlp.down_proj": "ffn_down.weight",
        "mlp.gate_proj": "ffn_gate.weight",
        "mlp.up_proj": "ffn_up.weight",
    }

    for hf_mod, gguf_mod in module_map.items():
        if hf_mod in peft_name:
            return f"blk.{layer_idx}.{gguf_mod}{suffix}"

    # Generic fallback
    clean = peft_name.replace("base_model.model.", "").replace("base_model.", "")
    clean = clean.replace(".lora_A.weight", suffix).replace(".lora_B.weight", suffix)
    return clean


def export_lora_to_gguf(
    adapter_dir: str | Path,
    output_file: str | Path,
    out_type: str = "f32",
    standard_names: bool = True,
    architecture: str = "llama",
) -> str:
    """Convert a PEFT safetensors LoRA adapter to GGUF LoRA format.

    Args:
        adapter_dir: Path to directory containing adapter_config.json and adapter_model.safetensors.
        output_file: Output path for the .gguf file.
        out_type: GGML weight precision ("f32" or "f16").
        standard_names: Whether to map HuggingFace names to standard llama.cpp blk.{i}... names.
        architecture: Target model architecture string (default "llama").

    Returns:
        Absolute path to the created .gguf file.
    """
    src_dir = Path(adapter_dir).expanduser().resolve()
    config_file = src_dir / "adapter_config.json"
    weights_file = src_dir / "adapter_model.safetensors"

    if not config_file.is_file():
        raise FileNotFoundError(f"Missing {config_file}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"Missing {weights_file}")

    with config_file.open("r", encoding="utf-8") as f:
        config = json.load(f)

    from safetensors.torch import load_file

    tensors = load_file(str(weights_file))

    writer = GGUFWriter(alignment=32)

    # Standard GGUF LoRA metadata required by llama.cpp
    base_model_name = str(config.get("base_model_name_or_path", "unknown"))
    lora_alpha = float(config.get("lora_alpha", config.get("r", 16)))

    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_string("general.architecture", architecture)
    writer.add_string("general.name", src_dir.name)
    writer.add_uint32("general.alignment", 32)
    writer.add_float32("adapter.lora.alpha", lora_alpha)
    writer.add_uint32("general.base_model.count", 1)
    writer.add_string("general.base_model.0.name", base_model_name)

    ggml_type = GGMLType.F16 if out_type.lower() == "f16" else GGMLType.F32

    for name, tensor in tensors.items():
        gguf_name = peft_to_gguf_tensor_name(name, standard=standard_names)
        arr = tensor.cpu().float().numpy()
        writer.add_tensor(gguf_name, arr, raw_dtype=ggml_type)

    dest = Path(output_file).expanduser().resolve()
    writer.write_to_file(dest)
    return str(dest)
