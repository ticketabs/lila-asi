#!/usr/bin/env python3
"""
ASI Immutable-Base Single-Cartridge Pairing Sealer (`ASI_SEC_PAIRING`, 0x10).

Seals a validated/approved neural pairing bridge section into a portable
`.asiv` or `.asi` cartridge while preserving ALL base weight/vision sections
(`ASI_SEC_WEIGHTS`, `ASI_SEC_VISION`, etc.) bit-for-bit untouched.
"""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ASI_DIR = Path(__file__).resolve().parent
if str(ASI_DIR) not in sys.path:
    sys.path.insert(0, str(ASI_DIR))

from asi_pairing_registry import PairingState  # noqa: E402
from asi_v2 import (  # noqa: E402
    ASI_V2_MAGIC,
    ASI_V2_VERSION,
    FLAG_HAS_ADAPTERS,
    SEC_PAIRING,
    SEC_VISION,
    SEC_WEIGHTS,
    TENSOR_LAYOUT_ROW_MAJOR,
    crc64,
    write_asi_v2,
)

ASI_PAIRING_MAGIC = b"ASIP0001"
ASI_PAIRING_FORMAT_VERSION = 1

BRIDGE_LINEAR = 1
BRIDGE_MLP_2LAYER = 2


def pack_pairing_section(
    weights: Dict[str, np.ndarray],
    *,
    state: int,
    bridge_kind: int,
    act_type: int,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    target_llm_arch_id: int,
    source_vision_sha256: bytes,
    target_llm_sha256: bytes = b"\x00" * 32,
    target_mrope_sections: Tuple[int, int, int, int] = (0, 0, 0, 0),
    val_passed: bool = False,
    val_cosine: float = 0.0,
    val_loss: float = 0.0,
    train_steps: int = 0,
    flags: int = 0,
) -> bytes:
    """Pack binary ASI_SEC_PAIRING (0x10) matching engine/asi/asi_pairing.h byte-for-byte."""
    if len(source_vision_sha256) != 32 or len(target_llm_sha256) != 32:
        raise ValueError("SHA-256 fingerprints must be 32 bytes")

    if bridge_kind == BRIDGE_LINEAR:
        order = ("bridge.0.weight", "bridge.0.bias")
    elif bridge_kind == BRIDGE_MLP_2LAYER:
        order = ("bridge.0.weight", "bridge.0.bias", "bridge.1.weight", "bridge.1.bias")
    else:
        raise ValueError(f"Unsupported bridge_kind={bridge_kind}")

    n_tensors = len(order)
    dir_bytes = 192 + n_tensors * 80
    weights_offset = (dir_bytes + 63) & ~63

    tensor_entries: List[bytes] = []
    blobs: List[bytes] = []
    cur_off = 0

    for name in order:
        if name not in weights:
            raise KeyError(f"Missing bridge tensor {name!r}")
        arr = np.ascontiguousarray(weights[name], dtype=np.float32)
        if arr.ndim == 1:
            d0, d1 = 1, int(arr.shape[0])
        elif arr.ndim == 2:
            # GGUF/ASI row-major convention: shape[0]=K (in_cols), shape[1]=N (out_rows)
            d0, d1 = int(arr.shape[1]), int(arr.shape[0])
        else:
            raise ValueError(f"Bridge tensor {name} must be 1D or 2D")
        raw = arr.tobytes()
        name_b = name.encode("utf-8")[:39]
        name_pad = name_b + b"\x00" * (40 - len(name_b))
        tensor_entries.append(
            struct.pack("<40sIIQQQQ", name_pad, 0, TENSOR_LAYOUT_ROW_MAJOR, d0, d1, cur_off, len(raw))
        )
        aligned_len = (len(raw) + 63) & ~63
        blobs.append(raw + b"\x00" * (aligned_len - len(raw)))
        cur_off += aligned_len

    hdr = struct.pack(
        "<8s14I2f2IQ32s32s10I",
        ASI_PAIRING_MAGIC,
        ASI_PAIRING_FORMAT_VERSION,
        int(state),
        int(bridge_kind),
        int(act_type),
        int(in_dim),
        int(hidden_dim),
        int(out_dim),
        int(target_llm_arch_id),
        int(target_mrope_sections[0]),
        int(target_mrope_sections[1]),
        int(target_mrope_sections[2]),
        int(target_mrope_sections[3]),
        1 if val_passed else 0,
        int(flags),
        float(val_cosine),
        float(val_loss),
        int(train_steps),
        n_tensors,
        weights_offset,
        source_vision_sha256,
        target_llm_sha256,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    assert len(hdr) == 192
    pad = b"\x00" * (weights_offset - dir_bytes)
    return hdr + b"".join(tensor_entries) + pad + b"".join(blobs)


def unpack_pairing_header(pairing_sec: bytes) -> dict:
    if len(pairing_sec) < 192:
        raise ValueError("Truncated pairing section")
    vals = struct.unpack("<8s14I2f2IQ32s32s10I", pairing_sec[:192])
    return {
        "magic": vals[0],
        "format_version": vals[1],
        "state": vals[2],
        "bridge_kind": vals[3],
        "act_type": vals[4],
        "in_dim": vals[5],
        "hidden_dim": vals[6],
        "out_dim": vals[7],
        "target_llm_arch_id": vals[8],
        "val_passed": bool(vals[13]),
        "flags": vals[14],
        "val_cosine": vals[15],
        "val_loss": vals[16],
        "train_steps": vals[17],
        "n_tensors": vals[18],
        "weights_offset": vals[19],
        "source_vision_sha256": vals[20].hex(),
        "target_llm_sha256": vals[21].hex(),
    }


def inspect_cartridge_sections(path: Path) -> dict:
    data = Path(path).read_bytes()
    if len(data) < 64:
        raise ValueError("Truncated ASI v2 cartridge")
    magic, version, arch_id, flags, n_sections, _res, total_size, table_off = struct.unpack(
        "<IIIIIIQQ", data[:40]
    )
    if magic != ASI_V2_MAGIC or version != ASI_V2_VERSION:
        raise ValueError("Not a valid ASI v2 cartridge")
    sections: List[Tuple[int, int, int, int, int, bytes]] = []
    for i in range(n_sections):
        off_i = table_off + i * 32
        stype, sflags, soff, ssz, scrc = struct.unpack("<IIQQQ", data[off_i : off_i + 32])
        sections.append((stype, sflags, soff, ssz, scrc, data[soff : soff + ssz]))
    return {
        "arch_id": arch_id,
        "flags": flags,
        "total_size": total_size,
        "sections": sections,
    }


def seal_pairing_into_cartridge(
    base_path: Path,
    pairing_section: bytes,
    out_path: Path,
    *,
    source_meta_path: Optional[Path] = None,
) -> dict:
    """
    Atomically seal an approved ASI_SEC_PAIRING (0x10) into a portable cartridge
    while guaranteeing bit-for-bit preservation of all base weight/vision sections.
    """
    del source_meta_path
    ph = unpack_pairing_header(pairing_section)
    if ph["magic"] != ASI_PAIRING_MAGIC or ph["format_version"] != ASI_PAIRING_FORMAT_VERSION:
        raise ValueError("Invalid pairing section magic/version")
    if ph["state"] not in (int(PairingState.APPROVED), int(PairingState.CONTINUOUS_IMPROVEMENT)) or not ph["val_passed"]:
        raise ValueError("Refusing to seal unapproved or unvalidated pairing bridge")

    base_info = inspect_cartridge_sections(base_path)
    preserved: List[Tuple[int, int, bytes]] = []
    base_immutable_records: Dict[int, Tuple[int, bytes]] = {}
    for stype, sflags, _soff, _ssz, scrc, raw in base_info["sections"]:
        if stype == SEC_PAIRING:
            continue
        preserved.append((stype, sflags, raw))
        if stype in (SEC_VISION, SEC_WEIGHTS):
            base_immutable_records[stype] = (scrc, raw)

    preserved.append((SEC_PAIRING, 0, pairing_section))
    tmp_out = Path(out_path).with_suffix(Path(out_path).suffix + ".tmp")
    write_asi_v2(
        tmp_out,
        arch_id=base_info["arch_id"],
        flags=base_info["flags"] | FLAG_HAS_ADAPTERS,
        sections=preserved,
    )
    os.replace(tmp_out, out_path)

    # Verify bit-for-bit immutability of base weight / vision sections
    sealed_info = inspect_cartridge_sections(out_path)
    verified_immutable = 0
    for stype, _sflags, _soff, _ssz, scrc, raw in sealed_info["sections"]:
        if stype in base_immutable_records:
            orig_crc, orig_raw = base_immutable_records[stype]
            if scrc != orig_crc or raw != orig_raw or crc64(raw) != orig_crc:
                raise RuntimeError(f"Base section {hex(stype)} mutated during sealing!")
            verified_immutable += 1

    if verified_immutable != len(base_immutable_records):
        raise RuntimeError("Missing immutable base section after sealing!")

    return {
        "out_path": str(out_path),
        "verified_immutable_sections": verified_immutable,
        "pairing_state": ph["state"],
        "val_cosine": ph["val_cosine"],
    }
