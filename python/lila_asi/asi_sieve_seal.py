#!/usr/bin/env python3
"""
ASI Pre-Serialized Sieve Sketch Sealer (`ASI_SEC_SIEVE_SKETCH`, 0x11).

Generates a bit-exact `ASI_SEC_SIEVE_SKETCH` payload via `lila_asi_export_sieve_section`
and atomically seals it into a `.asi` v2 cartridge without modifying `ASI_SEC_WEIGHTS`
or any other base section.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASI_DIR = Path(__file__).resolve().parent
if str(ASI_DIR) not in sys.path:
    sys.path.insert(0, str(ASI_DIR))

from asi_v2 import (  # noqa: E402
    ASI_V2_MAGIC,
    ASI_V2_VERSION,
    SEC_SIEVE_SKETCH,
    SEC_WEIGHTS,
    crc64,
    write_asi_v2,
)


def _load_lila_asi(lib_path: Optional[Path] = None) -> ctypes.CDLL:
    so = Path(lib_path) if lib_path else (ASI_DIR / "liblila_asi.so")
    if not so.exists():
        raise FileNotFoundError(f"liblila_asi.so not found at {so}")
    lib = ctypes.CDLL(str(so))
    lib.lila_asi_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.lila_asi_open.restype = ctypes.c_void_p
    lib.lila_asi_close.argtypes = [ctypes.c_void_p]
    lib.lila_asi_close.restype = None
    lib.lila_asi_export_sieve_section.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.lila_asi_export_sieve_section.restype = ctypes.c_int
    return lib


def build_sieve_section_bytes(
    asi_path: Path,
    mode: str = "1bit1024",
    lib_path: Optional[Path] = None,
) -> bytes:
    lib = _load_lila_asi(lib_path)
    err = ctypes.create_string_buffer(256)
    model = lib.lila_asi_open(str(asi_path).encode("utf-8"), err, 256)
    if not model:
        raise RuntimeError(f"lila_asi_open failed: {err.value.decode('utf-8', 'replace')}")
    try:
        needed = ctypes.c_size_t(0)
        mode_b = mode.encode("utf-8")
        if lib.lila_asi_export_sieve_section(model, mode_b, None, 0, ctypes.byref(needed)) != 0:
            raise RuntimeError(f"lila_asi_export_sieve_section size query failed for mode={mode}")
        buf = (ctypes.c_uint8 * needed.value)()
        written = ctypes.c_size_t(0)
        if lib.lila_asi_export_sieve_section(model, mode_b, buf, needed.value, ctypes.byref(written)) != 0:
            raise RuntimeError(f"lila_asi_export_sieve_section failed for mode={mode}")
        return bytes(bytearray(buf[: written.value]))
    finally:
        lib.lila_asi_close(model)


def _read_sections(path: Path) -> Tuple[int, int, List[Tuple[int, int, int, bytes]]]:
    data = Path(path).read_bytes()
    if len(data) < 64:
        raise ValueError("Truncated ASI v2 cartridge")
    magic, version, arch_id, flags, n_sections, _res, _total, table_off = struct.unpack(
        "<IIIIIIQQ", data[:40]
    )
    if magic != ASI_V2_MAGIC or version != ASI_V2_VERSION:
        raise ValueError("Not a valid ASI v2 cartridge")
    sections: List[Tuple[int, int, int, bytes]] = []
    for i in range(n_sections):
        off_i = table_off + i * 32
        stype, sflags, soff, ssz, scrc = struct.unpack("<IIQQQ", data[off_i : off_i + 32])
        sections.append((stype, sflags, scrc, data[soff : soff + ssz]))
    return arch_id, flags, sections


def seal_sieve_into_cartridge(
    base_asi_path: Path,
    out_asi_path: Path,
    *,
    mode: str = "1bit1024",
    lib_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Seal ASI_SEC_SIEVE_SKETCH into a new .asi cartridge preserving SEC_WEIGHTS bit-for-bit."""
    arch_id, flags, sections = _read_sections(base_asi_path)
    sieve_sec = build_sieve_section_bytes(base_asi_path, mode=mode, lib_path=lib_path)

    preserved: List[Tuple[int, int, bytes]] = []
    orig_weights_sha = ""
    orig_weights_crc = 0
    for stype, sflags, scrc, payload in sections:
        if stype == SEC_SIEVE_SKETCH:
            continue
        if stype == SEC_WEIGHTS:
            orig_weights_crc = scrc
            orig_weights_sha = hashlib.sha256(payload).hexdigest()
        preserved.append((stype, sflags, payload))
    preserved.append((SEC_SIEVE_SKETCH, 0, sieve_sec))

    out_path = Path(out_asi_path)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    write_asi_v2(tmp_path, arch_id=arch_id, flags=flags, sections=preserved)
    os.replace(tmp_path, out_path)

    _, _, new_sections = _read_sections(out_path)
    verified_weights = False
    for stype, _sflags, scrc, payload in new_sections:
        if stype == SEC_WEIGHTS:
            if scrc != orig_weights_crc or crc64(payload) != orig_weights_crc:
                raise RuntimeError("SEC_WEIGHTS CRC mismatch after sealing!")
            if hashlib.sha256(payload).hexdigest() != orig_weights_sha:
                raise RuntimeError("SEC_WEIGHTS SHA-256 mismatch after sealing!")
            verified_weights = True
    if not verified_weights:
        raise RuntimeError("SEC_WEIGHTS missing after sealing!")

    return {
        "out_path": str(out_path),
        "mode": mode,
        "sieve_bytes": len(sieve_sec),
        "weights_sha256": orig_weights_sha,
        "verified_immutable_weights": True,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Seal pre-serialized ASI_SEC_SIEVE_SKETCH into .asi cartridge")
    ap.add_argument("base_asi", type=Path)
    ap.add_argument("out_asi", type=Path)
    ap.add_argument("--mode", default="1bit1024", choices=["1bit512", "1bit1024", "2bit1024"])
    args = ap.parse_args()
    print(seal_sieve_into_cartridge(args.base_asi, args.out_asi, mode=args.mode))
