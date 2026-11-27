#!/usr/bin/env python3
"""
ASI Offline Requantizer (E07 / v2.6).

Converts an existing ASI v2 cartridge (e.g. Q6_K) to:
  - 4-bit AVX2-Repacked (ASI_QUANT_Q4_REPACK = 100, 4.25 bpw) [primary]
  - 3-bit Split-Plane vpshufb LUT (ASI_QUANT_Q3_SPLITLUT = 101, 3.125 bpw) [opt-in]
with optional column-major layout on `blk.*.ffn_down.weight` (ASI_TENSOR_LAYOUT_COL_MAJOR)
and automatic pre-serialized ASI_SEC_SIEVE_SKETCH (0x11) sealing.

Preserves `token_embd.weight` / `output.weight` / norms / `ssm_a` / `ssm_conv1d` / `ssm_dt.bias`
byte-for-byte so Sieve LM-head and exact embeddings remain untouched.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ASI_DIR = Path(__file__).resolve().parent
if str(ASI_DIR) not in sys.path:
    sys.path.insert(0, str(ASI_DIR))

from asi_v2 import (  # noqa: E402
    FLAG_HAS_SIGNATURE,
    QUANT_Q3_SPLITLUT,
    QUANT_Q4_REPACK,
    SEC_METADATA,
    SEC_SIEVE_SKETCH,
    SEC_SIGNATURE,
    SEC_WEIGHTS,
    TENSOR_LAYOUT_COL_MAJOR,
    TENSOR_LAYOUT_ROW_MAJOR,
    pack_weights_section,
    read_asi_v2_sections,
    write_asi_v2,
)
from asi_sieve_seal import build_sieve_section_bytes  # noqa: E402

_BODY_PROJ_RE = re.compile(
    r"^blk\.\d+\.(attn_qkv|attn_gate|attn_q|attn_k|attn_v|attn_output|"
    r"ssm_alpha|ssm_beta|ssm_out|ffn_gate|ffn_up|ffn_down)\.weight$"
)


def is_body_projection(name: str, qtype: int, shape: Tuple[int, int]) -> bool:
    if shape[0] <= 1 or shape[1] <= 1 or qtype == 0:
        return False
    return bool(_BODY_PROJ_RE.match(name))


def _load_c_helpers(lib_path: Optional[Path] = None):
    so = Path(lib_path) if lib_path else (ASI_DIR / "liblila_asi.so")
    if not so.exists():
        return None
    try:
        lib = ctypes.CDLL(str(so))
        lib.lila_asi_dequant_f32.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.lila_asi_dequant_f32.restype = ctypes.c_int

        lib.lila_asi_requant_q4_repack.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.lila_asi_requant_q4_repack.restype = ctypes.c_int

        lib.lila_asi_requant_q3_splitlut.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.lila_asi_requant_q3_splitlut.restype = ctypes.c_int
        return lib
    except (OSError, AttributeError):
        return None


def _dequant_q6k_numpy(raw: bytes, cols: int, rows: int) -> np.ndarray:
    """Vectorized NumPy fallback Q6_K dequantizer for (rows, cols)."""
    n_blk = (cols + 255) // 256
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, n_blk, 210)
    ql = arr[:, :, :128]
    qh = arr[:, :, 128:192]
    sc = arr[:, :, 192:208].view(np.int8).astype(np.float32)
    d = arr[:, :, 208:210].view(np.float16).astype(np.float32)

    out = np.empty((rows, n_blk, 256), dtype=np.float32)
    for half in range(2):
        qlh = ql[:, :, half * 64 : (half + 1) * 64]
        qhh = qh[:, :, half * 32 : (half + 1) * 32]
        sch = sc[:, :, half * 8 : (half + 1) * 8]
        base = half * 128
        ql0 = qlh[:, :, :32]
        ql1 = qlh[:, :, 32:64]
        q0 = (ql0 & 0x0F) | ((qhh & 0x03) << 4)
        q1 = (ql1 & 0x0F) | (((qhh >> 2) & 0x03) << 4)
        q2 = (ql0 >> 4) | (((qhh >> 4) & 0x03) << 4)
        q3 = (ql1 >> 4) | (((qhh >> 6) & 0x03) << 4)
        s0 = np.concatenate([np.repeat(sch[:, :, 0:1], 16, axis=2), np.repeat(sch[:, :, 1:2], 16, axis=2)], axis=2)
        s2 = np.concatenate([np.repeat(sch[:, :, 2:3], 16, axis=2), np.repeat(sch[:, :, 3:4], 16, axis=2)], axis=2)
        s4 = np.concatenate([np.repeat(sch[:, :, 4:5], 16, axis=2), np.repeat(sch[:, :, 5:6], 16, axis=2)], axis=2)
        s6 = np.concatenate([np.repeat(sch[:, :, 6:7], 16, axis=2), np.repeat(sch[:, :, 7:8], 16, axis=2)], axis=2)
        out[:, :, base + 0 : base + 32] = d * s0 * (q0.astype(np.float32) - 32.0)
        out[:, :, base + 32 : base + 64] = d * s2 * (q1.astype(np.float32) - 32.0)
        out[:, :, base + 64 : base + 96] = d * s4 * (q2.astype(np.float32) - 32.0)
        out[:, :, base + 96 : base + 128] = d * s6 * (q3.astype(np.float32) - 32.0)
    return out.reshape(rows, n_blk * 256)[:, :cols].copy()


def _requant_q4_repack_numpy(w: np.ndarray, col_major: bool) -> bytes:
    """Vectorized NumPy fallback for Q4_REPACK (4.25 bpw)."""
    mat = w.T if col_major else w
    n_lines, line_len = mat.shape
    n_blk = (line_len + 63) // 64
    pad = n_blk * 64 - line_len
    if pad > 0:
        mat = np.pad(mat, ((0, 0), (0, pad)), mode="constant")
    grp = mat.reshape(n_lines, n_blk, 64)
    amax = np.max(np.abs(grp), axis=2, keepdims=True)
    sc = np.where(amax > 0, amax / 7.5, 1.0).astype(np.float32)
    sc16 = sc.astype(np.float16)
    sc_val = sc16.astype(np.float32)
    inv = np.where(sc_val > 0, 1.0 / sc_val, 0.0)
    q = np.clip(np.rint(grp * inv) + 8.0, 0, 15).astype(np.uint8)
    packed = (q[:, :, :32] | (q[:, :, 32:] << 4)).astype(np.uint8)
    d_bytes = sc16.reshape(n_lines, n_blk, 1).view(np.uint8).reshape(n_lines, n_blk, 2)
    blocks = np.concatenate([d_bytes, packed], axis=2)
    return np.ascontiguousarray(blocks).tobytes()


def requantize_cartridge(
    in_path: Path,
    out_path: Path,
    *,
    bits: int = 4,
    col_major_ffn_down: bool = True,
    search_steps: int = 12,
    sieve_mode: Optional[str] = "1bit1024",
    min_rows_q3: int = 64,
    lib_path: Optional[Path] = None,
) -> Dict[str, object]:
    if bits not in (3, 4):
        raise ValueError(f"Unsupported bits={bits}; expected 3 or 4")

    t0 = time.perf_counter()
    c_lib = _load_c_helpers(lib_path)

    header_info, raw_sections = read_asi_v2_sections(in_path)
    arch_id = int(header_info["arch_id"])
    flags = int(header_info["flags"]) & ~FLAG_HAS_SIGNATURE

    w_body = None
    other_sections: List[Tuple[int, int, bytes]] = []
    meta_dict: Dict[str, object] = {}

    for stype, sflags, _scrc, payload in raw_sections:
        if stype in (SEC_SIGNATURE, SEC_SIEVE_SKETCH):
            continue
        if stype == SEC_WEIGHTS:
            w_body = payload
        elif stype == SEC_METADATA:
            try:
                meta_dict = json.loads(payload.decode("utf-8"))
            except Exception:
                meta_dict = {}
        else:
            other_sections.append((stype, sflags, payload))

    if w_body is None:
        raise ValueError(f"Missing SEC_WEIGHTS in {in_path}")

    n_tensors, _wflags, data_offset = struct.unpack("<IIQ", w_body[:16])
    tensor_entries = []
    for i in range(n_tensors):
        off = 16 + i * 80
        name_raw, qtype, lflags, d0, d1, toff, tsz = struct.unpack(
            "<40sIIQQQQ", w_body[off : off + 80]
        )
        name = name_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        tensor_entries.append((name, qtype, lflags, (d0, d1), toff, tsz))

    w_blob = memoryview(w_body)[data_offset:]
    new_tensors: List[Tuple[str, int, int, Tuple[int, int], bytes]] = []
    orig_body_bytes = 0
    new_body_bytes = 0
    kept_bytes = 0
    cos_by_group: Dict[str, List[float]] = {}
    n_requant = 0

    for name, qtype, lflags, (cols, rows), toff, tsz in tensor_entries:
        raw_slice = bytes(w_blob[toff : toff + tsz])
        if not is_body_projection(name, qtype, (cols, rows)):
            new_tensors.append((name, qtype, lflags, (cols, rows), raw_slice))
            kept_bytes += len(raw_slice)
            continue

        orig_body_bytes += len(raw_slice)
        # Dequantize source tensor -> FP32 [rows, cols]
        w_f32 = np.empty((rows, cols), dtype=np.float32)
        if c_lib is not None:
            raw_arr = np.frombuffer(raw_slice, dtype=np.uint8)
            rc = c_lib.lila_asi_dequant_f32(
                raw_arr.ctypes.data,
                qtype,
                lflags,
                cols,
                rows,
                w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc != 0:
                raise RuntimeError(f"lila_asi_dequant_f32 failed for {name} (qt={qtype})")
        elif qtype == 14:
            w_f32 = _dequant_q6k_numpy(raw_slice, cols, rows)
        else:
            raise RuntimeError(f"NumPy fallback only supports Q6_K; build liblila_asi.so for qt={qtype}")

        use_q3 = (bits == 3) and (rows >= min_rows_q3)
        use_col_major = bool(col_major_ffn_down and not use_q3 and name.endswith(".ffn_down.weight"))

        if use_q3:
            target_qt = QUANT_Q3_SPLITLUT
            target_lf = TENSOR_LAYOUT_ROW_MAJOR
            n_tiles = (rows + 63) // 64
            n_blk128 = (cols + 127) // 128
            out_bytes_len = n_tiles * n_blk128 * 3200
            if c_lib is not None:
                out_buf = (ctypes.c_uint8 * out_bytes_len)()
                rc = c_lib.lila_asi_requant_q3_splitlut(
                    w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    rows,
                    cols,
                    search_steps,
                    out_buf,
                )
                if rc != 0:
                    raise RuntimeError(f"lila_asi_requant_q3_splitlut failed for {name}")
                new_raw = bytes(bytearray(out_buf))
            else:
                raise RuntimeError("3-bit requant requires liblila_asi.so")
        else:
            target_qt = QUANT_Q4_REPACK
            target_lf = TENSOR_LAYOUT_COL_MAJOR if use_col_major else TENSOR_LAYOUT_ROW_MAJOR
            if use_col_major:
                n_blk = (rows + 63) // 64
                out_bytes_len = cols * n_blk * 34
            else:
                n_blk = (cols + 63) // 64
                out_bytes_len = rows * n_blk * 34
            if c_lib is not None:
                out_buf = (ctypes.c_uint8 * out_bytes_len)()
                rc = c_lib.lila_asi_requant_q4_repack(
                    w_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    rows,
                    cols,
                    1 if use_col_major else 0,
                    search_steps,
                    out_buf,
                )
                if rc != 0:
                    raise RuntimeError(f"lila_asi_requant_q4_repack failed for {name}")
                new_raw = bytes(bytearray(out_buf))
            else:
                new_raw = _requant_q4_repack_numpy(w_f32, use_col_major)

        # Verify dequant round-trip cosine on first layer for immediate sanity
        if c_lib is not None and (name.startswith("blk.0.") or name.startswith("blk.3.")):
            rec_f32 = np.empty((rows, cols), dtype=np.float32)
            new_arr = np.frombuffer(new_raw, dtype=np.uint8)
            c_lib.lila_asi_dequant_f32(
                new_arr.ctypes.data,
                target_qt,
                target_lf,
                cols,
                rows,
                rec_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            a = w_f32.ravel().astype(np.float64)
            b = rec_f32.ravel().astype(np.float64)
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
            kind = name.split(".", 2)[2]
            cos_by_group.setdefault(kind, []).append(cos)

        new_tensors.append((name, target_qt, target_lf, (cols, rows), new_raw))
        new_body_bytes += len(new_raw)
        n_requant += 1

    new_weights_sec = pack_weights_section(new_tensors)

    meta_dict.update(
        {
            "requantized_from": in_path.name,
            "quant_bits": bits,
            "primary_body_quant": "Q3_SPLITLUT" if bits == 3 else "Q4_REPACK",
            "col_major_ffn_down": bool(col_major_ffn_down and bits == 4),
            "search_steps": search_steps,
            "n_requantized_tensors": n_requant,
        }
    )
    meta_bytes = json.dumps(meta_dict, separators=(",", ":")).encode("utf-8")

    sections: List[Tuple[int, int, bytes]] = []
    # Preserve order: MODEL_CONFIG, WEIGHTS, rest, METADATA
    for stype, sflags, payload in other_sections:
        sections.append((stype, sflags, payload))
        if stype == 0x01:
            sections.append((SEC_WEIGHTS, 0, new_weights_sec))
    if not any(st == SEC_WEIGHTS for st, _, _ in sections):
        sections.insert(0, (SEC_WEIGHTS, 0, new_weights_sec))
    sections.append((SEC_METADATA, 0, meta_bytes))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_unsealed = out_path.with_name(f".{out_path.name}.unsealed.{os.getpid()}.tmp")
    tmp_final = out_path.with_name(f".{out_path.name}.final.{os.getpid()}.tmp")
    try:
        write_asi_v2(tmp_unsealed, arch_id=arch_id, flags=flags, sections=sections)
        if sieve_mode:
            sieve_sec = build_sieve_section_bytes(tmp_unsealed, mode=sieve_mode, lib_path=lib_path)
            sections.append((SEC_SIEVE_SKETCH, 0, sieve_sec))
            write_asi_v2(tmp_final, arch_id=arch_id, flags=flags, sections=sections)
            os.replace(tmp_final, out_path)
        else:
            os.replace(tmp_unsealed, out_path)
    finally:
        for p in (tmp_unsealed, tmp_final):
            if p.exists():
                p.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    avg_cos = {k: round(sum(v) / len(v), 6) for k, v in sorted(cos_by_group.items())}
    file_size = out_path.stat().st_size
    sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    return {
        "out_path": str(out_path),
        "bits": bits,
        "col_major_ffn_down": bool(col_major_ffn_down and bits == 4),
        "search_steps": search_steps,
        "sieve_mode": sieve_mode,
        "n_requantized_tensors": n_requant,
        "orig_body_mib": round(orig_body_bytes / 1048576.0, 2),
        "new_body_mib": round(new_body_bytes / 1048576.0, 2),
        "kept_mib": round(kept_bytes / 1048576.0, 2),
        "file_bytes": file_size,
        "file_mib": round(file_size / 1048576.0, 2),
        "sha256": sha256,
        "sample_weight_cos": avg_cos,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Requantize an ASI v2 cartridge body to Q4_REPACK (4.25 bpw) or Q3_SPLITLUT (3.125 bpw)")
    ap.add_argument("input_asi", type=Path)
    ap.add_argument("output_asi", type=Path)
    ap.add_argument("--bits", type=int, default=4, choices=[3, 4], help="Target body quantization bits (default: 4)")
    ap.add_argument("--search-steps", type=int, default=12, help="Per-group MSE scale search steps (default: 12)")
    ap.add_argument("--col-major-ffn-down", dest="col_major_ffn_down", action="store_true", default=None)
    ap.add_argument("--no-col-major-ffn-down", dest="col_major_ffn_down", action="store_false")
    ap.add_argument("--sieve-mode", default="1bit1024", help="Sieve sketch mode to seal (or 'none')")
    ap.add_argument("--min-rows-q3", type=int, default=64, help="Min rows to use 64-row Q3_SPLITLUT tiles (smaller use Q4_REPACK)")
    ap.add_argument("--lib", type=Path, default=None, help="Optional path to liblila_asi.so")
    args = ap.parse_args()

    col_major = (args.bits == 4) if args.col_major_ffn_down is None else args.col_major_ffn_down
    sieve = None if str(args.sieve_mode).lower() in ("none", "off", "0", "") else args.sieve_mode
    report = requantize_cartridge(
        args.input_asi,
        args.output_asi,
        bits=args.bits,
        col_major_ffn_down=col_major,
        search_steps=args.search_steps,
        sieve_mode=sieve,
        min_rows_q3=args.min_rows_q3,
        lib_path=args.lib,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
