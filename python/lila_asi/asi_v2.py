#!/usr/bin/env python3
"""
ASI v2 Python Cartridge Writer & Validator.
Matches source/engine/asi/asi_v2.h byte-for-byte.
"""

import ctypes
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ASI_V2_MAGIC = 0x41534921
ASI_V2_VERSION = 2
ASI_V2_PAGE_SIZE = 4096

FLAG_HAS_VISION = 1 << 0
FLAG_HAS_AUDIO = 1 << 1
FLAG_HAS_ADAPTERS = 1 << 2
FLAG_HAS_SIGNATURE = 1 << 3
FLAG_HAS_CHAT_TEMPLATE = 1 << 4

ASI_SIG_MAGIC = b"ASISIG1\x00"
ASI_SIG_VERSION = 1
ASI_SIG_ALG_ED25519 = 1
ASI_SIG_DOMAIN = b"LILA-ASI-V2-SIG-V1\x00"

ARCH_IDS = {
    "unknown": 0,
    "dense_llama": 1,
    "hybrid_qwen35": 2,
    "dense_gemma": 3,
    "dense_qwen2": 4,
    "dense_qwen3": 5,
}

SEC_MODEL_CONFIG = 0x01
SEC_WEIGHTS = 0x02
SEC_MEMORY_FABRIC = 0x03
SEC_TOKENIZER = 0x04
SEC_PERSONALITY = 0x07
SEC_METADATA = 0x08
SEC_CHAT_TEMPLATE = 0x09
SEC_VISION = 0x0A
SEC_AUDIO = 0x0B
SEC_ADAPTERS = 0x0C
SEC_RUNTIME_HINTS = 0x0D
SEC_SIGNATURE = 0x0E
SEC_GRAPH = 0x0F
SEC_PAIRING = 0x10
SEC_SIEVE_SKETCH = 0x11

ASI_SIEVE_MAGIC = b"ASISV01\x00"

TENSOR_LAYOUT_ROW_MAJOR = 1 << 0
TENSOR_LAYOUT_AVX2_REPACK = 1 << 1
TENSOR_LAYOUT_COL_MAJOR = 1 << 2

# Tensor quant_type IDs (0..30 follow GGML; 100+ are ASI-native E07/v2.6 formats)
QUANT_F32 = 0
QUANT_F16 = 1
QUANT_Q4_0 = 2
QUANT_Q8_0 = 8
QUANT_Q4_K = 12
QUANT_Q5_K = 13
QUANT_Q6_K = 14
QUANT_BF16 = 30
QUANT_Q4_REPACK = 100
QUANT_Q3_SPLITLUT = 101


def page_align(n: int) -> int:
    return (n + ASI_V2_PAGE_SIZE - 1) & ~(ASI_V2_PAGE_SIZE - 1)


def _build_crc64_table():
    poly = 0x42F0E1EBA9EA3693
    table = []
    for i in range(256):
        crc = i << 56
        for _ in range(8):
            if crc & 0x8000000000000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFFFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFFFFFFFFFF
        table.append(crc)
    return tuple(table)


_CRC64_TABLE = _build_crc64_table()


def _load_c_lib():
    base = Path(__file__).resolve().parent
    names = ["libasi_v2.dll", "asi_v2.dll"] if sys.platform.startswith("win") else ["libasi_v2.so", "libasi_v2.dylib"]
    for name in names:
        so_path = base / name
        if not so_path.exists():
            continue
        try:
            return ctypes.CDLL(str(so_path))
        except OSError:
            continue
    return None


def _load_c_crc64():
    lib = _load_c_lib()
    if lib is None:
        return None
    try:
        fn = lib.asi_v2_crc64
        fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        fn.restype = ctypes.c_uint64
        return fn
    except AttributeError:
        return None


_C_LIB = _load_c_lib()
_C_CRC64 = _load_c_crc64()


def crc64(data: bytes) -> int:
    if _C_CRC64 is not None:
        return int(_C_CRC64(data, len(data)))
    crc = 0
    table = _CRC64_TABLE
    for byte in data:
        crc = ((crc << 8) ^ table[((crc >> 56) ^ byte) & 0xFF]) & 0xFFFFFFFFFFFFFFFF
    return crc


def pack_model_config(
    *,
    arch_id: int,
    n_layers: int,
    hidden_size: int,
    intermediate_size: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    vocab_size: int,
    max_seq_len: int,
    rope_dim: int = 0,
    rope_theta: float = 10000.0,
    rms_norm_eps: float = 1e-6,
    embed_scale: float = 1.0,
    full_attention_interval: int = 1,
    ssm_conv_kernel: int = 0,
    ssm_state_size: int = 0,
    ssm_group_count: int = 0,
    ssm_inner_size: int = 0,
    ssm_time_step_rank: int = 0,
    default_quant_type: int = 2,
) -> bytes:
    return struct.pack(
        "<IIIIIIIIIIfffIIIIIII4I",
        arch_id,
        n_layers,
        hidden_size,
        intermediate_size,
        n_heads,
        n_kv_heads,
        head_dim,
        vocab_size,
        max_seq_len,
        rope_dim,
        rope_theta,
        rms_norm_eps,
        embed_scale,
        full_attention_interval,
        ssm_conv_kernel,
        ssm_state_size,
        ssm_group_count,
        ssm_inner_size,
        ssm_time_step_rank,
        default_quant_type,
        0,
        0,
        0,
        0,
    )


def pack_weights_section(tensors: List[Tuple[str, int, int, Tuple[int, int], bytes]]) -> bytes:
    """
    tensors: list of (name, quant_type, layout_flags, (dim0, dim1), raw_bytes)
    """
    n_tensors = len(tensors)
    dir_bytes = 16 + n_tensors * 80
    data_offset = (dir_bytes + 63) & ~63  # 64-byte aligned start for SIMD loads

    entries = []
    blobs = []
    cur = 0
    for name, qtype, lflags, (d0, d1), raw in tensors:
        name_b = name.encode("utf-8")[:39]
        name_padded = name_b + b"\x00" * (40 - len(name_b))
        entries.append(
            struct.pack(
                "<40sIIQQQQ",
                name_padded,
                qtype,
                lflags,
                d0,
                d1,
                cur,
                len(raw),
            )
        )
        aligned_len = (len(raw) + 63) & ~63
        blobs.append(raw + b"\x00" * (aligned_len - len(raw)))
        cur += aligned_len

    header = struct.pack("<IIQ", n_tensors, 0, data_offset)
    pad = b"\x00" * (data_offset - dir_bytes)
    return header + b"".join(entries) + pad + b"".join(blobs)


def write_asi_v2(
    out_path: Path,
    *,
    arch_id: int,
    flags: int,
    sections: List[Tuple[int, int, bytes]],
) -> Path:
    header_size = 64
    table_offset = header_size
    table_size = len(sections) * 32
    cur_off = page_align(table_offset + table_size)

    table_entries = []
    padded_bodies = []
    digest = hashlib.sha256()
    for sec_type, sec_flags, body in sections:
        csum = crc64(body)
        digest.update(body)
        table_entries.append(struct.pack("<IIQQQ", sec_type, sec_flags, cur_off, len(body), csum))
        aligned = page_align(len(body))
        padded_bodies.append(body + b"\x00" * (aligned - len(body)))
        cur_off += aligned

    header = struct.pack(
        "<IIIIIIQQ24s",
        ASI_V2_MAGIC,
        ASI_V2_VERSION,
        arch_id,
        flags,
        len(sections),
        0,
        cur_off,
        table_offset,
        digest.digest()[:24],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(header)
        for entry in table_entries:
            f.write(entry)
        f.write(b"\x00" * (page_align(table_offset + table_size) - (table_offset + table_size)))
        for body in padded_bodies:
            f.write(body)
    return out_path


# =============================================================================
# Phase 6: Ed25519 Cryptographic Cartridge Signing & Verification
# =============================================================================


def _require_cryptography_ed25519():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    return Ed25519PrivateKey, Ed25519PublicKey, serialization


def generate_ed25519_keypair() -> Tuple[bytes, bytes]:
    """Generate a raw 32-byte Ed25519 (private_seed, public_key) pair."""
    Ed25519PrivateKey, _, serialization = _require_cryptography_ed25519()
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv_raw, pub_raw


def save_ed25519_keypair(out_dir: Path, name: str = "lila") -> Tuple[Path, Path, str]:
    """Save Ed25519 private (PEM + 0600) and public (.pub.hex + .pub.pem) keys."""
    Ed25519PrivateKey, _, serialization = _require_cryptography_ed25519()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key()
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pub_raw = pub.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    priv_path = out_dir / f"{name}.key.pem"
    pub_pem_path = out_dir / f"{name}.pub.pem"
    pub_hex_path = out_dir / f"{name}.pub.hex"

    priv_path.write_bytes(priv_pem)
    try:
        os.chmod(priv_path, 0o600)
    except OSError:
        pass
    pub_pem_path.write_bytes(pub_pem)
    pub_hex_path.write_text(pub_raw.hex() + "\n", encoding="utf-8")
    return priv_path, pub_hex_path, pub_raw.hex()


def load_ed25519_private_key(path_or_hex: str | Path) -> Tuple[bytes, bytes]:
    """Load Ed25519 private key from PEM file, raw 32-byte file, or 64-hex string."""
    Ed25519PrivateKey, _, serialization = _require_cryptography_ed25519()
    s = str(path_or_hex).strip()
    p = Path(s)
    if p.exists():
        raw = p.read_bytes()
        if raw.lstrip().startswith(b"-----") and b"PRIVATE KEY" in raw:
            priv = serialization.load_pem_private_key(raw, password=None)
        elif len(raw) == 32:
            priv = Ed25519PrivateKey.from_private_bytes(raw)
        elif len(raw.strip()) == 64:
            priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(raw.strip().decode("ascii")))
        else:
            raise ValueError(f"Unrecognized private key format in {p}")
    elif len(s) == 64:
        priv = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(s))
    else:
        raise FileNotFoundError(f"Private key not found: {path_or_hex}")

    priv_raw = priv.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv_raw, pub_raw


def load_ed25519_public_key(path_or_hex: str | Path) -> bytes:
    """Load 32-byte Ed25519 public key from 64-hex string, .pub.hex, .pub.pem, or raw 32B."""
    s = str(path_or_hex).strip()
    p = Path(s)
    if p.exists():
        raw = p.read_bytes()
        if b"BEGIN PUBLIC KEY" in raw:
            _, _, serialization = _require_cryptography_ed25519()
            pub = serialization.load_pem_public_key(raw)
            return pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        if len(raw) == 32:
            return raw
        text = raw.decode("utf-8", errors="strict").strip()
        if len(text) == 64:
            return bytes.fromhex(text)
        raise ValueError(f"Unrecognized public key format in {p}")
    if len(s) == 64:
        return bytes.fromhex(s)
    raise ValueError(f"Invalid Ed25519 public key or file not found: {path_or_hex}")


def read_asi_v2_sections(asi_path: Path) -> Tuple[dict, List[Tuple[int, int, int, bytes]]]:
    """Read header metadata and all sections `(type, flags, stored_crc, body)` from an ASI v2 file."""
    data = Path(asi_path).read_bytes()
    if len(data) < 64:
        raise ValueError("File smaller than 64-byte ASI v2 header")
    magic, version, arch_id, flags, n_sections, reserved, total_size, tbl_off, id_hash = struct.unpack(
        "<IIIIIIQQ24s", data[:64]
    )
    if magic != ASI_V2_MAGIC or version != ASI_V2_VERSION:
        raise ValueError(f"Not an ASI v2 cartridge (magic=0x{magic:08x}, version={version})")
    if n_sections == 0 or n_sections > 64 or tbl_off + n_sections * 32 > len(data):
        raise ValueError("Corrupt section table in ASI v2 cartridge")

    header = {
        "magic": magic,
        "version": version,
        "arch_id": arch_id,
        "flags": flags,
        "n_sections": n_sections,
        "reserved": reserved,
        "total_size": total_size,
        "table_offset": tbl_off,
        "identity_hash": id_hash,
    }
    sections: List[Tuple[int, int, int, bytes]] = []
    for i in range(n_sections):
        off = tbl_off + i * 32
        stype, sflags, soff, ssize, scrc = struct.unpack("<IIQQQ", data[off : off + 32])
        if soff + ssize > len(data):
            raise ValueError(f"Section 0x{stype:02x} out of bounds")
        sections.append((stype, sflags, scrc, data[soff : soff + ssize]))
    return header, sections


def compute_signing_digest(
    *,
    magic: int,
    version: int,
    arch_id: int,
    flags: int,
    non_sig_sections: List[Tuple[int, int, int, bytes]],
    sig_version: int,
    alg_id: int,
    signed_at_unix: int,
    publisher_padded: bytes,
    public_key: bytes,
) -> bytes:
    """Compute canonical SHA-256 digest matching asi_v2.c `asi_v2_compute_signing_digest` byte-for-byte."""
    h = hashlib.sha256()
    h.update(ASI_SIG_DOMAIN)
    h.update(
        struct.pack(
            "<IIIII",
            magic,
            version,
            arch_id,
            flags & ~FLAG_HAS_SIGNATURE,
            len(non_sig_sections),
        )
    )
    for stype, sflags, scrc, body in non_sig_sections:
        sec_sha = hashlib.sha256(body).digest()
        h.update(struct.pack("<IIQQ", stype, sflags, len(body), scrc))
        h.update(sec_sha)
    h.update(struct.pack("<IIQ", sig_version, alg_id, signed_at_unix))
    h.update(publisher_padded)
    h.update(public_key)
    return h.digest()


def sign_asi_v2(
    in_path: Path,
    out_path: Optional[Path] = None,
    *,
    private_key_bytes: bytes,
    publisher: str = "lila",
    signed_at_unix: Optional[int] = None,
) -> Tuple[Path, bytes]:
    """Sign an ASI v2 cartridge with Ed25519 (`ASI_SEC_SIGNATURE`), writing atomically."""
    Ed25519PrivateKey, _, serialization = _require_cryptography_ed25519()
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path is not None else in_path

    priv = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    pub_raw = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    header, raw_sections = read_asi_v2_sections(in_path)
    non_sig: List[Tuple[int, int, int, bytes]] = []
    for stype, sflags, scrc, body in raw_sections:
        if stype == SEC_SIGNATURE:
            continue
        actual_crc = crc64(body)
        if actual_crc != scrc:
            raise ValueError(f"Refusing to sign corrupt section 0x{stype:02x}: CRC-64 mismatch")
        non_sig.append((stype, sflags, scrc, body))

    ts = int(time.time()) if signed_at_unix is None else int(signed_at_unix)
    pub_b = publisher.encode("utf-8")[:23]
    pub_padded = pub_b + b"\x00" * (24 - len(pub_b))

    digest = compute_signing_digest(
        magic=header["magic"],
        version=header["version"],
        arch_id=header["arch_id"],
        flags=header["flags"],
        non_sig_sections=non_sig,
        sig_version=ASI_SIG_VERSION,
        alg_id=ASI_SIG_ALG_ED25519,
        signed_at_unix=ts,
        publisher_padded=pub_padded,
        public_key=pub_raw,
    )
    sig_bytes = priv.sign(digest)

    sig_block = struct.pack(
        "<8sIIQ24s32s64s",
        ASI_SIG_MAGIC,
        ASI_SIG_VERSION,
        ASI_SIG_ALG_ED25519,
        ts,
        pub_padded,
        pub_raw,
        sig_bytes,
    )
    new_sections = [(stype, sflags, body) for stype, sflags, _, body in non_sig]
    new_sections.append((SEC_SIGNATURE, 0, sig_block))

    tmp_out = out_path.with_name(f".{out_path.name}.signing.{os.getpid()}.tmp")
    write_asi_v2(
        tmp_out,
        arch_id=header["arch_id"],
        flags=header["flags"] | FLAG_HAS_SIGNATURE,
        sections=new_sections,
    )
    os.replace(tmp_out, out_path)
    return out_path, pub_raw


def verify_asi_v2(
    asi_path: Path,
    *,
    trusted_pubkey: Optional[bytes] = None,
    use_c: bool = True,
) -> dict:
    """Verify CRC-64 and Ed25519 signature of an ASI v2 cartridge (Python +/or C verifier)."""
    try:
        from cryptography.exceptions import InvalidSignature
        _, Ed25519PublicKey, _ = _require_cryptography_ed25519()
        has_py_crypto = True
    except ImportError:
        InvalidSignature = Exception
        Ed25519PublicKey = None
        has_py_crypto = False

    if not has_py_crypto and (not use_c or _C_LIB is None):
        raise ValueError("Neither `cryptography` nor C `libasi_v2` verifier is available.")

    asi_path = Path(asi_path)
    header, raw_sections = read_asi_v2_sections(asi_path)

    non_sig: List[Tuple[int, int, int, bytes]] = []
    sig_body: Optional[bytes] = None
    for stype, sflags, scrc, body in raw_sections:
        if crc64(body) != scrc:
            raise ValueError(f"Section 0x{stype:02x} CRC-64 checksum mismatch")
        if stype == SEC_SIGNATURE:
            sig_body = body
        else:
            non_sig.append((stype, sflags, scrc, body))

    if sig_body is None:
        raise ValueError("Cartridge is unsigned (missing ASI_SEC_SIGNATURE)")
    if len(sig_body) < 144:
        raise ValueError("SIGNATURE section truncated (< 144 bytes)")

    smagic, sver, alg_id, ts, pub_padded, pub_raw, sig_bytes = struct.unpack(
        "<8sIIQ24s32s64s", sig_body[:144]
    )
    if smagic != ASI_SIG_MAGIC or sver != ASI_SIG_VERSION or alg_id != ASI_SIG_ALG_ED25519:
        raise ValueError("Unsupported or invalid SIGNATURE block header")
    if trusted_pubkey is not None and pub_raw != trusted_pubkey:
        raise ValueError("Signer public key does not match trusted publisher key")

    if has_py_crypto:
        digest = compute_signing_digest(
            magic=header["magic"],
            version=header["version"],
            arch_id=header["arch_id"],
            flags=header["flags"],
            non_sig_sections=non_sig,
            sig_version=sver,
            alg_id=alg_id,
            signed_at_unix=ts,
            publisher_padded=pub_padded,
            public_key=pub_raw,
        )
        try:
            Ed25519PublicKey.from_public_bytes(pub_raw).verify(sig_bytes, digest)
        except InvalidSignature as exc:
            raise ValueError("Ed25519 signature verification failed") from exc

    c_verified = False
    if use_c and _C_LIB is not None and hasattr(_C_LIB, "asi_v2_open") and hasattr(_C_LIB, "asi_v2_verify_signature"):
        class _Cart(ctypes.Structure):
            _fields_ = [("_raw", ctypes.c_uint8 * 512)]

        class _SigInfo(ctypes.Structure):
            _fields_ = [
                ("valid", ctypes.c_int),
                ("alg_id", ctypes.c_uint32),
                ("signed_at_unix", ctypes.c_uint64),
                ("publisher", ctypes.c_char * 25),
                ("public_key", ctypes.c_uint8 * 32),
            ]

        cart = _Cart()
        err = ctypes.create_string_buffer(256)
        _C_LIB.asi_v2_open.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        _C_LIB.asi_v2_open.restype = ctypes.c_int
        _C_LIB.asi_v2_close.argtypes = [ctypes.c_void_p]
        _C_LIB.asi_v2_verify_signature.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_SigInfo), ctypes.c_char_p, ctypes.c_size_t
        ]
        _C_LIB.asi_v2_verify_signature.restype = ctypes.c_int

        if _C_LIB.asi_v2_open(str(asi_path).encode("utf-8"), ctypes.byref(cart), err, 256) != 0:
            raise ValueError(f"C validator failed: {err.value.decode('utf-8', 'replace')}")
        try:
            info = _SigInfo()
            pk_arg = ctypes.c_char_p(trusted_pubkey) if trusted_pubkey is not None else None
            rc = _C_LIB.asi_v2_verify_signature(ctypes.byref(cart), pk_arg, ctypes.byref(info), err, 256)
            if rc != 0:
                raise ValueError(f"C Ed25519 verification failed: {err.value.decode('utf-8', 'replace')}")
            c_verified = True
        finally:
            _C_LIB.asi_v2_close(ctypes.byref(cart))

    publisher_name = pub_padded.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return {
        "valid": True,
        "c_verified": c_verified,
        "alg_id": alg_id,
        "publisher": publisher_name,
        "signed_at_unix": ts,
        "public_key_hex": pub_raw.hex(),
        "sections": len(raw_sections),
    }
