"""
Lila ASI Runtime Backend — standalone ctypes runtime for liblila_asi.so + .asi cartridges.
Zero dependency on llama.cpp; exposes the create_chat_completion() interface used by LilaCore.
"""

from __future__ import annotations

import codecs
from collections import deque
import ctypes
import hashlib
import json
import logging
import os
import re
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from .private_storage import restrictive_umask, secure_private_directory
except ImportError:
    try:
        from src.core.private_storage import restrictive_umask, secure_private_directory
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def restrictive_umask():
            prev = os.umask(0o077)
            try:
                yield
            finally:
                os.umask(prev)

        def secure_private_directory(path: Path) -> Path:
            d = Path(path)
            if d.is_symlink():
                raise PermissionError(f"Private directory must not be a symlink: {d}")
            with restrictive_umask():
                d.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(d, 0o700)
            return d

_SEC_TOKENIZER = 0x04
_SEC_CHAT_TEMPLATE = 0x09
_DEFAULT_PREFIX_CACHE_MAX_FILES = 8

ASI_ENGINE_VERSION = "ASI v2.6"
# Feature-scoped working names (per naming policy):
# - Sieve: two-stage full-vocabulary lm_head screening (E03)
# - Anchor: CoW state tree, crash-proof WAL, live confidence & provenance (E06)
# - Torrent: Q4_REPACK, BLIS prefill, ChunkWY-32, sparse ffn_down, B<=2 spec (E07)
ASI_SIEVE_WORKING_NAME = "Sieve"
ASI_ANCHOR_WORKING_NAME = "Anchor"
ASI_TORRENT_WORKING_NAME = "Torrent"
ASI_ACTIVE_CODENAME: Optional[str] = None

# Plain-English engine presets for non-technical users (LILA_ASI_MODE=fast|balanced|exact).
# Never modifies model weights; controls how the engine trades speed, memory, and exactness.
ASI_MODE_PRESETS: Dict[str, Dict[str, object]] = {
    "fast": {
        "title": "Fast (Default)",
        "what_it_is": (
            "Everyday high-speed mode using Sieve (1-bit dictionary skimmer) plus smart autocomplete."
        ),
        "what_it_does": (
            "Instead of reading all 248,320 full word entries on every word, the engine skims a tiny "
            "31 MiB summary of every word in every language to pick the top 2,048 finalists, then checks "
            "only those finalists with exact math. Smart autocomplete guesses obvious repeats ahead."
        ),
        "how_it_affects_engine_and_model": (
            "Does NOT change model weights or drop any language. Runs ~35-45% faster than exact mode "
            "(~20 tok/s on 2-core laptop), uses +31 MiB extra RAM (645 MiB total) and +0.6 s once at boot, "
            "and picks the exact same top word 99.96% of the time (5,060/5,062 tested positions)."
        ),
        "env": {
            "LILA_ASI_MODE": "fast",
            "LILA_ASI_LMHEAD_MODE": "1bit1024",
            "LILA_ASI_LMHEAD_K": "2048",
            "LILA_ASI_SPECULATIVE": "1",
            "LILA_ASI_SPEC_GATE": "1",
        },
    },
    "balanced": {
        "title": "Balanced (Extra-Strict Accuracy)",
        "what_it_is": (
            "High-precision fast mode using a 2-bit Sieve dictionary summary plus smart autocomplete."
        ),
        "what_it_does": (
            "Uses a slightly richer 61 MiB summary of every word before checking the top 1,024 finalists "
            "with exact math, preserving not only the #1 word but also the full top-64 word rankings "
            "used during creative/random sampling."
        ),
        "how_it_affects_engine_and_model": (
            "Does NOT change model weights. Slightly slower than 'fast' (~17-19 tok/s) and uses +61 MiB "
            "extra RAM (670 MiB total) and +1.2 s once at boot, while achieving 100.00% top-64 recall "
            "and near-zero probability drift (KL = 0.00002)."
        ),
        "env": {
            "LILA_ASI_MODE": "balanced",
            "LILA_ASI_LMHEAD_MODE": "2bit1024",
            "LILA_ASI_LMHEAD_K": "1024",
            "LILA_ASI_SPECULATIVE": "1",
            "LILA_ASI_SPEC_GATE": "1",
        },
    },
    "exact": {
        "title": "Exact (Zero Shortcuts & Lowest RAM)",
        "what_it_is": (
            "100% exact dictionary math, one word at a time, with all speed shortcuts turned off."
        ),
        "what_it_does": (
            "Reads every single full entry of the 248,320-word dictionary on every word and disables "
            "multi-word autocomplete guessing."
        ),
        "how_it_affects_engine_and_model": (
            "Does NOT change model weights. Uses the least memory (~610 MiB RAM), starts fastest (~0.08 s), "
            "and guarantees 100.000% exact baseline math for strict audits or tight-RAM machines, at "
            "slower everyday writing speed (~15-16 tok/s)."
        ),
        "env": {
            "LILA_ASI_MODE": "exact",
            "LILA_ASI_LMHEAD_MODE": "off",
            "LILA_ASI_SPECULATIVE": "0",
            "LILA_ASI_SPEC_GATE": "0",
        },
    },
}


def resolve_asi_mode(mode: Optional[str] = None) -> str:
    """Resolve active plain-English preset ('fast', 'balanced', or 'exact')."""
    raw = (mode if mode is not None else os.environ.get("LILA_ASI_MODE", "")).strip().lower()
    if raw in ASI_MODE_PRESETS:
        return raw
    lhm = os.environ.get("LILA_ASI_LMHEAD_MODE", "").strip().lower()
    if lhm in ("off", "0", "none", "false"):
        return "exact"
    if lhm == "2bit1024":
        return "balanced"
    return "fast"


def get_asi_mode_preset(mode: Optional[str] = None) -> Dict[str, object]:
    """Return the dictionary description and environment variables for a preset ('fast', 'balanced', 'exact')."""
    resolved = resolve_asi_mode(mode)
    return dict(ASI_MODE_PRESETS[resolved])


def apply_asi_mode_preset(mode: str = "fast") -> str:
    """Apply the environment variables for a plain-English preset ('fast', 'balanced', or 'exact')."""
    resolved = resolve_asi_mode(mode)
    env_map = ASI_MODE_PRESETS[resolved].get("env", {})
    if isinstance(env_map, dict):
        for k, v in env_map.items():
            os.environ[str(k)] = str(v)
    return resolved


def describe_asi_modes() -> Dict[str, Dict[str, object]]:
    """Return a copy of all plain-English ASI mode presets."""
    return {k: dict(v) for k, v in ASI_MODE_PRESETS.items()}

ASI_VERSION_CATALOG: Dict[str, Dict[str, Optional[object]]] = {
    "ASI v2.0": {
        "codename": None,
        "status": "historical",
        "location": "git 30d89a8..652ccb6 (Trials 01-06)",
        "summary": "Founding standalone C11/AVX2 .asi v2 cartridge runtime with mmap and zero-copy tokenizer.",
    },
    "ASI v2.1": {
        "codename": None,
        "status": "historical",
        "location": "git db0a696..49d958e (Trials 07-09)",
        "summary": "1x4 register prefill GEMM, dTLB T_TILE, 1x-unpack tile_out, bounded-spin pool, FP32-Ref audit mode.",
    },
    "ASI v2.2": {
        "codename": None,
        "status": "merged_into_production",
        "location": "git 969f703..d3eec6c (Trial 10)",
        "summary": "dot2 default kernels, SMT-aware pool, ASIPFX02 disk prefix snapshots, AVX-512 8-lane alignment.",
    },
    "ASI v2.3": {
        "codename": None,
        "status": "superseded_by_v2.4",
        "location": "git bf85dbe (Trial 12, archived SHA-256 c3a468f...)",
        "runtime_modes": {
            "LILA_ASI_SPECULATIVE=1 ('working')": (
                "Same binary with zero-rollback stack-scratch n-gram speculation enabled."
            ),
            "LILA_ASI_SPECULATIVE=0 ('working_nospec')": (
                "Same binary with speculation disabled (straight-line B=1 decode)."
            ),
        },
        "summary": "vpshufb scale broadcasts, 32-byte aligned Q8K, parallel conv1d/full_attn, zero-rollback stack-scratch speculation.",
    },
    "ASI v2.4 (with Sieve)": {
        "codename": None,
        "feature_working_name": ASI_SIEVE_WORKING_NAME,
        "status": "active_promoted_source",
        "location": "engine/asi/liblila_asi.so (promoted from Exp E03 + Exp E04)",
        "runtime_modes": {
            "Default (Sieve ON: 1bit1024, K=2048, spec_gate=1)": (
                "Full-vocab 2-stage 1-bit screening + sorted exact Q6_K Stage-2 refinement + "
                "confidence-gated speculation (20.57/19.80 tok/s, 5,060/5,062 = 99.96% exact top-1)."
            ),
            "LILA_ASI_LMHEAD_MODE=off": (
                "Exact full-vocabulary Q6_K lm_head (identical exact path to v2.3)."
            ),
            "Dense Transformer Handler (Llama/Gemma/Qwen2/Qwen3)": (
                "Standard dense architecture support via g_asi_arch_dense_transformer with Sieve hook."
            ),
        },
        "summary": "Promoted v2.4 runtime integrating Sieve (working name for 2-stage full-vocab lm_head screening) + dense transformer support.",
    },
    "Exp E05 (grammar_sparse_head)": {
        "codename": None,
        "status": "active_promoted_source",
        "location": "engine/asi/liblila_asi.so (asi_grammar_sparse.c + e05_bridge.c)",
        "summary": "Model-triggered C11 DFA <tool_call> format guardrails + Grammar-Aware Sparse lm_head + forced-token fast-forward (v2: 512 tools, optional args, RFC-8259 UTF-8 strings).",
    },
    "Exp E06 (Anchor)": {
        "codename": None,
        "feature_working_name": ASI_ANCHOR_WORKING_NAME,
        "status": "active_promoted_source",
        "location": "engine/asi/liblila_asi.so (asi_state_centric.c)",
        "summary": "Anchor (E06): C Forkable CoW State Tree (ASITREE1) + per-branch provenance + WAL SIGKILL recovery + Confidence API.",
    },
    "ASI v2.5 (Native Vision & Universal Graph)": {
        "codename": None,
        "status": "active_promoted_source",
        "location": "engine/asi/liblila_asi.so (asi_vision.c, asi_kernels.c, asi_graph.c, asi_pairing.c, asi_vision_sdk.py)",
        "summary": "V1-V9 Two-Tier Universal Vision Runtime + 3D M-RoPE multimodal decoder + Immutable V8 Pairing + V9 SDK & Active Inspection.",
    },
    "ASI v2.6 (Torrent)": {
        "codename": None,
        "feature_working_name": ASI_TORRENT_WORKING_NAME,
        "status": "active_promoted_source",
        "location": "engine/asi/liblila_asi.so (q4k.h, asi_arch.c, asi_v2.c, asi_requant.py)",
        "summary": "Torrent (E07): Q4_REPACK (4.25 bpw) + sparse col-major ffn_down + ChunkWY-32 DeltaNet prefill + BLIS 6x16 (>=96 tok) + B<=2 spec cap + MADV_HUGEPAGE + slice-by-8 CRC-64.",
    },
}


def active_asi_label() -> str:
    """Return the active ASI engine version, plain-English preset, Sieve status, and speculative mode."""
    preset = resolve_asi_mode()
    lhm = os.environ.get("LILA_ASI_LMHEAD_MODE", "").strip().lower()
    sieve_off = lhm in ("off", "0", "none", "false") or (not lhm and preset == "exact")
    sieve_tag = "exact lm_head" if sieve_off else f"with {ASI_SIEVE_WORKING_NAME}"
    spec_env = os.environ.get("LILA_ASI_SPECULATIVE", "").strip()
    spec_off = (spec_env == "0") or (not spec_env and preset == "exact")
    spec_tag = "speculative=off" if spec_off else "speculative=on"
    return f"{ASI_ENGINE_VERSION} {sieve_tag} [mode={preset}] ({spec_tag})"


active_asi_codename = active_asi_label


class _AsiBranchProvenance(ctypes.Structure):
    _fields_ = [
        ("branch_id", ctypes.c_uint32),
        ("parent_branch_id", ctypes.c_uint32),
        ("state_version", ctypes.c_uint32),
        ("n_tokens", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("token_trajectory_sha256", ctypes.c_uint8 * 32),
    ]


class _AsiCheckpointInfo(ctypes.Structure):
    _fields_ = [
        ("cp_id", ctypes.c_uint32),
        ("parent_cp_id", ctypes.c_uint32),
        ("rng_state", ctypes.c_uint64),
        ("prov", _AsiBranchProvenance),
        ("label", ctypes.c_char * 64),
    ]


class _AsiCowStats(ctypes.Structure):
    _fields_ = [
        ("live_slots", ctypes.c_uint32),
        ("max_slots", ctypes.c_uint32),
        ("physical_kv_pages", ctypes.c_uint32),
        ("logical_kv_pages", ctypes.c_uint32),
        ("shared_kv_pages", ctypes.c_uint32),
        ("cow_reuses", ctypes.c_uint64),
        ("cow_allocs", ctypes.c_uint64),
    ]


class _AsiV2Header(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("arch_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("n_sections", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("total_size", ctypes.c_uint64),
        ("section_table_offset", ctypes.c_uint64),
        ("identity_hash", ctypes.c_uint8 * 24),
    ]


assert ctypes.sizeof(_AsiV2Header) == 64


class _AsiV2ModelConfig(ctypes.Structure):
    _fields_ = [
        ("arch_id", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("n_heads", ctypes.c_uint32),
        ("n_kv_heads", ctypes.c_uint32),
        ("head_dim", ctypes.c_uint32),
        ("vocab_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32),
        ("rope_dim", ctypes.c_uint32),
        ("rope_theta", ctypes.c_float),
        ("rms_norm_eps", ctypes.c_float),
        ("embed_scale", ctypes.c_float),
        ("full_attention_interval", ctypes.c_uint32),
        ("ssm_conv_kernel", ctypes.c_uint32),
        ("ssm_state_size", ctypes.c_uint32),
        ("ssm_group_count", ctypes.c_uint32),
        ("ssm_inner_size", ctypes.c_uint32),
        ("ssm_time_step_rank", ctypes.c_uint32),
        ("default_quant_type", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _AsiToolArgSchema(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("type", ctypes.c_int),
        ("max_len", ctypes.c_uint32),
        ("max_digits", ctypes.c_uint32),
        ("enum_values", ctypes.c_char_p * 16),
        ("n_enum_values", ctypes.c_uint32),
        ("optional", ctypes.c_int),
    ]


class _AsiToolSchema(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("args", _AsiToolArgSchema * 16),
        ("n_args", ctypes.c_uint32),
    ]


class _AsiGrammarConfig(ctypes.Structure):
    _fields_ = [
        ("wrap_xml", ctypes.c_int),
        ("default_max_str", ctypes.c_uint32),
        ("default_max_digits", ctypes.c_uint32),
        ("sparse_limit", ctypes.c_uint32),
    ]


class _AsiGrammarVocab(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_uint32),
        ("tokens", ctypes.POINTER(ctypes.c_char_p)),
        ("token_lens", ctypes.POINTER(ctypes.c_uint32)),
        ("eos_token_id", ctypes.c_int32),
        ("im_start_id", ctypes.c_int32),
        ("im_end_id", ctypes.c_int32),
        ("tool_call_open_id", ctypes.c_int32),
        ("tool_call_close_id", ctypes.c_int32),
    ]


def _gpt2_bytes_to_unicode() -> Dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class AsiBpeTokenizer:
    """Byte-level BPE tokenizer backed by SEC_TOKENIZER inside model.asi."""

    # Bump whenever special-token rules, pre-tokenization, or BPE behavior changes
    # so old prefix snapshots / caches keyed on previous token IDs refuse to load.
    TOKENIZER_VERSION: int = 2

    _PRE_RE = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, tok_json: dict):
        self.tokens: List[str] = tok_json.get("tokens", [])
        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        self.eos_token_id: int = int(tok_json.get("eos_token_id") or 248046)
        self.b2u = _gpt2_bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

        merges_raw = tok_json.get("merges", [])
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for idx, m in enumerate(merges_raw):
            if isinstance(m, str) and " " in m:
                a, b = m.split(" ", 1)
                self.bpe_ranks[(a, b)] = idx

        token_types = tok_json.get("token_type", [])
        special_set = set()
        if isinstance(token_types, list):
            for idx, tt in enumerate(token_types):
                if tt != 1 and idx < len(self.tokens):
                    tok_str = self.tokens[idx]
                    if isinstance(tok_str, str) and tok_str.startswith("<") and tok_str.endswith(">"):
                        special_set.add(tok_str)
        for s in (
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<think>",
            "</think>",
        ):
            if s in self.vocab:
                special_set.add(s)
        # Longest-first so prefix-overlapping special tags never split prematurely
        self._special_tokens = sorted(special_set, key=lambda x: (-len(x), x))
        self._assistant_special_tokens = [
            s
            for s in ("<think>", "</think>", "<tool_call>", "</tool_call>")
            if s in self.vocab
        ]
        self._bpe_cache: Dict[str, Tuple[str, ...]] = {}

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        if len(word) <= 1 or not self.bpe_ranks:
            return word
        while True:
            best_pair = None
            best_rank = 1 << 30
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.bpe_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            first, second = best_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
        self._bpe_cache[token] = word
        return word

    def encode_untrusted(self, text: str) -> List[int]:
        """Encode untrusted content (user text, tool outputs, web pages) as pure byte-level BPE
        without parsing any `<...>` control or special tokens into atomic control IDs.
        """
        return self.encode(text, allow_special=False)

    def encode(
        self,
        text: str,
        *,
        allow_special: Union[bool, Sequence[str]] = True,
    ) -> List[int]:
        if allow_special is True:
            active_specials = self._special_tokens
        elif allow_special is False or not allow_special:
            active_specials = []
        else:
            allowed_set = {s for s in allow_special if s in self.vocab}
            active_specials = sorted(allowed_set, key=lambda x: (-len(x), x))

        if active_specials:
            pattern = "(" + "|".join(re.escape(s) for s in active_specials) + ")"
            chunks = re.split(pattern, text)
            active_set = set(active_specials)
        else:
            chunks = [text]
            active_set = set()

        ids: List[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in active_set and chunk in self.vocab:
                ids.append(self.vocab[chunk])
                continue
            for piece in self._PRE_RE.findall(chunk) or [chunk]:
                bstr = "".join(self.b2u[b] for b in piece.encode("utf-8"))
                for btok in self._bpe(bstr):
                    tid = self.vocab.get(btok)
                    if tid is not None:
                        ids.append(tid)
                    else:
                        for ch in btok:
                            if ch in self.vocab:
                                ids.append(self.vocab[ch])
        return ids

    def token_bytes(self, ids: Sequence[int]) -> bytes:
        out = bytearray()
        for tid in ids:
            if 0 <= tid < len(self.tokens):
                s = self.tokens[tid]
                for ch in s:
                    if ch in self.u2b:
                        out.append(self.u2b[ch])
                    else:
                        out.extend(ch.encode("utf-8"))
        return bytes(out)

    def decode(self, ids: List[int]) -> str:
        return self.token_bytes(ids).decode("utf-8", errors="replace")

    def create_incremental_decoder(self, errors: str = "replace") -> "IncrementalBpeUtf8Decoder":
        return IncrementalBpeUtf8Decoder(self, errors=errors)

    def iter_decode_tokens(self, ids: Sequence[int], errors: str = "replace") -> Iterator[str]:
        decoder = self.create_incremental_decoder(errors=errors)
        for tid in ids:
            piece = decoder.decode_token(tid, final=False)
            if piece:
                yield piece
        tail = decoder.flush()
        if tail:
            yield tail


class IncrementalBpeUtf8Decoder:
    """
    Incremental UTF-8 byte decoder for AsiBpeTokenizer.
    Buffers partial multi-byte UTF-8 sequences split across token boundaries
    until completed or flushed at stream end.
    """

    def __init__(self, tokenizer: AsiBpeTokenizer, errors: str = "replace"):
        self._tokenizer = tokenizer
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors=errors)

    def decode_tokens(self, ids: Sequence[int], final: bool = False) -> str:
        raw = self._tokenizer.token_bytes(ids)
        return self._utf8_decoder.decode(raw, final=final)

    def decode_token(self, token_id: int, final: bool = False) -> str:
        return self.decode_tokens((token_id,), final=final)

    def flush(self) -> str:
        return self._utf8_decoder.decode(b"", final=True)

    def reset(self) -> None:
        self._utf8_decoder.reset()


def _host_supports_avx2_fma_f16c() -> bool:
    """Return True when the host CPU supports AVX2 + FMA + F16C for portable library selection."""
    override = os.environ.get("LILA_ASI_ARCH", "").strip().lower()
    if override == "scalar":
        return False
    if override == "avx2":
        return True
    try:
        if sys.platform.startswith("linux"):
            info = Path("/proc/cpuinfo").read_text(errors="ignore").lower()
            flags_line = next((ln for ln in info.splitlines() if ln.startswith("flags")), "")
            tokens = set(flags_line.split())
            return {"avx2", "fma", "f16c"}.issubset(tokens)
        if sys.platform.startswith("win"):
            # Windows IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE == 40)
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    except Exception:
        pass
    return False


def _default_lib_path() -> Path:
    env_path = os.environ.get("LILA_ASI_LIB")
    if env_path:
        return Path(env_path)
    here = Path(__file__).resolve().parent
    has_avx2 = _host_supports_avx2_fma_f16c()
    if sys.platform.startswith("win"):
        names = (
            ("lila_asi_avx2.dll", "lila_asi.dll", "lila_asi_scalar.dll")
            if has_avx2
            else ("lila_asi_scalar.dll", "lila_asi.dll")
        )
    elif sys.platform == "darwin":
        names = ("liblila_asi.dylib",)
    else:
        names = (
            ("liblila_asi.so", "liblila_asi_portable_avx2.so", "liblila_asi_portable_scalar.so")
            if has_avx2
            else ("liblila_asi_portable_scalar.so", "liblila_asi.so")
        )

    search_dirs = (
        here,
        here.parent / "lib",
        here.parents[1] / "engine" / "asi",
    )
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return here.parents[1] / "engine" / "asi" / names[0]


class AsiRuntime:
    """
    Standalone Python runtime for liblila_asi.so / lila_asi.dll + .asi cartridges (zero llama.cpp dependency).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int = 0,
        lib_path: Optional[str] = None,
        prefix_cache_dir: Optional[Union[str, Path]] = None,
        **_: object,
    ):
        so = Path(lib_path) if lib_path else _default_lib_path()
        if not so.exists():
            raise FileNotFoundError(f"ASI runtime shared library not found at {so}")

        self._lib = ctypes.CDLL(str(so))
        self._has_model_header = False
        self._supports_prefix_file = False
        self._bind_abi()

        err = ctypes.create_string_buffer(256)
        self._model = self._lib.lila_asi_open(model_path.encode("utf-8"), err, 256)
        if not self._model:
            raise RuntimeError(f"lila_asi_open failed: {err.value.decode('utf-8', 'replace')}")

        self._cfg = _AsiV2ModelConfig()
        self._lib.lila_asi_model_info(self._model, ctypes.byref(self._cfg))

        self._model_hdr = _AsiV2Header()
        if self._has_model_header:
            if self._lib.lila_asi_model_header(self._model, ctypes.byref(self._model_hdr)) != 0:
                self._model_hdr = _AsiV2Header()
                self._supports_prefix_file = False

        sec_ptr = ctypes.c_void_p()
        sec_sz = ctypes.c_size_t()
        if self._lib.lila_asi_get_section(self._model, _SEC_TOKENIZER, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) != 0:
            self._lib.lila_asi_close(self._model)
            raise RuntimeError("SEC_TOKENIZER missing from ASI cartridge")

        tok_json = json.loads(ctypes.string_at(sec_ptr.value, sec_sz.value).decode("utf-8"))
        self.tokenizer = AsiBpeTokenizer(tok_json)

        self.chat_template = ""
        if self._lib.lila_asi_get_section(self._model, _SEC_CHAT_TEMPLATE, ctypes.byref(sec_ptr), ctypes.byref(sec_sz)) == 0:
            self.chat_template = ctypes.string_at(sec_ptr.value, sec_sz.value).decode("utf-8", "replace")

        self._session = self._lib.lila_asi_session_create(self._model, int(n_ctx), int(n_threads), err, 256)
        if not self._session:
            self._lib.lila_asi_close(self._model)
            raise RuntimeError(f"lila_asi_session_create failed: {err.value.decode('utf-8', 'replace')}")

        raw_cache_dir: Optional[Union[str, Path]] = prefix_cache_dir
        if raw_cache_dir is None and os.environ.get("LILA_ASI_DISABLE_DISK_PREFIX") != "1":
            env_cache_dir = os.environ.get("LILA_ASI_PREFIX_CACHE_DIR", "").strip()
            raw_cache_dir = env_cache_dir or None
        elif isinstance(raw_cache_dir, str):
            raw_cache_dir = raw_cache_dir.strip() or None

        self._prefix_cache_dir: Optional[Path] = (
            Path(os.path.abspath(Path(raw_cache_dir).expanduser()))
            if raw_cache_dir is not None
            else None
        )

        self._logits = (ctypes.c_float * self._cfg.vocab_size)()
        self._rng = ctypes.c_uint64(0x9E3779B97F4A7C15)
        self._n_threads = max(1, int(n_threads or 2))
        self._cached_prefix_ids: Optional[List[int]] = None
        self._cached_prefix_text: Optional[str] = None
        self._session_token_ids: List[int] = []
        self._vision_session: Optional[Any] = None
        self._vision_controller: Optional[Any] = None
        self._vision_asiv_path: Optional[str] = None
        self.engine_version = ASI_ENGINE_VERSION
        self.runtime_label = active_asi_label()
        self.codename = ASI_ACTIVE_CODENAME
        self.last_prefix_cache_hit: Optional[str] = None
        self.prefix_cache_stats: Dict[str, int] = {"ram": 0, "disk": 0, "miss": 0, "session": 0}
        self._lock = threading.Lock()

    def _bind_abi(self) -> None:
        lib = self._lib
        lib.lila_asi_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
        lib.lila_asi_open.restype = ctypes.c_void_p
        lib.lila_asi_close.argtypes = [ctypes.c_void_p]
        lib.lila_asi_model_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AsiV2ModelConfig)]
        lib.lila_asi_model_info.restype = ctypes.c_int
        lib.lila_asi_get_section.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)
        ]
        lib.lila_asi_get_section.restype = ctypes.c_int
        lib.lila_asi_session_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t
        ]
        lib.lila_asi_session_create.restype = ctypes.c_void_p
        lib.lila_asi_session_free.argtypes = [ctypes.c_void_p]
        lib.lila_asi_session_reset.argtypes = [ctypes.c_void_p]
        lib.lila_asi_session_save_prefix.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        lib.lila_asi_session_save_prefix.restype = ctypes.c_int
        lib.lila_asi_session_restore_prefix.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        lib.lila_asi_session_restore_prefix.restype = ctypes.c_int
        lib.lila_asi_eval_token.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        lib.lila_asi_eval_token.restype = ctypes.c_int
        lib.lila_asi_eval_batch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)
        ]
        lib.lila_asi_eval_batch.restype = ctypes.c_int
        lib.lila_asi_sample.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_float, ctypes.POINTER(ctypes.c_uint64)
        ]
        lib.lila_asi_sample.restype = ctypes.c_int32

        self._has_model_header = hasattr(lib, "lila_asi_model_header")
        if self._has_model_header:
            lib.lila_asi_model_header.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_AsiV2Header),
            ]
            lib.lila_asi_model_header.restype = ctypes.c_int

        has_save_prefix_file = hasattr(lib, "lila_asi_session_save_prefix_file")
        if has_save_prefix_file:
            lib.lila_asi_session_save_prefix_file.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_char_p,
            ]
            lib.lila_asi_session_save_prefix_file.restype = ctypes.c_int

        has_load_prefix_file = hasattr(lib, "lila_asi_session_load_prefix_file")
        if has_load_prefix_file:
            lib.lila_asi_session_load_prefix_file.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.lila_asi_session_load_prefix_file.restype = ctypes.c_int

        self._supports_prefix_file = bool(
            self._has_model_header and has_save_prefix_file and has_load_prefix_file
        )

        self._supports_e05 = bool(
            hasattr(lib, "asi_grammar_create")
            and hasattr(lib, "e05_load_vocab_from_model")
            and hasattr(lib, "e05_hook_configure")
            and hasattr(lib, "e05_hook_set_skip_lm_head")
        )
        if self._supports_e05:
            lib.e05_load_vocab_from_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AsiGrammarVocab)]
            lib.e05_load_vocab_from_model.restype = ctypes.c_int
            lib.asi_grammar_vocab_free.argtypes = [ctypes.POINTER(_AsiGrammarVocab)]
            lib.asi_grammar_vocab_free.restype = None
            lib.asi_grammar_create.argtypes = [
                ctypes.POINTER(_AsiGrammarVocab),
                ctypes.POINTER(_AsiToolSchema),
                ctypes.c_uint32,
                ctypes.POINTER(_AsiGrammarConfig),
            ]
            lib.asi_grammar_create.restype = ctypes.c_void_p
            lib.asi_grammar_free.argtypes = [ctypes.c_void_p]
            lib.asi_grammar_free.restype = None
            lib.asi_grammar_reset.argtypes = [ctypes.c_void_p]
            lib.asi_grammar_reset.restype = None
            lib.asi_grammar_is_complete.argtypes = [ctypes.c_void_p]
            lib.asi_grammar_is_complete.restype = ctypes.c_int
            lib.asi_grammar_accept_token.argtypes = [ctypes.c_void_p, ctypes.c_int32]
            lib.asi_grammar_accept_token.restype = ctypes.c_int
            lib.asi_grammar_collect_forced_run.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.asi_grammar_collect_forced_run.restype = ctypes.c_int
            lib.e05_hook_configure.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32]
            lib.e05_hook_configure.restype = None
            lib.e05_hook_set_skip_lm_head.argtypes = [ctypes.c_int]
            lib.e05_hook_set_skip_lm_head.restype = None

        self._supports_multimodal = bool(
            hasattr(lib, "lila_asi_eval_embd")
            and hasattr(lib, "lila_asi_build_mrope_grid")
            and hasattr(lib, "lila_asi_session_get_rope_delta")
        )
        if self._supports_multimodal:
            lib.lila_asi_eval_embd.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.lila_asi_eval_embd.restype = ctypes.c_int
            lib.lila_asi_build_mrope_grid.argtypes = [
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_size_t,
            ]
            lib.lila_asi_build_mrope_grid.restype = ctypes.c_uint32
            lib.lila_asi_session_get_rope_delta.argtypes = [ctypes.c_void_p]
            lib.lila_asi_session_get_rope_delta.restype = ctypes.c_int64
            lib.lila_asi_session_set_rope_delta.argtypes = [ctypes.c_void_p, ctypes.c_int64]
            lib.lila_asi_session_set_rope_delta.restype = None

        self._supports_anchor = bool(
            hasattr(lib, "lila_asi_cp_fork")
            and hasattr(lib, "lila_asi_cp_restore")
            and hasattr(lib, "lila_asi_cp_get_info")
            and hasattr(lib, "lila_asi_cp_cow_stats")
        )
        if self._supports_anchor:
            lib.lila_asi_cp_fork.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint64,
                ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            lib.lila_asi_cp_fork.restype = ctypes.c_int
            lib.lila_asi_cp_restore.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            lib.lila_asi_cp_restore.restype = ctypes.c_int
            lib.lila_asi_cp_get_info.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.POINTER(_AsiCheckpointInfo),
            ]
            lib.lila_asi_cp_get_info.restype = ctypes.c_int
            lib.lila_asi_cp_cow_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_AsiCowStats)]
            lib.lila_asi_cp_cow_stats.restype = ctypes.c_int
            if hasattr(lib, "lila_asi_cp_commit"):
                lib.lila_asi_cp_commit.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                lib.lila_asi_cp_commit.restype = ctypes.c_int
            if hasattr(lib, "lila_asi_cp_abandon"):
                lib.lila_asi_cp_abandon.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                lib.lila_asi_cp_abandon.restype = ctypes.c_int

    def _free_e05_locked(self) -> None:
        if getattr(self, "_supports_e05", False) and getattr(self, "_lib", None) is not None:
            try:
                self._lib.e05_hook_set_skip_lm_head(0)
                self._lib.e05_hook_configure(0, None, 0, 0)
            except Exception:
                pass
        if getattr(self, "_e05_grammar", None):
            try:
                self._lib.asi_grammar_free(self._e05_grammar)
            except Exception:
                pass
            self._e05_grammar = None
        if getattr(self, "_e05_vocab_loaded", False):
            try:
                self._lib.asi_grammar_vocab_free(ctypes.byref(self._e05_vocab))
            except Exception:
                pass
            self._e05_vocab_loaded = False
        self._e05_tool_sig = None

    def _ensure_e05_grammar_locked(self, active_tools: Optional[Sequence[Any]] = None) -> bool:
        """Lazily compile/refresh the model-triggered E05 C11 <tool_call> grammar from registered tools."""
        if not getattr(self, "_supports_e05", False):
            return False
        if os.environ.get("LILA_ASI_DISABLE_E05", "").strip() == "1":
            return False
        if "<tool_call>" not in self.tokenizer.vocab or "</tool_call>" not in self.tokenizer.vocab:
            return False

        try:
            if active_tools is not None:
                raw_tools = list(active_tools)
            else:
                from src.harness.tools import Tool, ToolArg, get_registry
                from src.harness.system_tools import register_all_system_tools
                from src.harness.extended_tools import register_extended_tools

                reg = get_registry()
                if not reg.get("file_read"):
                    register_all_system_tools()
                    register_extended_tools()
                raw_tools = list(reg.all() if hasattr(reg, "all") else reg.list_tools())
                existing_names = {t.name for t in raw_tools}
                meta_tools = [
                    Tool(
                        name="discover_capabilities",
                        description="Search tools, skills, and capabilities.",
                        args=[
                            ToolArg("query", "string", "Search query", required=True),
                            ToolArg("kind", "string", "Filter kind", required=False),
                            ToolArg("limit", "int", "Max matches", required=False),
                        ],
                        execute=lambda **_: None,
                    ),
                    Tool(
                        name="activate_capability",
                        description="Activate a capability bundle.",
                        args=[
                            ToolArg("capability_id", "string", "Bundle ID", required=True),
                            ToolArg("reason", "string", "Activation reason", required=False),
                            ToolArg("task_id", "string", "Task ID", required=False),
                            ToolArg("ttl_turns", "int", "TTL turns", required=False),
                        ],
                        execute=lambda **_: None,
                    ),
                    Tool(
                        name="deactivate_capability",
                        description="Deactivate a capability bundle.",
                        args=[
                            ToolArg("capability_id", "string", "Bundle ID", required=True),
                            ToolArg("outcome_summary", "string", "Outcome summary", required=False),
                        ],
                        execute=lambda **_: None,
                    ),
                    Tool(
                        name="request_capability_creation",
                        description="Propose a missing skill or tool.",
                        args=[
                            ToolArg("kind", "string", "Entity kind", required=True),
                            ToolArg("title", "string", "Title", required=True),
                            ToolArg("problem", "string", "Problem statement", required=True),
                            ToolArg("proposed_inputs", "string", "Proposed inputs", required=False),
                        ],
                        execute=lambda **_: None,
                    ),
                    Tool(
                        name="load_tool_schema",
                        description="Load full argument schemas for 1-3 tools.",
                        args=[
                            ToolArg("tools", "string", "Tool names", required=True),
                        ],
                        execute=lambda **_: None,
                    ),
                ]
                for mt in meta_tools:
                    if mt.name not in existing_names:
                        raw_tools.append(mt)
            all_tools = sorted(raw_tools, key=lambda t: t.name)[:512]
        except ImportError:
            return False
        except Exception as exc:
            logger.warning("E05 tool discovery failed: %s", exc)
            return False

        if not all_tools:
            return False

        sig = tuple(
            (
                t.name,
                tuple(
                    (
                        a.name,
                        a.type,
                        bool(getattr(a, "required", True)),
                        tuple(getattr(a, "choices", ()) or ()),
                    )
                    for a in t.args[:16]
                ),
            )
            for t in all_tools
        )
        if getattr(self, "_e05_grammar", None) and getattr(self, "_e05_tool_sig", None) == sig:
            return True

        if not getattr(self, "_e05_vocab_loaded", False):
            self._e05_vocab = _AsiGrammarVocab()
            if self._lib.e05_load_vocab_from_model(self._model, ctypes.byref(self._e05_vocab)) != 0:
                return False
            self._e05_vocab_loaded = True

        if getattr(self, "_e05_grammar", None):
            self._lib.asi_grammar_free(self._e05_grammar)
            self._e05_grammar = None

        c_tools = (_AsiToolSchema * len(all_tools))()
        keepalive: List[bytes] = []
        type_map = {
            "string": 0,
            "path": 0,
            "int": 1,
            "integer": 1,
            "float": 2,
            "number": 2,
            "bool": 3,
            "boolean": 3,
        }
        for i, t in enumerate(all_tools):
            nb = t.name.encode("utf-8")
            keepalive.append(nb)
            c_tools[i].name = nb
            args_slice = t.args[:16]
            c_tools[i].n_args = len(args_slice)
            for j, a in enumerate(args_slice):
                ab = a.name.encode("utf-8")
                keepalive.append(ab)
                c_tools[i].args[j].name = ab
                atype_str = str(a.type).lower()
                raw_choices = getattr(a, "choices", None) or []
                str_choices = [
                    str(c)
                    for c in raw_choices
                    if isinstance(c, str) and c and len(c.encode("utf-8")) < 120
                ]
                if atype_str in ("string", "path") and 0 < len(str_choices) <= 16 and len(str_choices) == len(raw_choices):
                    c_tools[i].args[j].type = 4  # ASI_ARG_ENUM
                    c_tools[i].args[j].n_enum_values = len(str_choices)
                    for k, choice_val in enumerate(str_choices):
                        cb = choice_val.encode("utf-8")
                        keepalive.append(cb)
                        c_tools[i].args[j].enum_values[k] = cb
                else:
                    c_tools[i].args[j].type = type_map.get(atype_str, 0)
                    c_tools[i].args[j].n_enum_values = 0
                c_tools[i].args[j].max_len = 0  # 0 -> unbounded RFC-8259 escaped JSON string
                c_tools[i].args[j].max_digits = 9
                c_tools[i].args[j].optional = 0 if getattr(a, "required", True) else 1

        gcfg = _AsiGrammarConfig(wrap_xml=1, default_max_str=0, default_max_digits=9, sparse_limit=2048)
        g = self._lib.asi_grammar_create(
            ctypes.byref(self._e05_vocab),
            c_tools,
            len(all_tools),
            ctypes.byref(gcfg),
        )
        if not g:
            logger.warning("E05 asi_grammar_create returned NULL for %d tools", len(all_tools))
            return False
        self._e05_keepalive = keepalive
        self._e05_grammar = g
        self._e05_tool_sig = sig
        return True

    def close_vision(self) -> None:
        self._vision_controller = None
        if getattr(self, "_vision_session", None) is not None:
            try:
                self._vision_session.close()
            except Exception:
                pass
            self._vision_session = None
        self._vision_asiv_path = None

    def evict_vision(self) -> int:
        if getattr(self, "_vision_session", None) is None:
            return 0
        return int(self._vision_session.evict())

    @property
    def vision_enabled(self) -> bool:
        return getattr(self, "_vision_session", None) is not None

    @property
    def vision_asiv_path(self) -> Optional[str]:
        return getattr(self, "_vision_asiv_path", None)

    def attach_vision(self, asiv_path: Union[str, Path], *, exec_mode: int = 0) -> None:
        """Lazily open `.asiv` sidecar and bind compatibility to this loaded text model."""
        try:
            from .asi_vision_sdk import (
                AdaptiveVisionController,
                AsiV2ModelConfig as VisionModelCfg,
                AsiVisionSession,
            )
        except ImportError:
            asi_dir = str(Path(__file__).resolve().parents[2] / "engine" / "asi")
            if asi_dir not in sys.path:
                sys.path.insert(0, asi_dir)
            from asi_vision_sdk import (
                AdaptiveVisionController,
                AsiV2ModelConfig as VisionModelCfg,
                AsiVisionSession,
            )

        with self._lock:
            if not self._model:
                raise RuntimeError("AsiRuntime is closed")
            self.close_vision()
            vsess = AsiVisionSession(Path(asiv_path), lib=self._lib, exec_mode=exec_mode)
            try:
                vcfg = VisionModelCfg()
                ctypes.memmove(ctypes.byref(vcfg), ctypes.byref(self._cfg), ctypes.sizeof(VisionModelCfg))
                vsess.check_compat(vcfg)
            except Exception:
                vsess.close()
                raise
            self._vision_session = vsess
            self._vision_controller = AdaptiveVisionController(vsess)
            self._vision_asiv_path = str(asiv_path)

    def encode_image(
        self,
        rgb_u8: object,
        *,
        min_pixels: int = 56 * 56,
        max_pixels: int = 448 * 448,
        n_threads: int = 2,
        use_pairing: Optional[bool] = None,
    ):
        with self._lock:
            if self._vision_session is None:
                raise RuntimeError("No .asiv attached; call attach_vision(asiv_path) first")
            return self._vision_session.encode_rgb(
                rgb_u8,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                n_threads=n_threads,
                use_pairing=use_pairing,
            )

    def create_multimodal_completion(
        self,
        prompt: str,
        images: Sequence[object],
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 32,
        temperature: float = 0.0,
        min_pixels: int = 56 * 56,
        max_pixels: int = 448 * 448,
        n_threads: int = 2,
        adaptive: bool = False,
        use_pairing: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Encode RGB images lazily via `.asiv` and decode natively with 3D M-RoPE."""
        import numpy as np

        if not self._supports_multimodal:
            raise RuntimeError("Loaded liblila_asi.so does not expose multimodal eval_embd ABI")
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiRuntime is closed")
            if self._vision_session is None:
                raise RuntimeError("No .asiv attached; call attach_vision(asiv_path) first")
            vis_start = self.tokenizer.vocab.get("<|vision_start|>")
            vis_end = self.tokenizer.vocab.get("<|vision_end|>")
            im_end = self.tokenizer.vocab.get("<|im_end|>")
            eot = self.tokenizer.vocab.get("<|endoftext|>", self.tokenizer.eos_token_id)
            if vis_start is None or vis_end is None or im_end is None:
                raise RuntimeError("Model tokenizer lacks Qwen3.5 vision/chat special tokens")

            self._lib.lila_asi_session_reset(self._session)
            self._session_token_ids = []
            kv_pos = 0
            rope_pos = 0
            ledger_records: List[Dict[str, Any]] = []

            if system_prompt:
                sys_ids = self._encode_message_tokens("system", system_prompt, close_message=True)
                if sys_ids:
                    sys_arr = (ctypes.c_int32 * len(sys_ids))(*sys_ids)
                    if self._lib.lila_asi_eval_batch(self._session, sys_arr, len(sys_ids), kv_pos, self._logits) != 0:
                        raise RuntimeError("Failed evaluating multimodal system prompt")
                    kv_pos += len(sys_ids)
                    rope_pos += len(sys_ids)

            prefix_ids = self.tokenizer.encode("<|im_start|>user\n", allow_special=("<|im_start|>",))
            arr = (ctypes.c_int32 * len(prefix_ids))(*prefix_ids)
            if self._lib.lila_asi_eval_batch(self._session, arr, len(prefix_ids), kv_pos, self._logits) != 0:
                raise RuntimeError("Failed evaluating multimodal user header")
            kv_pos += len(prefix_ids)
            rope_pos += len(prefix_ids)

            for img in images:
                vs_arr = (ctypes.c_int32 * 1)(vis_start)
                if self._lib.lila_asi_eval_batch(self._session, vs_arr, 1, kv_pos, self._logits) != 0:
                    raise RuntimeError("Failed evaluating <|vision_start|>")
                kv_pos += 1
                rope_pos += 1

                if adaptive and self._vision_controller is not None:
                    res = self._vision_controller.inspect(
                        np.asarray(img, dtype=np.uint8), n_threads=n_threads, use_pairing=use_pairing
                    )
                    enc_list = [res["coarse"]] + list(res.get("rois") or ())
                    for rec in res["ledger"].records:
                        ledger_records.append(
                            {
                                "record_id": rec.record_id,
                                "region_id": rec.region_id,
                                "original_box_xyxy": rec.original_box_xyxy,
                                "status": rec.status.value,
                                "n_tokens": rec.n_tokens,
                                "cached": rec.cached,
                            }
                        )
                else:
                    enc_list = [
                        self._vision_session.encode_rgb(
                            np.asarray(img, dtype=np.uint8),
                            min_pixels=min_pixels,
                            max_pixels=max_pixels,
                            n_threads=n_threads,
                            use_pairing=use_pairing,
                        )
                    ]

                for enc in enc_list:
                    flat_embds = np.ascontiguousarray(enc.embds, dtype=np.float32).reshape(-1)
                    n_tok = int(enc.n_tokens)
                    pos3d = np.zeros(n_tok * 3, dtype=np.uint32)
                    thw_c = (ctypes.c_uint32 * 3)(*enc.grid_thw)
                    next_rope = int(
                        self._lib.lila_asi_build_mrope_grid(
                            rope_pos,
                            thw_c,
                            pos3d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                            pos3d.size,
                        )
                    )
                    rc = self._lib.lila_asi_eval_embd(
                        self._session,
                        flat_embds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        n_tok,
                        kv_pos,
                        pos3d.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                        None,
                        1,  # LILA_ASI_MROPE_INTERLEAVED
                        next_rope,
                        None,
                        None,
                    )
                    if rc != 0:
                        raise RuntimeError(f"lila_asi_eval_embd failed with rc={rc}")
                    kv_pos += n_tok
                    rope_pos = next_rope

                ve_arr = (ctypes.c_int32 * 1)(vis_end)
                if self._lib.lila_asi_eval_batch(self._session, ve_arr, 1, kv_pos, self._logits) != 0:
                    raise RuntimeError("Failed evaluating <|vision_end|>")
                kv_pos += 1
                rope_pos += 1

            tail_str = f"{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            tail_ids = self.tokenizer.encode(
                tail_str,
                allow_special=("<|im_end|>", "<|im_start|>", "<think>", "</think>"),
            )
            tail_arr = (ctypes.c_int32 * len(tail_ids))(*tail_ids)
            if self._lib.lila_asi_eval_batch(self._session, tail_arr, len(tail_ids), kv_pos, self._logits) != 0:
                raise RuntimeError("Failed evaluating multimodal text suffix")
            kv_pos += len(tail_ids)

            rope_delta = int(self._lib.lila_asi_session_get_rope_delta(self._session))
            gen_ids: List[int] = []
            inc_dec = IncrementalBpeUtf8Decoder(self.tokenizer)
            chunks: List[str] = []
            for _ in range(max_tokens):
                nxt = int(
                    self._lib.lila_asi_sample(
                        self._logits,
                        self._cfg.vocab_size,
                        float(temperature),
                        20,
                        0.95,
                        ctypes.byref(self._rng),
                    )
                )
                if nxt < 0 or nxt in (im_end, eot):
                    break
                gen_ids.append(nxt)
                chunks.append(inc_dec.decode_token(nxt, final=False))
                if self._lib.lila_asi_eval_token(self._session, nxt, kv_pos, self._logits) != 0:
                    break
                kv_pos += 1
            chunks.append(inc_dec.flush())
            return {
                "text": "".join(chunks).strip(),
                "tokens": gen_ids,
                "rope_delta": rope_delta,
                "evidence_ledger": ledger_records,
            }

    def close(self) -> None:
        self._cached_prefix_ids = None
        self._cached_prefix_text = None
        self._session_token_ids = []
        self.close_vision()
        self._free_e05_locked()
        if getattr(self, "_session", None):
            self._lib.lila_asi_session_free(self._session)
            self._session = None
        if getattr(self, "_model", None):
            self._lib.lila_asi_close(self._model)
            self._model = None

    def __del__(self):
        self.close()

    def _format_message(self, role: str, content: object) -> str:
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def _format_message_prefix(self, role: str, prefix_content: str) -> str:
        """Format the opening prefix of a message without appending <|im_end|>."""
        return f"<|im_start|>{role}\n{prefix_content}"

    def _encode_message_tokens(self, role: str, content: object, *, close_message: bool = True) -> List[int]:
        """Encode a chat message so special-token parsing applies ONLY to harness-controlled
        template headers/footers (and assistant-emitted `<think>`/`<tool_call>` tags), never to
        untrusted user/tool content.
        """
        if isinstance(content, list):
            content = "\n".join(
                c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        role_str = str(role or "user")
        content_str = str(content or "")
        header_ids = self.tokenizer.encode(
            f"<|im_start|>{role_str}\n",
            allow_special=("<|im_start|>",),
        )
        if role_str == "assistant":
            body_ids = self.tokenizer.encode(
                content_str,
                allow_special=getattr(
                    self.tokenizer,
                    "_assistant_special_tokens",
                    ("<think>", "</think>", "<tool_call>", "</tool_call>"),
                ),
            )
        else:
            body_ids = self.tokenizer.encode(content_str, allow_special=False)
        if not close_message:
            return header_ids + body_ids
        footer_ids = self.tokenizer.encode(
            "<|im_end|>\n",
            allow_special=("<|im_end|>",),
        )
        return header_ids + body_ids + footer_ids

    def _encode_chat_messages(self, messages: List[Dict]) -> List[int]:
        """Encode full chat message list with template-scoped special token isolation."""
        out: List[int] = []
        for m in messages:
            out.extend(
                self._encode_message_tokens(
                    str(m.get("role", "user")),
                    m.get("content", ""),
                    close_message=True,
                )
            )
        out.extend(
            self.tokenizer.encode(
                "<|im_start|>assistant\n",
                allow_special=("<|im_start|>",),
            )
        )
        think_mode = os.environ.get("LILA_THINK_MODE", "auto").strip().lower()
        if think_mode in {"off", "no_think", "nothink", "false", "0", "disabled"}:
            out.extend(
                self.tokenizer.encode(
                    "<think>\n\n</think>\n\n",
                    allow_special=("<think>", "</think>"),
                )
            )
        elif think_mode in {"on", "think", "true", "1", "enabled", "force"}:
            out.extend(
                self.tokenizer.encode(
                    "<think>\n",
                    allow_special=("<think>",),
                )
            )
        return out

    @staticmethod
    def _extract_static_system_prefix_content(msg: Dict) -> Optional[str]:
        """Extract static prefix content from message metadata or dynamic memory boundary."""
        explicit = msg.get("static_prefix")
        if isinstance(explicit, str) and explicit:
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith(explicit) and len(content) > len(explicit):
                return explicit
        content = msg.get("content", "")
        if isinstance(content, str):
            marker = "\n\nTemporary memory notes (from local markdown memory):\n"
            idx = content.find(marker)
            if idx > 0:
                return content[:idx]
        return None

    def _format_chat(self, messages: List[Dict]) -> str:
        parts = [self._format_message(m.get("role", "user"), m.get("content", "")) for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    @staticmethod
    def _fp32_ref_flag() -> int:
        raw = (os.environ.get("LILA_ASI_FP32_REF") or "").strip().lower()
        if raw in ("1", "true", "yes"):
            return 1
        try:
            return 1 if raw and int(raw) != 0 else 0
        except ValueError:
            return 0

    @staticmethod
    def _max_prefix_cache_files() -> int:
        raw = (os.environ.get("LILA_ASI_PREFIX_CACHE_MAX_FILES") or "").strip()
        if not raw:
            return _DEFAULT_PREFIX_CACHE_MAX_FILES
        try:
            return max(1, int(raw))
        except ValueError:
            return _DEFAULT_PREFIX_CACHE_MAX_FILES

    def _compute_prefix_key_hash(
        self,
        sys_ids: Sequence[int],
        sys_text: str = "",
        *,
        tokenizer_version: Optional[int] = None,
    ) -> bytes:
        """Compute the 32-byte SHA-256 cache key for a model, fp32_ref mode, tokenizer_version, text, and token sequence."""
        ids = [int(t) for t in sys_ids]
        fp32_ref = self._fp32_ref_flag()
        tok_ver = (
            int(tokenizer_version)
            if tokenizer_version is not None
            else int(getattr(self.tokenizer, "TOKENIZER_VERSION", 2))
        )
        packed_ids = struct.pack(f"<{len(ids)}i", *ids) if ids else b""
        payload = (
            b"ASIPFX02"
            + bytes(self._model_hdr.identity_hash)
            + struct.pack("<IIII", int(self._model_hdr.arch_id), fp32_ref, tok_ver, len(ids))
            + sys_text.encode("utf-8")
            + packed_ids
        )
        return hashlib.sha256(payload).digest()

    def _record_prefix_hit(self, kind: str) -> None:
        self.last_prefix_cache_hit = kind
        if kind in self.prefix_cache_stats:
            self.prefix_cache_stats[kind] += 1

    @staticmethod
    def _prune_prefix_cache(
        cache_dir: Path,
        max_files: int,
        keep_path: Optional[Path] = None,
    ) -> None:
        """Evict oldest regular .asip snapshot files beyond max_files."""
        if max_files < 1:
            return
        try:
            entries = []
            for p in cache_dir.iterdir():
                if p.suffix != ".asip" or p.is_symlink() or not p.is_file():
                    continue
                try:
                    st = p.stat(follow_symlinks=False)
                except OSError:
                    continue
                is_keep = 1 if (keep_path is not None and p == keep_path) else 0
                entries.append((is_keep, st.st_mtime_ns, p.name, p))
            if len(entries) <= max_files:
                return
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            to_remove = len(entries) - max_files
            for _, _, _, victim in entries[:to_remove]:
                try:
                    victim.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def _eval_and_save_ram_prefix_locked(self, ids: List[int], sys_text: str) -> bool:
        """Reset session, prefill prefix token IDs, and snapshot into session RAM."""
        self._lib.lila_asi_session_reset(self._session)
        self._session_token_ids = []
        sys_arr = (ctypes.c_int32 * len(ids))(*ids)
        if self._lib.lila_asi_eval_batch(self._session, sys_arr, len(ids), 0, self._logits) != 0:
            raise RuntimeError("lila_asi_eval_batch(prefix) failed")
        self._session_token_ids = list(ids)
        if self._lib.lila_asi_session_save_prefix(self._session, len(ids)) == 0:
            self._cached_prefix_ids = list(ids)
            self._cached_prefix_text = sys_text
            return True
        return False

    def _save_prefix_file_locked(self, target: Path, key_arr: ctypes.Array) -> bool:
        """Write current RAM prefix snapshot to target with 0600 permissions, rejecting symlinks."""
        if target.is_symlink():
            return False
        with restrictive_umask():
            rc = self._lib.lila_asi_session_save_prefix_file(
                self._session,
                key_arr,
                str(target).encode("utf-8"),
            )
        if rc != 0:
            return False
        if not target.is_symlink() and target.is_file():
            os.chmod(target, 0o600)
        return True

    def _load_prefix_file_locked(
        self,
        target: Path,
        key_arr: ctypes.Array,
        ids: List[int],
        sys_text: str,
    ) -> Optional[int]:
        """Load and validate a disk prefix snapshot into session RAM and live state."""
        if target.is_symlink() or not target.is_file():
            return None
        out_n = ctypes.c_uint32(0)
        rc = self._lib.lila_asi_session_load_prefix_file(
            self._session,
            key_arr,
            str(target).encode("utf-8"),
            ctypes.byref(out_n),
        )
        if rc != 0 or int(out_n.value) != len(ids):
            return None
        restored_n = ctypes.c_uint32(0)
        self._lib.lila_asi_session_restore_prefix(self._session, ctypes.byref(restored_n))
        self._cached_prefix_ids = list(ids)
        self._cached_prefix_text = sys_text
        self._session_token_ids = list(ids)
        self._record_prefix_hit("disk")
        return int(out_n.value)

    def _try_load_prefix_cache_file(
        self,
        sys_text: str,
        ids: List[int],
    ) -> Tuple[bool, Optional[Path], Optional[ctypes.Array]]:
        """Try loading a cached disk prefix snapshot; return (loaded, cache_path, key_arr)."""
        if self._prefix_cache_dir is None or not self._supports_prefix_file:
            return False, None, None
        try:
            cache_dir = secure_private_directory(self._prefix_cache_dir)
            key_hash = self._compute_prefix_key_hash(ids, sys_text)
            key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
            cache_path = cache_dir / f"prefix_{key_hash[:16].hex()}.asip"
            if self._load_prefix_file_locked(cache_path, key_arr, ids, sys_text) is not None:
                try:
                    os.utime(cache_path, None)
                except OSError:
                    pass
                return True, cache_path, key_arr
            return False, cache_path, key_arr
        except Exception:
            return False, None, None

    def _save_prefix_locked(self, sys_text: str, sys_ids: Sequence[int]) -> None:
        ids = [int(t) for t in sys_ids]
        if not ids:
            return

        loaded, cache_path, key_arr = self._try_load_prefix_cache_file(sys_text, ids)
        if loaded:
            return

        saved_ram = self._eval_and_save_ram_prefix_locked(ids, sys_text)
        self._record_prefix_hit("miss")

        if saved_ram and cache_path is not None and key_arr is not None:
            try:
                if self._save_prefix_file_locked(cache_path, key_arr):
                    self._prune_prefix_cache(
                        cache_path.parent,
                        self._max_prefix_cache_files(),
                        keep_path=cache_path,
                    )
            except Exception:
                pass

    def save_prefix_snapshot(
        self,
        filepath: Union[str, Path],
        token_ids: Sequence[int],
        sys_text: str = "",
    ) -> bool:
        """Evaluate (if needed) and save a prefix snapshot for token_ids + sys_text to filepath."""
        ids = [int(t) for t in token_ids]
        if not ids:
            return False
        target = Path(filepath).expanduser()
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiRuntime is closed")
            if not self._supports_prefix_file or target.is_symlink():
                return False
            try:
                if self._cached_prefix_ids != ids:
                    if not self._eval_and_save_ram_prefix_locked(ids, sys_text):
                        return False
                key_hash = self._compute_prefix_key_hash(ids, sys_text)
                key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
                with restrictive_umask():
                    target.parent.mkdir(parents=True, exist_ok=True)
                return self._save_prefix_file_locked(target, key_arr)
            except Exception:
                return False

    def load_prefix_snapshot(
        self,
        filepath: Union[str, Path],
        token_ids: Sequence[int],
        sys_text: str = "",
    ) -> Optional[int]:
        """Load and validate a prefix snapshot from filepath into the session."""
        ids = [int(t) for t in token_ids]
        if not ids:
            return None
        target = Path(filepath).expanduser()
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiRuntime is closed")
            if not self._supports_prefix_file:
                return None
            try:
                key_hash = self._compute_prefix_key_hash(ids, sys_text)
                key_arr = (ctypes.c_uint8 * 32).from_buffer_copy(key_hash)
                return self._load_prefix_file_locked(target, key_arr, ids, sys_text)
            except Exception:
                return None

    def warm_prefix(
        self,
        system_prompt: str,
        role: str = "system",
        *,
        is_partial_prefix: bool = False,
    ) -> int:
        """Pre-evaluate and snapshot the system prompt prefix so Turn 1 skips system prefill."""
        if not system_prompt:
            return 0
        marker = "\n\nTemporary memory notes (from local markdown memory):\n"
        if not is_partial_prefix and marker in system_prompt:
            idx = system_prompt.find(marker)
            if idx > 0:
                system_prompt = system_prompt[:idx]
                is_partial_prefix = True
        sys_text = (
            self._format_message_prefix(role, system_prompt)
            if is_partial_prefix
            else self._format_message(role, system_prompt)
        )
        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiRuntime is closed")
            if self._cached_prefix_text == sys_text and self._cached_prefix_ids:
                self._record_prefix_hit("ram")
                return len(self._cached_prefix_ids)
            sys_ids = self._encode_message_tokens(
                role,
                system_prompt,
                close_message=not is_partial_prefix,
            )
            if not sys_ids:
                return 0
            if self._cached_prefix_ids == sys_ids:
                self._cached_prefix_text = sys_text
                self._record_prefix_hit("ram")
                return len(sys_ids)
            self._save_prefix_locked(sys_text, sys_ids)
            return len(sys_ids)

    def _stream_tokens(
        self,
        messages: List[Dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        cancel_event: Optional[threading.Event] = None,
        active_tools: Optional[Sequence[Any]] = None,
        max_think_tokens: int = 0,
    ) -> Iterator[Dict]:
        has_system_prefix = len(messages) > 1 and messages[0].get("role") == "system"
        sys_text = ""
        sys_role = "system"
        sys_content = ""
        static_sub: Optional[str] = None
        static_prefix_text = ""
        if has_system_prefix:
            sys_role = messages[0].get("role", "system")
            sys_content = messages[0].get("content", "")
            sys_text = self._format_message(sys_role, sys_content)
            static_sub = self._extract_static_system_prefix_content(messages[0])
            if static_sub:
                static_prefix_text = self._format_message_prefix(sys_role, static_sub)

        with self._lock:
            if not self._session or not self._model:
                raise RuntimeError("AsiRuntime is closed")

            # Fast path: reuse cached system-prompt tokenization when boundary tokens are atomic.
            can_split_sys = (
                has_system_prefix
                and self._cached_prefix_text == sys_text
                and self._cached_prefix_ids
                and "<|im_start|>" in self.tokenizer._special_tokens
                and "<|im_end|>" in self.tokenizer._special_tokens
            )
            if can_split_sys:
                sys_ids = self._cached_prefix_ids
                rest_ids = self._encode_chat_messages(messages[1:])
                prompt_ids = sys_ids + rest_ids
            else:
                prompt_ids = self._encode_chat_messages(messages)
                sys_ids = (
                    self._encode_message_tokens(sys_role, sys_content, close_message=True)
                    if has_system_prefix
                    else []
                )

            if not prompt_ids:
                return

            # Candidate prefix to restore or cache:
            # 1) If a static prefix boundary is present on the system message, check exact
            #    token-boundary match: prompt_ids[:len(static_ids)] == static_ids.
            # 2) Else if the RAM cache already holds a valid token prefix of prompt_ids
            #    (e.g., from an explicit warm_prefix(is_partial_prefix=True) call), reuse it.
            # 3) Otherwise fall back to full system message caching.
            target_prefix_ids: List[int] = []
            target_prefix_text: str = ""

            if static_prefix_text:
                if (
                    self._cached_prefix_text == static_prefix_text
                    and self._cached_prefix_ids
                    and len(prompt_ids) > len(self._cached_prefix_ids)
                    and prompt_ids[: len(self._cached_prefix_ids)] == self._cached_prefix_ids
                ):
                    target_prefix_ids = list(self._cached_prefix_ids)
                    target_prefix_text = static_prefix_text
                else:
                    static_ids = self._encode_message_tokens(
                        sys_role,
                        static_sub or "",
                        close_message=False,
                    )
                    if (
                        static_ids
                        and len(prompt_ids) > len(static_ids)
                        and prompt_ids[: len(static_ids)] == static_ids
                    ):
                        target_prefix_ids = static_ids
                        target_prefix_text = static_prefix_text
            elif (
                has_system_prefix
                and self._cached_prefix_ids
                and len(prompt_ids) > len(self._cached_prefix_ids)
                and prompt_ids[: len(self._cached_prefix_ids)] == self._cached_prefix_ids
            ):
                target_prefix_ids = list(self._cached_prefix_ids)
                target_prefix_text = self._cached_prefix_text or ""

            if (
                not target_prefix_ids
                and sys_ids
                and len(prompt_ids) > len(sys_ids)
                and prompt_ids[: len(sys_ids)] == sys_ids
            ):
                target_prefix_ids = sys_ids
                target_prefix_text = sys_text

            # Automatic system-prompt prefix & multi-round session state continuation (DeltaNet + KV)
            session_ids = getattr(self, "_session_token_ids", None) or []
            prefix_len = 0
            if (
                session_ids
                and 0 < len(session_ids) < len(prompt_ids)
                and prompt_ids[: len(session_ids)] == session_ids
                and len(session_ids) >= len(target_prefix_ids)
            ):
                # Multi-round continuation: session already holds a strict prefix >= target_prefix_ids.
                # Continue directly from len(session_ids) without resetting or rewinding!
                prefix_len = len(session_ids)
                self._record_prefix_hit("ram")
                if "session" in self.prefix_cache_stats:
                    self.prefix_cache_stats["session"] += 1
            elif (
                target_prefix_ids
                and len(prompt_ids) > len(target_prefix_ids)
                and prompt_ids[: len(target_prefix_ids)] == target_prefix_ids
            ):
                n_restored = ctypes.c_uint32(0)
                if (
                    self._cached_prefix_ids == target_prefix_ids
                    and self._lib.lila_asi_session_restore_prefix(self._session, ctypes.byref(n_restored)) == 0
                    and n_restored.value == len(target_prefix_ids)
                ):
                    self._cached_prefix_text = target_prefix_text
                    prefix_len = len(target_prefix_ids)
                    self._session_token_ids = list(target_prefix_ids)
                    self._record_prefix_hit("ram")
                else:
                    self._save_prefix_locked(target_prefix_text, target_prefix_ids)
                    prefix_len = len(target_prefix_ids)
                    self._session_token_ids = list(target_prefix_ids)

            if prefix_len == 0:
                self._lib.lila_asi_session_reset(self._session)
                self._session_token_ids = []

            e05_ready = self._ensure_e05_grammar_locked(active_tools=active_tools)
            V = int(self._cfg.vocab_size)
            H = int(getattr(self._cfg, "hidden_size", 1024) or 1024)
            if e05_ready:
                self._lib.e05_hook_set_skip_lm_head(0)
                self._lib.e05_hook_configure(0, None, V, H)

            rem_ids = prompt_ids[prefix_len:]
            if cancel_event is not None and cancel_event.is_set():
                return
            if rem_ids:
                arr = (ctypes.c_int32 * len(rem_ids))(*rem_ids)
                if self._lib.lila_asi_eval_batch(self._session, arr, len(rem_ids), prefix_len, self._logits) != 0:
                    self._session_token_ids = []
                    raise RuntimeError("lila_asi_eval_batch failed")
                self._session_token_ids = list(prompt_ids)

            pos = len(prompt_ids)
            eos = self.tokenizer.eos_token_id
            im_end_id = self.tokenizer.vocab.get("<|im_end|>", -1)
            think_open_id = self.tokenizer.vocab.get("<think>", -1)
            think_close_id = self.tokenizer.vocab.get("</think>", -1)
            tc_open_id = int(self._e05_vocab.tool_call_open_id) if e05_ready else -1
            grammar = self._e05_grammar if e05_ready else None
            grammar_active = False
            in_think = False
            think_tokens_count = 0
            think_finished = False
            eff_max_think = int(max_think_tokens or 0)
            if eff_max_think <= 0:
                try:
                    eff_max_think = int(os.environ.get("LILA_MAX_THINK_TOKENS", "0") or 0)
                except ValueError:
                    eff_max_think = 0
            newline_ids = self.tokenizer.encode("\n", allow_special=False)
            n_emitted = 0
            inc_decoder = self.tokenizer.create_incremental_decoder()
            recent_tokens: deque[int] = deque(maxlen=64)
            try:
                repeat_penalty = float(os.environ.get("LILA_ASI_REPEAT_PENALTY", "1.12"))
            except ValueError:
                repeat_penalty = 1.12
            skip_penalty_ids = {eos, im_end_id, think_open_id, think_close_id, tc_open_id}
            vocab_cap = int(getattr(self._cfg, "vocab_size", 0) or 0)
            try:
                while n_emitted < max_tokens:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    # Prevent the model from opening a second <think> block after thinking has completed
                    if think_finished and not grammar_active and 0 <= think_open_id < vocab_cap:
                        self._logits[think_open_id] = -1e30
                    if repeat_penalty > 1.0 and not grammar_active and recent_tokens and self._logits is not None and vocab_cap > 0:
                        try:
                            for prev_id in set(recent_tokens):
                                if 0 <= prev_id < vocab_cap and prev_id not in skip_penalty_ids:
                                    logit_val = self._logits[prev_id]
                                    self._logits[prev_id] = (
                                        (logit_val / repeat_penalty) if logit_val > 0.0 else (logit_val * repeat_penalty)
                                    )
                        except (TypeError, IndexError, ValueError):
                            pass
                    # If the in-stream thinking budget is reached, force </think> right here without restarting the stream
                    if in_think and eff_max_think > 0 and think_tokens_count >= eff_max_think and think_close_id >= 0:
                        tok = int(think_close_id)
                    else:
                        tok = int(
                            self._lib.lila_asi_sample(
                                self._logits,
                                self._cfg.vocab_size,
                                float(temperature),
                                int(top_k),
                                float(top_p),
                                ctypes.byref(self._rng),
                            )
                        )
                    if not grammar_active and (tok == eos or tok == im_end_id or tok < 0):
                        break
                    if tok < 0:
                        break
                    recent_tokens.append(tok)
                    piece = inc_decoder.decode_token(tok, final=False)
                    if piece:
                        yield {"choices": [{"delta": {"content": piece}}]}
                    n_emitted += 1
                    if cancel_event is not None and cancel_event.is_set():
                        break

                    if not grammar_active:
                        if tok == think_open_id:
                            in_think = True
                            think_tokens_count = 0
                        elif in_think and tok == think_close_id:
                            was_forced_close = (eff_max_think > 0 and think_tokens_count >= eff_max_think)
                            in_think = False
                            think_finished = True
                            if was_forced_close and newline_ids:
                                if e05_ready:
                                    self._lib.e05_hook_set_skip_lm_head(0)
                                    self._lib.e05_hook_configure(0, None, V, H)
                                if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                    break
                                self._session_token_ids.append(tok)
                                pos += 1
                                nl_tok = int(newline_ids[0])
                                nl_piece = inc_decoder.decode_token(nl_tok, final=False)
                                if nl_piece:
                                    yield {"choices": [{"delta": {"content": nl_piece}}]}
                                n_emitted += 1
                                if self._lib.lila_asi_eval_token(self._session, nl_tok, pos, self._logits) != 0:
                                    break
                                self._session_token_ids.append(nl_tok)
                                pos += 1
                                continue
                        elif in_think:
                            think_tokens_count += 1

                    if not e05_ready:
                        if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                            break
                        self._session_token_ids.append(tok)
                        pos += 1
                        continue

                    if not grammar_active:
                        if not in_think and tok == tc_open_id:
                            grammar_active = True
                            self._lib.asi_grammar_reset(grammar)
                            if self._lib.asi_grammar_accept_token(grammar, tok) != 0:
                                grammar_active = False
                                self._lib.e05_hook_set_skip_lm_head(0)
                                self._lib.e05_hook_configure(0, None, V, H)
                                if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                    break
                                self._session_token_ids.append(tok)
                                pos += 1
                                continue

                            forced_buf = (ctypes.c_int32 * 64)()
                            reason = ctypes.c_int(0)
                            n_forced = int(
                                self._lib.asi_grammar_collect_forced_run(
                                    grammar, forced_buf, 64, ctypes.byref(reason)
                                )
                            )
                            if n_forced > 0:
                                take = min(n_forced, max_tokens - n_emitted)
                                forced_ids = [int(forced_buf[i]) for i in range(take)]
                                for fid in forced_ids:
                                    fpiece = inc_decoder.decode_token(fid, final=False)
                                    if fpiece:
                                        yield {"choices": [{"delta": {"content": fpiece}}]}
                                n_emitted += take
                                if self._lib.asi_grammar_is_complete(grammar):
                                    grammar_active = False
                                    self._lib.asi_grammar_reset(grammar)
                                batch_list = [tok] + forced_ids
                                if n_emitted >= max_tokens:
                                    break
                                if len(batch_list) > 1:
                                    self._lib.e05_hook_set_skip_lm_head(1)
                                    arr = (ctypes.c_int32 * (len(batch_list) - 1))(*batch_list[:-1])
                                    if self._lib.lila_asi_eval_batch(self._session, arr, len(batch_list) - 1, pos, self._logits) != 0:
                                        break
                                    self._session_token_ids.extend(batch_list[:-1])
                                    pos += len(batch_list) - 1
                                self._lib.e05_hook_set_skip_lm_head(0)
                                self._lib.e05_hook_configure(1 if grammar_active else 0, grammar if grammar_active else None, V, H)
                                if self._lib.lila_asi_eval_token(self._session, batch_list[-1], pos, self._logits) != 0:
                                    break
                                self._session_token_ids.append(batch_list[-1])
                                pos += 1
                                continue
                            else:
                                self._lib.e05_hook_set_skip_lm_head(0)
                                self._lib.e05_hook_configure(1, grammar, V, H)
                                if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                    break
                                self._session_token_ids.append(tok)
                                pos += 1
                                continue
                        else:
                            self._lib.e05_hook_set_skip_lm_head(0)
                            self._lib.e05_hook_configure(0, None, V, H)
                            if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                break
                            self._session_token_ids.append(tok)
                            pos += 1
                    else:
                        if self._lib.asi_grammar_accept_token(grammar, tok) != 0:
                            grammar_active = False
                            self._lib.asi_grammar_reset(grammar)
                            self._lib.e05_hook_set_skip_lm_head(0)
                            self._lib.e05_hook_configure(0, None, V, H)
                            if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                break
                            self._session_token_ids.append(tok)
                            pos += 1
                            continue
                        if self._lib.asi_grammar_is_complete(grammar):
                            grammar_active = False
                            self._lib.asi_grammar_reset(grammar)
                            self._lib.e05_hook_set_skip_lm_head(0)
                            self._lib.e05_hook_configure(0, None, V, H)
                            if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                break
                            self._session_token_ids.append(tok)
                            pos += 1
                            continue
                        forced_buf = (ctypes.c_int32 * 64)()
                        reason = ctypes.c_int(0)
                        n_forced = int(
                            self._lib.asi_grammar_collect_forced_run(
                                grammar, forced_buf, 64, ctypes.byref(reason)
                            )
                        )
                        if n_forced > 0:
                            take = min(n_forced, max_tokens - n_emitted)
                            forced_ids = [int(forced_buf[i]) for i in range(take)]
                            for fid in forced_ids:
                                fpiece = inc_decoder.decode_token(fid, final=False)
                                if fpiece:
                                    yield {"choices": [{"delta": {"content": fpiece}}]}
                            n_emitted += take
                            if self._lib.asi_grammar_is_complete(grammar):
                                grammar_active = False
                                self._lib.asi_grammar_reset(grammar)
                            batch_list = [tok] + forced_ids
                            if n_emitted >= max_tokens:
                                break
                            if len(batch_list) > 1:
                                self._lib.e05_hook_set_skip_lm_head(1)
                                arr = (ctypes.c_int32 * (len(batch_list) - 1))(*batch_list[:-1])
                                if self._lib.lila_asi_eval_batch(self._session, arr, len(batch_list) - 1, pos, self._logits) != 0:
                                    break
                                self._session_token_ids.extend(batch_list[:-1])
                                pos += len(batch_list) - 1
                            self._lib.e05_hook_set_skip_lm_head(0)
                            self._lib.e05_hook_configure(1 if grammar_active else 0, grammar if grammar_active else None, V, H)
                            if self._lib.lila_asi_eval_token(self._session, batch_list[-1], pos, self._logits) != 0:
                                break
                            self._session_token_ids.append(batch_list[-1])
                            pos += 1
                        else:
                            self._lib.e05_hook_set_skip_lm_head(0)
                            self._lib.e05_hook_configure(1, grammar, V, H)
                            if self._lib.lila_asi_eval_token(self._session, tok, pos, self._logits) != 0:
                                break
                            self._session_token_ids.append(tok)
                            pos += 1
            finally:
                if e05_ready:
                    self._lib.e05_hook_set_skip_lm_head(0)
                    self._lib.e05_hook_configure(0, None, V, H)
            tail = inc_decoder.flush()
            if tail:
                yield {"choices": [{"delta": {"content": tail}}]}

    def cp_fork(self, label: Optional[str] = None) -> int:
        """Anchor (E06): Fork a Copy-on-Write checkpoint slot of the current session state."""
        with self._lock:
            if not self._session or not self._supports_anchor:
                raise RuntimeError("Anchor state-tree ABI is unavailable on this session")
            n_tokens = len(self._session_token_ids)
            if n_tokens <= 0:
                raise RuntimeError("Cannot fork Anchor checkpoint before evaluating tokens")
            cp_id = ctypes.c_uint32(0)
            lbl_b = label.encode("utf-8") if label else None
            rc = self._lib.lila_asi_cp_fork(
                self._session,
                ctypes.c_uint32(n_tokens),
                self._rng.value,
                lbl_b,
                ctypes.byref(cp_id),
            )
            if rc != 0:
                raise RuntimeError(f"lila_asi_cp_fork failed (rc={rc}; unsupported arch or tree full)")
            if not hasattr(self, "_anchor_cp_ids"):
                self._anchor_cp_ids: List[int] = []
            if int(cp_id.value) not in self._anchor_cp_ids:
                self._anchor_cp_ids.append(int(cp_id.value))
            return int(cp_id.value)

    def cp_restore(self, cp_id: int) -> int:
        """Anchor (E06): Restore session state (KV/SSM/M-RoPE/RNG) to checkpoint cp_id."""
        with self._lock:
            if not self._session or not self._supports_anchor:
                raise RuntimeError("Anchor state-tree ABI is unavailable on this session")
            out_n = ctypes.c_uint32(0)
            out_rng = ctypes.c_uint64(0)
            rc = self._lib.lila_asi_cp_restore(
                self._session,
                ctypes.c_uint32(int(cp_id)),
                ctypes.byref(out_n),
                ctypes.byref(out_rng),
            )
            if rc != 0:
                raise RuntimeError(f"lila_asi_cp_restore failed for cp_id={cp_id} (rc={rc})")
            self._rng = ctypes.c_uint64(int(out_rng.value))
            n_tok = int(out_n.value)
            if len(self._session_token_ids) >= n_tok:
                self._session_token_ids = list(self._session_token_ids[:n_tok])
            else:
                self._session_token_ids = []
            return n_tok

    def cp_list(self) -> List[Dict[str, Any]]:
        """Anchor (E06): Inspect all known live checkpoint slots for this session."""
        with self._lock:
            if not self._session or not self._supports_anchor:
                return []
            status_map = {0: "active", 1: "committed", 2: "abandoned"}
            seen = list(getattr(self, "_anchor_cp_ids", []))
            for probe in range(1, 33):
                if probe not in seen:
                    seen.append(probe)
            out: List[Dict[str, Any]] = []
            for cid in seen:
                info = _AsiCheckpointInfo()
                if self._lib.lila_asi_cp_get_info(self._session, ctypes.c_uint32(cid), ctypes.byref(info)) == 0:
                    out.append(
                        {
                            "cp_id": int(info.cp_id),
                            "parent_cp_id": int(info.parent_cp_id),
                            "rng_state": int(info.rng_state),
                            "branch_id": int(info.prov.branch_id),
                            "parent_branch_id": int(info.prov.parent_branch_id),
                            "state_version": int(info.prov.state_version),
                            "n_tokens": int(info.prov.n_tokens),
                            "status": status_map.get(int(info.prov.status), f"unknown({info.prov.status})"),
                            "label": info.label.decode("utf-8", "replace"),
                        }
                    )
            return out

    def cp_cow_stats(self) -> Dict[str, int]:
        """Anchor (E06): Return Copy-on-Write KV page sharing statistics."""
        with self._lock:
            if not self._session or not self._supports_anchor:
                return {}
            st = _AsiCowStats()
            if self._lib.lila_asi_cp_cow_stats(self._session, ctypes.byref(st)) != 0:
                return {}
            return {
                "live_slots": int(st.live_slots),
                "max_slots": int(st.max_slots),
                "physical_kv_pages": int(st.physical_kv_pages),
                "logical_kv_pages": int(st.logical_kv_pages),
                "shared_kv_pages": int(st.shared_kv_pages),
                "cow_reuses": int(st.cow_reuses),
                "cow_allocs": int(st.cow_allocs),
            }

    @staticmethod
    def _decode_image_url_to_rgb(url: str) -> object:
        import base64
        import io
        from urllib.parse import unquote, urlparse
        import numpy as np
        from PIL import Image

        if url.startswith("data:"):
            if "," not in url:
                raise ValueError("Invalid data URI for image_url")
            _, b64_data = url.split(",", 1)
            raw = base64.b64decode(b64_data)
            with Image.open(io.BytesIO(raw)) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        parsed = urlparse(url)
        if parsed.scheme == "file":
            p = Path(unquote(parsed.path))
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        if not parsed.scheme:
            p = Path(url).expanduser()
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
        raise ValueError(f"Unsupported image_url scheme: {parsed.scheme}")

    def create_chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        stream: bool = False,
        cancel_event: Optional[threading.Event] = None,
        active_tools: Optional[Sequence[Any]] = None,
        max_think_tokens: int = 0,
        **kwargs: object,
    ):
        mm_images: List[object] = []
        mm_system_parts: List[str] = []
        mm_texts: List[str] = []
        for m in messages:
            role = str(m.get("role", "user") or "user")
            content = m.get("content", "")
            msg_texts: List[str] = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        txt = str(part.get("text", ""))
                        if txt:
                            msg_texts.append(txt)
                    elif ptype == "image_url":
                        img_obj = part.get("image_url")
                        url = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj or "")
                        if url:
                            mm_images.append(self._decode_image_url_to_rgb(url))
            elif isinstance(content, str) and content:
                msg_texts.append(content)
            joined_msg = "\n".join(msg_texts).strip()
            if joined_msg:
                if role == "system":
                    mm_system_parts.append(joined_msg)
                else:
                    mm_texts.append(joined_msg)

        if mm_images:
            if self._vision_session is None:
                raise RuntimeError("Multimodal image_url supplied cleanly, but no .asiv vision sidecar is attached")
            mm_res = self.create_multimodal_completion(
                prompt="\n".join(mm_texts).strip(),
                images=mm_images,
                system_prompt="\n\n".join(mm_system_parts) or None,
                max_tokens=max_tokens,
                temperature=temperature,
                n_threads=int(getattr(self, "_n_threads", 2) or 2),
                adaptive=bool(kwargs.get("adaptive", False)),
            )
            text = str(mm_res.get("text", ""))
            if stream:
                def _one_chunk() -> Iterator[Dict]:
                    yield {"choices": [{"delta": {"content": text}}]}
                return _one_chunk()
            return {"choices": [{"message": {"role": "assistant", "content": text}}], "multimodal": mm_res}

        mm_images: List[object] = []
        mm_texts: List[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                part_texts: List[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "text":
                        txt = str(part.get("text", ""))
                        if txt:
                            part_texts.append(txt)
                    elif ptype == "image_url":
                        img_obj = part.get("image_url")
                        url = img_obj.get("url", "") if isinstance(img_obj, dict) else str(img_obj or "")
                        if url:
                            mm_images.append(self._decode_image_url_to_rgb(url))
                if part_texts:
                    mm_texts.append("\n".join(part_texts))
            elif isinstance(content, str) and content:
                mm_texts.append(content)

        if mm_images:
            if self._vision_session is None:
                raise RuntimeError("Multimodal image_url supplied cleanly, but no .asiv vision sidecar is attached")
            mm_res = self.create_multimodal_completion(
                prompt="\n".join(mm_texts).strip(),
                images=mm_images,
                max_tokens=max_tokens,
                temperature=temperature,
                adaptive=bool(kwargs.get("adaptive", False)),
            )
            text = str(mm_res.get("text", ""))
            if stream:
                def _one_chunk() -> Iterator[Dict]:
                    yield {"choices": [{"delta": {"content": text}}]}
                return _one_chunk()
            return {"choices": [{"message": {"role": "assistant", "content": text}}], "multimodal": mm_res}

        if stream:
            return self._stream_tokens(
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                cancel_event=cancel_event,
                active_tools=active_tools,
                max_think_tokens=max_think_tokens,
            )
        text = "".join(
            ev["choices"][0]["delta"]["content"]
            for ev in self._stream_tokens(
                messages,
                max_tokens,
                temperature,
                top_p,
                top_k,
                cancel_event=cancel_event,
                active_tools=active_tools,
                max_think_tokens=max_think_tokens,
            )
        )
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
