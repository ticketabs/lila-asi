"""
lila_asi — Standalone Zero-Dependency Python SDK for Lila ASI v2.4 Cartridges (.asi)

Usage:
    from lila_asi import AsiRuntime, convert_gguf_to_asi_v2

    # 1. Pack a GGUF model into a page-aligned, CRC-64 verified .asi cartridge:
    # convert_gguf_to_asi_v2(Path("model.gguf"), Path("model.asi"))

    # 2. Run inference with Sieve lm_head + E05 Grammar + E06 State Tree:
    llm = AsiRuntime("model.asi", n_ctx=8192, n_threads=2)
    resp = llm.create_chat_completion([{"role": "user", "content": "Hello!"}])
    print(resp["choices"][0]["message"]["content"])
"""

from .asi_backend import (
    ASI_ENGINE_VERSION,
    ASI_MODE_PRESETS,
    AsiBpeTokenizer,
    AsiRuntime,
    active_asi_label,
    apply_asi_mode_preset,
    describe_asi_modes,
    get_asi_mode_preset,
)
from .gguf_to_asi_v2 import convert_gguf_to_asi_v2

__all__ = [
    "ASI_ENGINE_VERSION",
    "ASI_MODE_PRESETS",
    "AsiBpeTokenizer",
    "AsiRuntime",
    "active_asi_label",
    "apply_asi_mode_preset",
    "convert_gguf_to_asi_v2",
    "describe_asi_modes",
    "get_asi_mode_preset",
]
