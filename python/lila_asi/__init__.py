"""
lila_asi — Standalone Python SDK for Lila ASI v2.6 Cartridges (.asi + .asiv Universal Vision)

Usage:
    from lila_asi import AsiRuntime, convert_gguf_to_asi_v2

    # 1. Pack any GGUF (text LLM -> .asi, vision mmproj -> .asiv automatically):
    # convert_gguf_to_asi_v2("model.gguf", "model.asi")
    # convert_gguf_to_asi_v2("mmproj.gguf")

    # 2. Run inference (with optional .asiv vision attachment):
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
from .gguf_to_asiv import convert_mmproj_to_asiv, is_vision_gguf

try:
    from .asi_vision_sdk import (
        AdaptiveVisionController,
        CoordTransformChain,
        EncodedImage,
        AsiVisionSession,
        VisualEvidenceLedger,
    )
except ImportError:  # numpy/Pillow optional for pure text deployments
    AdaptiveVisionController = None  # type: ignore[assignment]
    CoordTransformChain = None  # type: ignore[assignment]
    EncodedImage = None  # type: ignore[assignment]
    AsiVisionSession = None  # type: ignore[assignment]
    VisualEvidenceLedger = None  # type: ignore[assignment]

__all__ = [
    "ASI_ENGINE_VERSION",
    "ASI_MODE_PRESETS",
    "AdaptiveVisionController",
    "AsiBpeTokenizer",
    "AsiRuntime",
    "AsiVisionSession",
    "CoordTransformChain",
    "EncodedImage",
    "VisualEvidenceLedger",
    "active_asi_label",
    "apply_asi_mode_preset",
    "convert_gguf_to_asi_v2",
    "convert_mmproj_to_asiv",
    "describe_asi_modes",
    "get_asi_mode_preset",
    "is_vision_gguf",
]
