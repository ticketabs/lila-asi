#!/usr/bin/env python3
"""
Convert a vision GGUF (mmproj) into a self-describing ASI v2.5 `.asiv` sidecar.

Reuses engine/asi/gguf_reader.py and engine/asi/asi_v2.py (zero llama_cpp),
matching engine/asi/asi_vision.h (`AsiV2VisionHeader` 192B + `AsiV2WeightsHeader`
16B + `AsiV2TensorEntry` 80B*N + `AsiV2VisionTensorShape` 40B*N + 64B-aligned payloads).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ASI_DIR = Path(__file__).resolve().parent
if str(ASI_DIR) not in sys.path:
    sys.path.insert(0, str(ASI_DIR))

try:
    from .asi_v2 import (  # noqa: E402
        ARCH_IDS,
        FLAG_HAS_VISION,
        SEC_GRAPH,
        SEC_METADATA,
        SEC_RUNTIME_HINTS,
        SEC_VISION,
        TENSOR_LAYOUT_ROW_MAJOR,
        write_asi_v2,
    )
    from .gguf_reader import GgufFile  # noqa: E402
except ImportError:
    from asi_v2 import (  # noqa: E402
        ARCH_IDS,
        FLAG_HAS_VISION,
        SEC_GRAPH,
        SEC_METADATA,
        SEC_RUNTIME_HINTS,
        SEC_VISION,
        TENSOR_LAYOUT_ROW_MAJOR,
        write_asi_v2,
    )
    from gguf_reader import GgufFile  # noqa: E402

ASIV_MAGIC = b"ASIV0001"
ASIV_FORMAT_VERSION = 1
ASIG_MAGIC = b"ASIG0001"
ASIG_FORMAT_VERSION = 1

OP_RESIZE_RGB_NORM_CHW = 1
OP_PATCH_EXTRACT_MERGED = 2
OP_CONV3D_PATCH_PAIR = 3
OP_POS_EMBD_2D_INTERP_ADD = 4
OP_ROPE_2D_VISION_TABLES = 5
OP_LAYERNORM = 6
OP_LINEAR_QUANT_BIAS = 8
OP_ROPE_2D_QKV_APPLY = 9
OP_SDPA_BIDIR_QKV = 10
OP_MERGE_2X2_VIEW = 13

ACT_NONE = 0
ACT_GELU_TANH = 1
ACT_GELU_ERF = 2

VISION_ADAPTER_UNKNOWN = 0
VISION_ADAPTER_QWEN3VL_VIT = 1

VISION_MODE_MULTIMODAL_EMBD = 1
VISION_MODE_INDEPENDENT_PERCEPTION = 2

VISION_OUT_EMBD = 1 << 0

VISION_FLAG_USE_GELU_TANH = 1 << 0
VISION_FLAG_HAS_DEEPSTACK = 1 << 1
VISION_FLAG_TEMPORAL_CONV3D = 1 << 2

REQUIRED_VISION_KEYS = (
    "clip.vision.block_count",
    "clip.vision.embedding_length",
    "clip.vision.attention.head_count",
    "clip.vision.feed_forward_length",
    "clip.vision.patch_size",
    "clip.vision.projection_dim",
)


def is_vision_gguf(gguf: GgufFile) -> bool:
    """Return True if a GGUF file is a standalone CLIP/vision encoder (mmproj)."""
    md = gguf.metadata
    if bool(md.get("clip.has_vision_encoder")):
        return True
    if str(md.get("general.architecture", "")).strip().lower() == "clip":
        return True
    if bool(md.get("clip.projector_type")) and any(k.startswith("clip.vision.") for k in md):
        return True
    return any(t.name.startswith("v.blk.") for t in gguf.tensors)


def _sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.digest()


def extract_vision_profile(mm_gguf: GgufFile) -> Dict[str, Any]:
    """Extract normalized vision hyperparameters + adapter/compatibility metadata."""
    md = mm_gguf.metadata
    missing = [k for k in REQUIRED_VISION_KEYS if k not in md]
    if missing:
        raise ValueError(f"mmproj GGUF missing required keys: {missing}")

    projector_type = str(md.get("clip.projector_type", "") or "").strip()
    is_qwen3vl = projector_type == "qwen3vl_merger"
    if is_qwen3vl and "clip.vision.spatial_merge_size" not in md:
        raise ValueError("qwen3vl_merger mmproj GGUF missing clip.vision.spatial_merge_size")

    patch_size = int(md["clip.vision.patch_size"])
    merge_size = max(1, int(md.get("clip.vision.spatial_merge_size", 1) or 1))
    projection_dim = int(md["clip.vision.projection_dim"])
    deepstack = [bool(v) for v in (md.get("clip.vision.is_deepstack_layers") or [])]
    mean = [float(x) for x in (md.get("clip.vision.image_mean") or [0.5, 0.5, 0.5])]
    std = [float(x) for x in (md.get("clip.vision.image_std") or [0.5, 0.5, 0.5])]
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Expected 3-channel image_mean and image_std")

    has_temporal_conv3d = any(t.name == "v.patch_embd.weight.1" for t in mm_gguf.tensors)
    use_gelu = bool(md.get("clip.use_gelu", True))

    if is_qwen3vl:
        adapter_id = VISION_ADAPTER_QWEN3VL_VIT
        adapter_name = "qwen3vl_vit"
        expected_llm_arch_id = ARCH_IDS["hybrid_qwen35"]
        expected_llm_arch_name = "hybrid_qwen35"
        mrope_sections = [11, 11, 10, 0]
        vision_start_id = 248053
        vision_end_id = 248054
        image_pad_id = 248056
        video_pad_id = 248057
        use_2d_rope = True
        image_size = int(md["clip.vision.image_size"])
    else:
        adapter_id = VISION_ADAPTER_UNKNOWN  # Tier-1 Universal ASI_SEC_GRAPH execution
        adapter_name = projector_type or "generic_vit_graph"
        expected_llm_arch_id = 0
        expected_llm_arch_name = "any"
        mrope_sections = [0, 0, 0, 0]
        vision_start_id = int(md.get("clip.vision.start_token_id", 0) or 0)
        vision_end_id = int(md.get("clip.vision.end_token_id", 0) or 0)
        image_pad_id = int(md.get("clip.vision.image_pad_token_id", 0) or 0)
        video_pad_id = int(md.get("clip.vision.video_pad_token_id", 0) or 0)
        use_2d_rope = projector_type.startswith("qwen") or bool(md.get("clip.vision.use_2d_rope", False))
        image_size = int(md.get("clip.vision.image_size", 384) or 384)

    return {
        "asiv_format_version": ASIV_FORMAT_VERSION,
        "adapter_id": adapter_id,
        "adapter_name": adapter_name,
        "mode": VISION_MODE_MULTIMODAL_EMBD,
        "output_kinds": VISION_OUT_EMBD,
        "expected_llm_arch_id": expected_llm_arch_id,
        "expected_llm_arch_name": expected_llm_arch_name,
        "expected_llm_hidden_size": projection_dim,
        "projector_type": projector_type or "generic_vit",
        "use_gelu_tanh": use_gelu,
        "use_2d_rope": use_2d_rope,
        "has_deepstack": any(deepstack),
        "has_temporal_conv3d": has_temporal_conv3d,
        "patch_size": patch_size,
        "temporal_patch_size": 2 if has_temporal_conv3d else 1,
        "spatial_merge_size": merge_size,
        "resize_multiple": patch_size * merge_size,
        "image_size": image_size,
        "hidden_size": int(md["clip.vision.embedding_length"]),
        "intermediate_size": int(md["clip.vision.feed_forward_length"]),
        "n_heads": int(md["clip.vision.attention.head_count"]),
        "n_layers": int(md["clip.vision.block_count"]),
        "projection_dim": projection_dim,
        "ln_eps": float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-6)),
        "rope_theta": float(md.get("clip.vision.rope.freq_base", 10000.0) or 10000.0),
        "image_mean": mean,
        "image_std": std,
        "mrope_sections": mrope_sections,
        "vision_start_id": vision_start_id,
        "vision_end_id": vision_end_id,
        "image_pad_id": image_pad_id,
        "video_pad_id": video_pad_id,
        "default_max_pixels": 448 * 448,
        "source_gguf": mm_gguf.path.name,
        "n_tensors": len(mm_gguf.tensors),
    }


def pack_vision_section(mm_gguf: GgufFile, profile: Dict[str, Any], fingerprint: bytes) -> bytes:
    """
    Pack ASI_SEC_VISION payload matching asi_vision.h byte-for-byte:
      [AsiV2VisionHeader (192B)]
      [AsiV2WeightsHeader (16B)]
      [AsiV2TensorEntry (80B * n_tensors)]
      [AsiV2VisionTensorShape (40B * n_tensors)]
      [0-padding to 64B alignment]
      [64B-aligned tensor payloads]
    """
    n_tensors = len(mm_gguf.tensors)
    flags = 0
    if profile["use_gelu_tanh"]:
        flags |= VISION_FLAG_USE_GELU_TANH
    if profile["has_deepstack"]:
        flags |= VISION_FLAG_HAS_DEEPSTACK
    if profile["has_temporal_conv3d"]:
        flags |= VISION_FLAG_TEMPORAL_CONV3D

    mean = profile["image_mean"]
    std = profile["image_std"]
    mrope = profile["mrope_sections"]

    vheader = struct.pack(
        "<8s15I2f3f3f4I7I4I32s",
        ASIV_MAGIC,
        ASIV_FORMAT_VERSION,
        profile["adapter_id"],
        profile["mode"],
        profile["output_kinds"],
        profile["expected_llm_arch_id"],
        profile["expected_llm_hidden_size"],
        profile["patch_size"],
        profile["temporal_patch_size"],
        profile["spatial_merge_size"],
        profile["image_size"],
        profile["hidden_size"],
        profile["intermediate_size"],
        profile["n_heads"],
        profile["n_layers"],
        profile["projection_dim"],
        profile["ln_eps"],
        profile["rope_theta"],
        mean[0], mean[1], mean[2],
        std[0], std[1], std[2],
        mrope[0], mrope[1], mrope[2], mrope[3],
        profile["vision_start_id"],
        profile["vision_end_id"],
        profile["image_pad_id"],
        profile["video_pad_id"],
        profile["default_max_pixels"],
        flags,
        n_tensors,
        0, 0, 0, 0,
        fingerprint,
    )
    assert len(vheader) == 192

    # AsiV2WeightsHeader.data_offset is relative to start of weights sub-block (right after vheader)
    dir_bytes = 16 + n_tensors * 80 + n_tensors * 40
    weights_data_offset = (dir_bytes + 63) & ~63

    tensor_entries: List[bytes] = []
    shape_entries: List[bytes] = []
    blobs: List[bytes] = []
    cur_data_off = 0

    with mm_gguf.path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            for t in mm_gguf.tensors:
                dims = tuple(int(x) for x in t.dims)
                ndim = len(dims)
                if ndim < 1 or ndim > 4:
                    raise ValueError(f"Tensor {t.name} has unsupported ndim={ndim}")
                d0 = dims[0]
                d1 = 1
                for extra in dims[1:]:
                    d1 *= extra
                raw = mm_gguf.read_tensor_bytes(mm, t)

                name_b = t.name.encode("utf-8")[:39]
                name_padded = name_b + b"\x00" * (40 - len(name_b))
                tensor_entries.append(
                    struct.pack(
                        "<40sIIQQQQ",
                        name_padded,
                        int(t.ggml_type),
                        TENSOR_LAYOUT_ROW_MAJOR,
                        d0,
                        d1,
                        cur_data_off,
                        len(raw),
                    )
                )
                raw4 = list(dims) + [1] * (4 - ndim)
                shape_entries.append(struct.pack("<II4Q", ndim, 0, raw4[0], raw4[1], raw4[2], raw4[3]))

                aligned_len = (len(raw) + 63) & ~63
                blobs.append(raw + b"\x00" * (aligned_len - len(raw)))
                cur_data_off += aligned_len

    wheader = struct.pack("<IIQ", n_tensors, 0, weights_data_offset)
    pad = b"\x00" * (weights_data_offset - dir_bytes)
    return vheader + wheader + b"".join(tensor_entries) + b"".join(shape_entries) + pad + b"".join(blobs)


def _pack_graph_node(
    op_code: int,
    *,
    flags: int = 0,
    in_slots: Tuple[int, int, int, int] = (-1, -1, -1, -1),
    out_slot: int = -1,
    weight_idx: Tuple[int, int, int, int] = (-1, -1, -1, -1),
    i_params: Tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
    f_params: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    stage_name: str = "",
) -> bytes:
    sname = stage_name.encode("utf-8")[:15]
    sname_pad = sname + b"\x00" * (16 - len(sname))
    raw = struct.pack(
        "<2I14i4f16s",
        op_code,
        flags,
        in_slots[0], in_slots[1], in_slots[2], in_slots[3],
        out_slot,
        weight_idx[0], weight_idx[1], weight_idx[2], weight_idx[3],
        i_params[0], i_params[1], i_params[2], i_params[3], i_params[4],
        f_params[0], f_params[1], f_params[2], f_params[3],
        sname_pad,
    )
    assert len(raw) == 96
    return raw


def build_vision_graph_section(mm_gguf: GgufFile, profile: Dict[str, Any]) -> bytes:
    """Export declarative ASIG0001 computation graph for any supported ViT + projector GGUF."""
    tidx: Dict[str, int] = {t.name: i for i, t in enumerate(mm_gguf.tensors)}
    act_gelu = ACT_GELU_TANH if profile["use_gelu_tanh"] else ACT_GELU_ERF
    use_rope = bool(profile.get("use_2d_rope", True))

    S_IMG = 0
    S_PATCH = 1
    S_X = 2
    S_XN = 3
    S_QKV = 4
    S_ATTN = 5
    S_FFN = 6
    S_COS = 7
    S_SIN = 8
    S_FINAL = 9
    N_SLOTS = 10

    nodes: List[bytes] = []
    # 1. Preprocess & patch extract
    nodes.append(_pack_graph_node(OP_RESIZE_RGB_NORM_CHW, out_slot=S_IMG))
    nodes.append(_pack_graph_node(OP_PATCH_EXTRACT_MERGED, in_slots=(S_IMG, -1, -1, -1), out_slot=S_PATCH))
    # 2. Patch Conv3D/Conv2D embed
    nodes.append(
        _pack_graph_node(
            OP_CONV3D_PATCH_PAIR,
            in_slots=(S_PATCH, -1, -1, -1),
            out_slot=S_X,
            weight_idx=(
                tidx["v.patch_embd.weight"],
                tidx.get("v.patch_embd.weight.1", -1),
                tidx.get("v.patch_embd.bias", -1),
                -1,
            ),
            stage_name="patch_embd",
        )
    )
    # 3. Optional 2D Position embedding interp + optional 2D Vision RoPE tables
    if "v.position_embd.weight" in tidx:
        nodes.append(
            _pack_graph_node(
                OP_POS_EMBD_2D_INTERP_ADD,
                in_slots=(S_X, -1, -1, -1),
                out_slot=S_X,
                weight_idx=(tidx["v.position_embd.weight"], -1, -1, -1),
                stage_name="pos_added",
            )
        )
    if use_rope:
        nodes.append(
            _pack_graph_node(
                OP_ROPE_2D_VISION_TABLES,
                in_slots=(S_SIN, -1, -1, -1),
                out_slot=S_COS,
                f_params=(float(profile["rope_theta"]), 0.0, 0.0, 0.0),
            )
        )
    # 4. ViT layers
    ln_eps = float(profile["ln_eps"])
    for l in range(int(profile["n_layers"])):
        p = f"v.blk.{l}."
        nodes.append(
            _pack_graph_node(
                OP_LAYERNORM,
                in_slots=(S_X, -1, -1, -1),
                out_slot=S_XN,
                weight_idx=(tidx[p + "ln1.weight"], tidx.get(p + "ln1.bias", -1), -1, -1),
                f_params=(ln_eps, 0.0, 0.0, 0.0),
            )
        )
        nodes.append(
            _pack_graph_node(
                OP_LINEAR_QUANT_BIAS,
                in_slots=(S_XN, -1, -1, -1),
                out_slot=S_QKV,
                weight_idx=(tidx[p + "attn_qkv.weight"], tidx.get(p + "attn_qkv.bias", -1), -1, -1),
                i_params=(ACT_NONE, 0, 0, 0, 0),
            )
        )
        if use_rope:
            nodes.append(
                _pack_graph_node(
                    OP_ROPE_2D_QKV_APPLY,
                    in_slots=(S_QKV, S_COS, S_SIN, -1),
                    out_slot=S_QKV,
                )
            )
        nodes.append(
            _pack_graph_node(
                OP_SDPA_BIDIR_QKV,
                in_slots=(S_QKV, -1, -1, -1),
                out_slot=S_ATTN,
            )
        )
        nodes.append(
            _pack_graph_node(
                OP_LINEAR_QUANT_BIAS,
                in_slots=(S_ATTN, S_X, -1, -1),
                out_slot=S_X,
                weight_idx=(tidx[p + "attn_out.weight"], tidx.get(p + "attn_out.bias", -1), -1, -1),
                i_params=(ACT_NONE, 1, 0, 0, 0),
            )
        )
        nodes.append(
            _pack_graph_node(
                OP_LAYERNORM,
                in_slots=(S_X, -1, -1, -1),
                out_slot=S_XN,
                weight_idx=(tidx[p + "ln2.weight"], tidx.get(p + "ln2.bias", -1), -1, -1),
                f_params=(ln_eps, 0.0, 0.0, 0.0),
            )
        )
        nodes.append(
            _pack_graph_node(
                OP_LINEAR_QUANT_BIAS,
                in_slots=(S_XN, -1, -1, -1),
                out_slot=S_FFN,
                weight_idx=(tidx[p + "ffn_up.weight"], tidx.get(p + "ffn_up.bias", -1), -1, -1),
                i_params=(act_gelu, 0, 0, 0, 0),
            )
        )
        nodes.append(
            _pack_graph_node(
                OP_LINEAR_QUANT_BIAS,
                in_slots=(S_FFN, S_X, -1, -1),
                out_slot=S_X,
                weight_idx=(tidx[p + "ffn_down.weight"], tidx.get(p + "ffn_down.bias", -1), -1, -1),
                i_params=(ACT_NONE, 1, 0, 0, 0),
                stage_name=f"blk_{l}",
            )
        )
    # 5. Optional Post-LN + optional spatial merge + Projector MLP
    cur_slot = S_X
    if "v.post_ln.weight" in tidx:
        nodes.append(
            _pack_graph_node(
                OP_LAYERNORM,
                in_slots=(S_X, -1, -1, -1),
                out_slot=S_XN,
                weight_idx=(tidx["v.post_ln.weight"], tidx.get("v.post_ln.bias", -1), -1, -1),
                f_params=(ln_eps, 0.0, 0.0, 0.0),
                stage_name="post_ln",
            )
        )
        cur_slot = S_XN
    merge_factor = int(profile["spatial_merge_size"]) ** 2
    if merge_factor > 1:
        nodes.append(
            _pack_graph_node(
                OP_MERGE_2X2_VIEW,
                in_slots=(cur_slot, -1, -1, -1),
                out_slot=cur_slot,
                i_params=(merge_factor, 0, 0, 0, 0),
            )
        )
    mm0_w = tidx.get("mm.0.weight", -1)
    mm2_w = tidx.get("mm.2.weight", -1)
    if mm0_w < 0:
        raise ValueError("mmproj GGUF missing projector weight mm.0.weight")
    has_mm2 = mm2_w >= 0
    nodes.append(
        _pack_graph_node(
            OP_LINEAR_QUANT_BIAS,
            in_slots=(cur_slot, -1, -1, -1),
            out_slot=S_FFN if has_mm2 else S_FINAL,
            weight_idx=(mm0_w, tidx.get("mm.0.bias", -1), -1, -1),
            i_params=(act_gelu if has_mm2 else ACT_NONE, 0, 1, 0, 0),
            stage_name="" if has_mm2 else "final_embds",
        )
    )
    if has_mm2:
        nodes.append(
            _pack_graph_node(
                OP_LINEAR_QUANT_BIAS,
                in_slots=(S_FFN, -1, -1, -1),
                out_slot=S_FINAL,
                weight_idx=(mm2_w, tidx.get("mm.2.bias", -1), -1, -1),
                i_params=(ACT_NONE, 0, 1, 0, 0),
                stage_name="final_embds",
            )
        )

    ghdr = struct.pack(
        "<8s14I",
        ASIG_MAGIC,
        ASIG_FORMAT_VERSION,
        N_SLOTS,
        len(nodes),
        S_IMG,
        S_FINAL,
        0,
        1,
        0, 0, 0, 0, 0, 0, 0,
    )
    assert len(ghdr) == 64
    return ghdr + b"".join(nodes)


build_qwen3vl_graph_section = build_vision_graph_section


def convert_mmproj_to_asiv(
    mmproj_gguf_path: Path,
    out_asiv_path: Path,
    *,
    gguf: Optional[GgufFile] = None,
) -> Path:
    """Convert an mmproj GGUF into a self-describing `.asiv` sidecar cartridge."""
    mm_gguf = gguf if gguf is not None else GgufFile(mmproj_gguf_path, skip_large_arrays=False)
    profile = extract_vision_profile(mm_gguf)
    fingerprint = _sha256_file(mmproj_gguf_path)
    vision_sec = pack_vision_section(mm_gguf, profile, fingerprint)
    graph_sec = build_vision_graph_section(mm_gguf, profile)

    metadata_json = dict(profile)
    metadata_json["source_fingerprint_sha256"] = fingerprint.hex()
    metadata_json["has_universal_graph"] = True
    runtime_hints = {
        "mrope_sections": profile["mrope_sections"],
        "vit_rope_theta": profile["rope_theta"],
        "default_max_pixels": profile["default_max_pixels"],
        "vision_start_id": profile["vision_start_id"],
        "vision_end_id": profile["vision_end_id"],
        "image_pad_id": profile["image_pad_id"],
        "video_pad_id": profile["video_pad_id"],
    }

    sections = [
        (SEC_VISION, 0, vision_sec),
        (SEC_GRAPH, 0, graph_sec),
        (SEC_METADATA, 0, json.dumps(metadata_json, separators=(",", ":")).encode("utf-8")),
        (SEC_RUNTIME_HINTS, 0, json.dumps(runtime_hints, separators=(",", ":")).encode("utf-8")),
    ]
    return write_asi_v2(
        out_asiv_path,
        arch_id=profile["expected_llm_arch_id"],
        flags=FLAG_HAS_VISION,
        sections=sections,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert mmproj GGUF to ASI v2.5 .asiv vision sidecar")
    ap.add_argument("mmproj_gguf", type=Path, help="Input mmproj .gguf path")
    ap.add_argument("output_asiv", type=Path, help="Output .asiv sidecar path")
    args = ap.parse_args()

    out = convert_mmproj_to_asiv(args.mmproj_gguf, args.output_asiv)
    print(f"[gguf_to_asiv] Converted {args.mmproj_gguf.name} -> {out} ({out.stat().st_size / (1024 * 1024):.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
