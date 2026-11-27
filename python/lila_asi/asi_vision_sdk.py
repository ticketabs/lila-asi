#!/usr/bin/env python3
"""
ASI Standalone Vision SDK, Coordinate Provenance, Visual Evidence Ledger,
and Bounded Adaptive Active Inspection Controller (Milestone V9).

Strictly standalone inside `engine/asi/` — zero imports from Harness, Core,
vTalks, or Sensa.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def _default_asi_lib() -> Path:
    return Path(__file__).resolve().parent / "liblila_asi.so"


class AsiV2ModelConfig(ctypes.Structure):
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


class AsiV2VisionHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_char * 8),
        ("format_version", ctypes.c_uint32),
        ("adapter_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("output_kinds", ctypes.c_uint32),
        ("expected_llm_arch_id", ctypes.c_uint32),
        ("expected_llm_hidden_size", ctypes.c_uint32),
        ("patch_size", ctypes.c_uint32),
        ("temporal_patch_size", ctypes.c_uint32),
        ("spatial_merge_size", ctypes.c_uint32),
        ("image_size", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("intermediate_size", ctypes.c_uint32),
        ("n_heads", ctypes.c_uint32),
        ("n_layers", ctypes.c_uint32),
        ("projection_dim", ctypes.c_uint32),
        ("ln_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("image_mean", ctypes.c_float * 3),
        ("image_std", ctypes.c_float * 3),
        ("mrope_sections", ctypes.c_uint32 * 4),
        ("vision_start_id", ctypes.c_uint32),
        ("vision_end_id", ctypes.c_uint32),
        ("image_pad_id", ctypes.c_uint32),
        ("video_pad_id", ctypes.c_uint32),
        ("default_max_pixels", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("n_tensor_shapes", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
        ("source_fingerprint", ctypes.c_uint8 * 32),
    ]


class AsiPairingHeader(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_char * 8),
        ("format_version", ctypes.c_uint32),
        ("state", ctypes.c_uint32),
        ("bridge_kind", ctypes.c_uint32),
        ("act_type", ctypes.c_uint32),
        ("in_dim", ctypes.c_uint32),
        ("hidden_dim", ctypes.c_uint32),
        ("out_dim", ctypes.c_uint32),
        ("target_llm_arch_id", ctypes.c_uint32),
        ("target_mrope_sections", ctypes.c_uint32 * 4),
        ("val_passed", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("val_cosine", ctypes.c_float),
        ("val_loss", ctypes.c_float),
        ("train_steps", ctypes.c_uint32),
        ("n_tensors", ctypes.c_uint32),
        ("weights_offset", ctypes.c_uint64),
        ("source_vision_sha256", ctypes.c_uint8 * 32),
        ("target_llm_sha256", ctypes.c_uint8 * 32),
        ("reserved", ctypes.c_uint32 * 10),
    ]


class AsiCoordTransform(ctypes.Structure):
    _fields_ = [
        ("src_width", ctypes.c_uint32),
        ("src_height", ctypes.c_uint32),
        ("crop_x", ctypes.c_uint32),
        ("crop_y", ctypes.c_uint32),
        ("crop_w", ctypes.c_uint32),
        ("crop_h", ctypes.c_uint32),
        ("dst_width", ctypes.c_uint32),
        ("dst_height", ctypes.c_uint32),
        ("scale_x", ctypes.c_float),
        ("scale_y", ctypes.c_float),
        ("rotation_deg", ctypes.c_float),
        ("frame_index", ctypes.c_uint32),
        ("timestamp_ms", ctypes.c_uint64),
        ("source_sha256", ctypes.c_uint8 * 32),
    ]


class AsiImageView(ctypes.Structure):
    _fields_ = [
        ("rgb_u8", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("stride_bytes", ctypes.c_uint32),
        ("provenance", AsiCoordTransform),
    ]


class AsiVisionPlan(ctypes.Structure):
    _fields_ = [
        ("resized_w", ctypes.c_uint32),
        ("resized_h", ctypes.c_uint32),
        ("grid_thw", ctypes.c_uint32 * 3),
        ("n_patches", ctypes.c_uint32),
        ("n_tokens", ctypes.c_uint32),
        ("est_scratch_bytes", ctypes.c_size_t),
    ]


assert ctypes.sizeof(AsiV2VisionHeader) == 192
assert ctypes.sizeof(AsiPairingHeader) == 192
assert ctypes.sizeof(AsiCoordTransform) == 88


@dataclass(frozen=True)
class CoordTransformHop:
    kind: str  # "crop" or "resize"
    src_w: float
    src_h: float
    dst_w: float
    dst_h: float
    crop_x: float = 0.0
    crop_y: float = 0.0


class CoordTransformChain:
    """Reversible chain of crop/resize operations preserving exact original pixel coordinates."""

    def __init__(self, orig_w: int, orig_h: int) -> None:
        self.orig_w = int(orig_w)
        self.orig_h = int(orig_h)
        self.hops: List[CoordTransformHop] = []

    def add_crop(self, x: int, y: int, w: int, h: int, cur_w: int, cur_h: int) -> "CoordTransformChain":
        self.hops.append(
            CoordTransformHop(
                kind="crop",
                src_w=float(cur_w),
                src_h=float(cur_h),
                dst_w=float(w),
                dst_h=float(h),
                crop_x=float(x),
                crop_y=float(y),
            )
        )
        return self

    def add_resize(self, src_w: int, src_h: int, dst_w: int, dst_h: int) -> "CoordTransformChain":
        self.hops.append(
            CoordTransformHop(
                kind="resize",
                src_w=float(src_w),
                src_h=float(src_h),
                dst_w=float(dst_w),
                dst_h=float(dst_h),
            )
        )
        return self

    def forward_xy(self, orig_x: float, orig_y: float) -> Tuple[float, float]:
        x, y = float(orig_x), float(orig_y)
        for hop in self.hops:
            if hop.kind == "crop":
                x -= hop.crop_x
                y -= hop.crop_y
            elif hop.kind == "resize":
                x *= hop.dst_w / hop.src_w
                y *= hop.dst_h / hop.src_h
        return x, y

    def to_original_xy(self, dst_x: float, dst_y: float) -> Tuple[float, float]:
        x, y = float(dst_x), float(dst_y)
        for hop in reversed(self.hops):
            if hop.kind == "resize":
                x *= hop.src_w / hop.dst_w
                y *= hop.src_h / hop.dst_h
            elif hop.kind == "crop":
                x += hop.crop_x
                y += hop.crop_y
        return x, y

    def to_original_box(
        self, x0: float, y0: float, x1: float, y1: float
    ) -> Tuple[float, float, float, float]:
        ox0, oy0 = self.to_original_xy(x0, y0)
        ox1, oy1 = self.to_original_xy(x1, y1)
        return (min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1))


class EvidenceStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    PREDICTED = "predicted"
    UNVERIFIED = "unverified"


@dataclass
class EvidenceRecord:
    record_id: str
    source_image_sha256: str
    region_id: str
    original_box_xyxy: Tuple[float, float, float, float]
    adapter_id: int
    confidence: float
    status: EvidenceStatus
    summary: str
    n_tokens: int
    grid_thw: Tuple[int, int, int]
    cached: bool = False


class VisualEvidenceLedger:
    """Append-only provenance ledger separating observed pixel evidence from guesses."""

    def __init__(self) -> None:
        self.records: List[EvidenceRecord] = []

    def add_record(
        self,
        *,
        source_image_sha256: str,
        region_id: str,
        original_box_xyxy: Tuple[float, float, float, float],
        adapter_id: int,
        confidence: float,
        status: EvidenceStatus,
        summary: str,
        n_tokens: int,
        grid_thw: Tuple[int, int, int],
        cached: bool = False,
    ) -> EvidenceRecord:
        rec = EvidenceRecord(
            record_id=f"ev_{len(self.records):04d}",
            source_image_sha256=source_image_sha256,
            region_id=region_id,
            original_box_xyxy=tuple(float(v) for v in original_box_xyxy),  # type: ignore[arg-type]
            adapter_id=int(adapter_id),
            confidence=float(confidence),
            status=EvidenceStatus(status),
            summary=str(summary),
            n_tokens=int(n_tokens),
            grid_thw=(int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])),
            cached=bool(cached),
        )
        self.records.append(rec)
        return rec

    def observed_records(self) -> List[EvidenceRecord]:
        return [r for r in self.records if r.status == EvidenceStatus.OBSERVED]

    def has_grounded_evidence(self, region_id: Optional[str] = None) -> bool:
        for r in self.records:
            if r.status == EvidenceStatus.OBSERVED and (region_id is None or r.region_id == region_id):
                return True
        return False

    def to_json(self) -> str:
        payload = []
        for r in self.records:
            d = asdict(r)
            d["status"] = r.status.value
            payload.append(d)
        return json.dumps(payload, indent=2)


@dataclass
class EncodedImage:
    embds: np.ndarray
    grid_thw: Tuple[int, int, int]
    n_tokens: int
    resized_w: int
    resized_h: int
    source_sha256: str
    adapter_id: int
    paired: bool


def _bind_vision_abi(lib: ctypes.CDLL) -> None:
    lib.lila_asi_vision_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.lila_asi_vision_open.restype = ctypes.c_void_p
    lib.lila_asi_vision_close.argtypes = [ctypes.c_void_p]
    lib.lila_asi_vision_close.restype = None
    lib.lila_asi_vision_evict.argtypes = [ctypes.c_void_p]
    lib.lila_asi_vision_evict.restype = ctypes.c_int
    lib.lila_asi_vision_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(AsiV2VisionHeader)]
    lib.lila_asi_vision_info.restype = ctypes.c_int
    lib.lila_asi_vision_set_exec_mode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.lila_asi_vision_set_exec_mode.restype = ctypes.c_int
    if hasattr(lib, "lila_asi_vision_has_graph"):
        lib.lila_asi_vision_has_graph.argtypes = [ctypes.c_void_p]
        lib.lila_asi_vision_has_graph.restype = ctypes.c_int
    lib.lila_asi_vision_has_pairing.argtypes = [ctypes.c_void_p]
    lib.lila_asi_vision_has_pairing.restype = ctypes.c_int
    lib.lila_asi_vision_pairing_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(AsiPairingHeader)]
    lib.lila_asi_vision_pairing_info.restype = ctypes.c_int
    lib.lila_asi_vision_check_compat.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(AsiV2ModelConfig), ctypes.c_char_p, ctypes.c_size_t
    ]
    lib.lila_asi_vision_check_compat.restype = ctypes.c_int
    lib.lila_asi_vision_plan_grid.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(AsiVisionPlan)
    ]
    lib.lila_asi_vision_plan_grid.restype = ctypes.c_int
    lib.lila_asi_vision_encode_rgb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AsiImageView),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(AsiCoordTransform),
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.lila_asi_vision_encode_rgb.restype = ctypes.c_int
    lib.lila_asi_vision_encode_paired_rgb.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AsiImageView),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(AsiCoordTransform),
    ]
    lib.lila_asi_vision_encode_paired_rgb.restype = ctypes.c_int


class AsiVisionSession:
    """Managed ctypes handle over a lazy-loaded `.asiv` sidecar."""

    def __init__(
        self,
        asiv_path: Path,
        *,
        lib: Optional[ctypes.CDLL] = None,
        lib_path: Optional[Path] = None,
        exec_mode: int = 0,
    ) -> None:
        self._lib = lib if lib is not None else ctypes.CDLL(str(lib_path or _default_asi_lib()))
        _bind_vision_abi(self._lib)
        err = ctypes.create_string_buffer(256)
        self._vmodel = self._lib.lila_asi_vision_open(str(asiv_path).encode("utf-8"), err, 256)
        if not self._vmodel:
            raise RuntimeError(f"lila_asi_vision_open failed: {err.value.decode('utf-8', 'replace')}")
        if exec_mode != 0:
            self._lib.lila_asi_vision_set_exec_mode(self._vmodel, int(exec_mode))
        self.header = AsiV2VisionHeader()
        if self._lib.lila_asi_vision_info(self._vmodel, ctypes.byref(self.header)) != 0:
            self.close()
            raise RuntimeError("lila_asi_vision_info failed")

    def close(self) -> None:
        if getattr(self, "_vmodel", None):
            self._lib.lila_asi_vision_close(self._vmodel)
            self._vmodel = None

    def __del__(self) -> None:
        self.close()

    def evict(self) -> int:
        if not self._vmodel:
            return 0
        return int(self._lib.lila_asi_vision_evict(self._vmodel))

    def has_graph(self) -> bool:
        return bool(
            self._vmodel
            and hasattr(self._lib, "lila_asi_vision_has_graph")
            and self._lib.lila_asi_vision_has_graph(self._vmodel) == 1
        )

    def has_pairing(self) -> bool:
        return bool(self._vmodel and self._lib.lila_asi_vision_has_pairing(self._vmodel) == 1)

    def pairing_info(self) -> Optional[AsiPairingHeader]:
        if not self.has_pairing():
            return None
        ph = AsiPairingHeader()
        if self._lib.lila_asi_vision_pairing_info(self._vmodel, ctypes.byref(ph)) != 0:
            return None
        return ph

    def check_compat(self, llm_cfg: AsiV2ModelConfig) -> None:
        if not self._vmodel:
            raise RuntimeError("AsiVisionSession is closed")
        err = ctypes.create_string_buffer(256)
        if self._lib.lila_asi_vision_check_compat(self._vmodel, ctypes.byref(llm_cfg), err, 256) != 0:
            raise RuntimeError(f"Vision compat check failed: {err.value.decode('utf-8', 'replace')}")

    def plan_grid(
        self, width: int, height: int, min_pixels: int = 56 * 56, max_pixels: int = 448 * 448
    ) -> AsiVisionPlan:
        if not self._vmodel:
            raise RuntimeError("AsiVisionSession is closed")
        plan = AsiVisionPlan()
        if (
            self._lib.lila_asi_vision_plan_grid(
                self._vmodel, int(width), int(height), int(min_pixels), int(max_pixels), ctypes.byref(plan)
            )
            != 0
        ):
            raise RuntimeError("lila_asi_vision_plan_grid failed")
        return plan

    def encode_rgb(
        self,
        rgb_u8: np.ndarray,
        *,
        min_pixels: int = 56 * 56,
        max_pixels: int = 448 * 448,
        n_threads: int = 2,
        use_pairing: Optional[bool] = None,
    ) -> EncodedImage:
        if not self._vmodel:
            raise RuntimeError("AsiVisionSession is closed")
        arr = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB [H, W, 3] uint8 image, got shape={arr.shape}")
        h, w, _ = arr.shape
        plan = self.plan_grid(w, h, min_pixels=min_pixels, max_pixels=max_pixels)

        paired = self.has_pairing() if use_pairing is None else bool(use_pairing)
        if paired:
            ph = self.pairing_info()
            if ph is None:
                raise RuntimeError("Requested paired encode on .asiv without approved pairing")
            out_dim = int(ph.out_dim)
        else:
            out_dim = int(self.header.projection_dim)

        embds = np.zeros(int(plan.n_tokens) * out_dim, dtype=np.float32)
        iv = AsiImageView()
        iv.rgb_u8 = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        iv.width = int(w)
        iv.height = int(h)
        iv.stride_bytes = int(w * 3)
        thw = (ctypes.c_uint32 * 3)()
        ntok = ctypes.c_uint32(0)
        xf = AsiCoordTransform()

        if paired:
            rc = self._lib.lila_asi_vision_encode_paired_rgb(
                self._vmodel,
                ctypes.byref(iv),
                int(min_pixels),
                int(max_pixels),
                int(n_threads),
                None,
                embds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                embds.size,
                thw,
                ctypes.byref(ntok),
                ctypes.byref(xf),
            )
        else:
            rc = self._lib.lila_asi_vision_encode_rgb(
                self._vmodel,
                ctypes.byref(iv),
                int(min_pixels),
                int(max_pixels),
                int(n_threads),
                None,
                embds.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                embds.size,
                thw,
                ctypes.byref(ntok),
                ctypes.byref(xf),
                None,
                None,
            )
        if rc != 0:
            raise RuntimeError(f"Vision encode failed with rc={rc}")

        src_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        return EncodedImage(
            embds=embds.reshape(int(ntok.value), out_dim),
            grid_thw=(int(thw[0]), int(thw[1]), int(thw[2])),
            n_tokens=int(ntok.value),
            resized_w=int(xf.dst_width),
            resized_h=int(xf.dst_height),
            source_sha256=src_sha,
            adapter_id=int(self.header.adapter_id),
            paired=paired,
        )


@dataclass
class InspectionBudget:
    max_iterations: int = 3
    region_budget: int = 2
    compute_budget_ms: float = 15000.0
    min_pixels: int = 56 * 56
    coarse_max_pixels: int = 112 * 112
    fine_max_pixels: int = 280 * 280


def propose_salient_rois(
    rgb_u8: np.ndarray, *, max_rois: int = 2, min_roi_frac: float = 0.25
) -> List[Tuple[int, int, int, int, float]]:
    """Deterministic gradient/variance ROI detector requiring zero neural compute."""
    arr = np.asarray(rgb_u8, dtype=np.float32)
    h, w, _ = arr.shape
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    if w > 1:
        gx[:, :-1] = np.abs(gray[:, 1:] - gray[:, :-1])
    if h > 1:
        gy[:-1, :] = np.abs(gray[1:, :] - gray[:-1, :])
    energy = gx + gy

    grid_r, grid_c = 4, 4
    cell_h = max(1, h // grid_r)
    cell_w = max(1, w // grid_c)
    candidates: List[Tuple[float, int, int, int, int]] = []
    roi_h = max(28, int(round(h * min_roi_frac)))
    roi_w = max(28, int(round(w * min_roi_frac)))

    for gy_i in range(grid_r):
        for gx_i in range(grid_c):
            y0 = gy_i * cell_h
            x0 = gx_i * cell_w
            y1 = h if gy_i == grid_r - 1 else min(h, y0 + cell_h)
            x1 = w if gx_i == grid_c - 1 else min(w, x0 + cell_w)
            patch = energy[y0:y1, x0:x1]
            score = float(np.mean(patch)) if patch.size else 0.0
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            bx0 = max(0, min(w - roi_w, cx - roi_w // 2))
            by0 = max(0, min(h - roi_h, cy - roi_h // 2))
            candidates.append((score, bx0, by0, min(w, bx0 + roi_w), min(h, by0 + roi_h)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    selected: List[Tuple[int, int, int, int, float]] = []
    for score, x0, y0, x1, y1 in candidates:
        if score <= 0.0:
            continue
        overlap = False
        for sx0, sy0, sx1, sy1, _ in selected:
            ix0, iy0 = max(x0, sx0), max(y0, sy0)
            ix1, iy1 = min(x1, sx1), min(y1, sy1)
            if ix1 > ix0 and iy1 > iy0:
                inter = float((ix1 - ix0) * (iy1 - iy0))
                area = float((x1 - x0) * (y1 - y0))
                if inter > 0.35 * area:
                    overlap = True
                    break
        if not overlap:
            selected.append((x0, y0, x1, y1, score))
            if len(selected) >= max_rois:
                break
    return selected


class AdaptiveVisionController:
    """
    Bounded coarse -> ROI active inspection controller with content-hash caching
    and exact coordinate provenance into VisualEvidenceLedger.
    """

    def __init__(self, session: AsiVisionSession) -> None:
        self.session = session
        self._cache: Dict[Tuple[str, Tuple[int, int, int, int], int, int, bool], EncodedImage] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def _encode_cached(
        self,
        full_sha: str,
        roi_box: Tuple[int, int, int, int],
        crop_rgb: np.ndarray,
        *,
        min_pixels: int,
        max_pixels: int,
        n_threads: int,
        use_pairing: Optional[bool],
    ) -> Tuple[EncodedImage, bool]:
        paired = self.session.has_pairing() if use_pairing is None else bool(use_pairing)
        key = (full_sha, roi_box, int(min_pixels), int(max_pixels), paired)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key], True
        self.cache_misses += 1
        enc = self.session.encode_rgb(
            crop_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            n_threads=n_threads,
            use_pairing=use_pairing,
        )
        self._cache[key] = enc
        return enc, False

    def inspect(
        self,
        rgb_u8: np.ndarray,
        *,
        budget: Optional[InspectionBudget] = None,
        n_threads: int = 2,
        use_pairing: Optional[bool] = None,
    ) -> Dict[str, object]:
        cfg = budget or InspectionBudget()
        arr = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
        h, w, _ = arr.shape
        full_sha = hashlib.sha256(arr.tobytes()).hexdigest()
        ledger = VisualEvidenceLedger()
        t0 = time.perf_counter()

        coarse_enc, coarse_cached = self._encode_cached(
            full_sha,
            (0, 0, w, h),
            arr,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.coarse_max_pixels,
            n_threads=n_threads,
            use_pairing=use_pairing,
        )
        ledger.add_record(
            source_image_sha256=full_sha,
            region_id="coarse_full",
            original_box_xyxy=(0.0, 0.0, float(w), float(h)),
            adapter_id=coarse_enc.adapter_id,
            confidence=1.0,
            status=EvidenceStatus.OBSERVED,
            summary=f"coarse full frame {w}x{h} -> {coarse_enc.resized_w}x{coarse_enc.resized_h}",
            n_tokens=coarse_enc.n_tokens,
            grid_thw=coarse_enc.grid_thw,
            cached=coarse_cached,
        )

        roi_encodings: List[EncodedImage] = []
        rois = propose_salient_rois(arr, max_rois=cfg.region_budget)
        iterations = 1
        for idx, (x0, y0, x1, y1, score) in enumerate(rois):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if iterations >= cfg.max_iterations or elapsed_ms >= cfg.compute_budget_ms:
                break
            crop_rgb = np.ascontiguousarray(arr[y0:y1, x0:x1])
            chain = CoordTransformChain(w, h).add_crop(x0, y0, x1 - x0, y1 - y0, w, h)
            fine_enc, fine_cached = self._encode_cached(
                full_sha,
                (x0, y0, x1, y1),
                crop_rgb,
                min_pixels=cfg.min_pixels,
                max_pixels=cfg.fine_max_pixels,
                n_threads=n_threads,
                use_pairing=use_pairing,
            )
            chain.add_resize(x1 - x0, y1 - y0, fine_enc.resized_w, fine_enc.resized_h)
            orig_box = chain.to_original_box(0.0, 0.0, float(fine_enc.resized_w), float(fine_enc.resized_h))
            ledger.add_record(
                source_image_sha256=full_sha,
                region_id=f"roi_{idx}",
                original_box_xyxy=orig_box,
                adapter_id=fine_enc.adapter_id,
                confidence=min(1.0, max(0.1, score / 32.0)),
                status=EvidenceStatus.OBSERVED,
                summary=f"roi_{idx} fine crop ({x0},{y0})-({x1},{y1}) edge={score:.2f}",
                n_tokens=fine_enc.n_tokens,
                grid_thw=fine_enc.grid_thw,
                cached=fine_cached,
            )
            roi_encodings.append(fine_enc)
            iterations += 1

        total_ms = (time.perf_counter() - t0) * 1000.0
        total_tokens = coarse_enc.n_tokens + sum(e.n_tokens for e in roi_encodings)
        return {
            "coarse": coarse_enc,
            "rois": roi_encodings,
            "ledger": ledger,
            "iterations": iterations,
            "total_tokens": total_tokens,
            "elapsed_ms": total_ms,
        }

    def compare_inspection_modes(
        self,
        rgb_u8: np.ndarray,
        *,
        budget: Optional[InspectionBudget] = None,
        n_threads: int = 2,
        use_pairing: Optional[bool] = None,
    ) -> Dict[str, Dict[str, object]]:
        """Compare single-pass low-res vs fixed high-res vs adaptive active inspection."""
        cfg = budget or InspectionBudget()
        t0 = time.perf_counter()
        low = self.session.encode_rgb(
            rgb_u8,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.coarse_max_pixels,
            n_threads=n_threads,
            use_pairing=use_pairing,
        )
        t_low = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        high = self.session.encode_rgb(
            rgb_u8,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.fine_max_pixels,
            n_threads=n_threads,
            use_pairing=use_pairing,
        )
        t_high = (time.perf_counter() - t1) * 1000.0

        self._cache.clear()
        adap = self.inspect(rgb_u8, budget=cfg, n_threads=n_threads, use_pairing=use_pairing)
        return {
            "single_pass_low_res": {"elapsed_ms": t_low, "n_tokens": low.n_tokens, "regions": 1},
            "fixed_high_res": {"elapsed_ms": t_high, "n_tokens": high.n_tokens, "regions": 1},
            "adaptive_coarse_roi": {
                "elapsed_ms": adap["elapsed_ms"],
                "n_tokens": adap["total_tokens"],
                "regions": len(adap["ledger"].records),  # type: ignore[union-attr]
            },
        }
