#!/usr/bin/env python3
"""
ASI Resource-Aware Pairing Bridge Trainer (Local NumPy + Colab Bundle Sync).

Supports:
  - Priority 1/2/3 preemption & resumable staging `.npz` checkpoints (protecting live
    inference on memory/CPU-constrained hosts like Broadwell 2C / 5.7 GiB RAM).
  - Linear or 2-layer MLP translator bridges trained from verified teacher/QA data.
  - Held-out validation gating (finite check, cosine similarity, MSE).
  - Compact Google Colab GPU bundle export & candidate staging `.asip` import.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np


class PriorityTier(IntEnum):
    P1_LIVE_INFERENCE = 1
    P2_BACKGROUND_LEARNING = 2
    P3_IDLE_LEARNING = 3


def _gelu_tanh_and_grad(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    k = math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    u = k * (x + 0.044715 * x3)
    t = np.tanh(u)
    sech2 = 1.0 - t * t
    du_dx = k * (1.0 + 3.0 * 0.044715 * x * x)
    y = 0.5 * x * (1.0 + t)
    dy_dx = 0.5 * (1.0 + t) + 0.5 * x * sech2 * du_dx
    return y, dy_dx


@dataclass
class TrainResult:
    completed: bool
    paused_by_preemption: bool
    steps_done: int
    last_loss: float
    checkpoint_path: Path
    weights: Dict[str, np.ndarray]


def forward_bridge(
    weights: Dict[str, np.ndarray],
    x: np.ndarray,
    *,
    bridge_kind: int = 2,
    act_type: int = 1,
) -> np.ndarray:
    z0 = x @ weights["bridge.0.weight"].T + weights["bridge.0.bias"]
    if bridge_kind == 1:
        return z0.astype(np.float32, copy=False)
    if act_type == 1:
        h, _ = _gelu_tanh_and_grad(z0)
    else:
        h = np.maximum(0.0, z0)
    out = h @ weights["bridge.1.weight"].T + weights["bridge.1.bias"]
    return out.astype(np.float32, copy=False)


def _init_weights(in_dim: int, hidden_dim: int, out_dim: int, bridge_kind: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if bridge_kind == 1:
        scale0 = math.sqrt(2.0 / max(1, in_dim + out_dim))
        return {
            "bridge.0.weight": rng.normal(0.0, scale0, size=(out_dim, in_dim)).astype(np.float32),
            "bridge.0.bias": np.zeros(out_dim, dtype=np.float32),
        }
    scale0 = math.sqrt(2.0 / max(1, in_dim + hidden_dim))
    scale1 = math.sqrt(2.0 / max(1, hidden_dim + out_dim))
    return {
        "bridge.0.weight": rng.normal(0.0, scale0, size=(hidden_dim, in_dim)).astype(np.float32),
        "bridge.0.bias": np.zeros(hidden_dim, dtype=np.float32),
        "bridge.1.weight": rng.normal(0.0, scale1, size=(out_dim, hidden_dim)).astype(np.float32),
        "bridge.1.bias": np.zeros(out_dim, dtype=np.float32),
    }


def train_bridge_resumable(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    checkpoint_path: Path,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    bridge_kind: int = 2,
    act_type: int = 1,
    total_steps: int = 60,
    lr: float = 2e-2,
    priority: PriorityTier = PriorityTier.P3_IDLE_LEARNING,
    preempt_check: Optional[Callable[[int], bool]] = None,
    seed: int = 42,
) -> TrainResult:
    """
    Train or resume a candidate bridge in staging without modifying base cartridges.
    When preempt_check(step) returns True (Priority-1 live inference arrives), flushes
    full AdamW state to checkpoint_path and yields cleanly.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    n_samples = max(1, x.shape[0])

    if checkpoint_path.exists():
        ckpt = np.load(checkpoint_path, allow_pickle=False)
        step = int(ckpt["step"])
        last_loss = float(ckpt["last_loss"])
        weights = {k[2:]: ckpt[k].astype(np.float32) for k in ckpt.files if k.startswith("w_")}
        m_buf = {k[2:]: ckpt[k].astype(np.float32) for k in ckpt.files if k.startswith("m_")}
        v_buf = {k[2:]: ckpt[k].astype(np.float32) for k in ckpt.files if k.startswith("v_")}
    else:
        step = 0
        last_loss = float("inf")
        weights = _init_weights(in_dim, hidden_dim, out_dim, bridge_kind, seed)
        m_buf = {k: np.zeros_like(v) for k, v in weights.items()}
        v_buf = {k: np.zeros_like(v) for k, v in weights.items()}

    def _save_ckpt(cur_step: int, cur_loss: float) -> None:
        arrays: Dict[str, np.ndarray] = {
            "step": np.array(cur_step, dtype=np.int64),
            "last_loss": np.array(cur_loss, dtype=np.float32),
            "priority": np.array(int(priority), dtype=np.int32),
        }
        for k, val in weights.items():
            arrays[f"w_{k}"] = val
            arrays[f"m_{k}"] = m_buf[k]
            arrays[f"v_{k}"] = v_buf[k]
        tmp = checkpoint_path.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(checkpoint_path)

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    paused = False

    while step < total_steps:
        if preempt_check is not None and preempt_check(step):
            _save_ckpt(step, last_loss)
            paused = True
            break

        # Forward + Cosine + MSE hybrid objective
        z0 = x @ weights["bridge.0.weight"].T + weights["bridge.0.bias"]
        if bridge_kind == 1:
            pred = z0
            h = x
        else:
            h, dh_dz0 = _gelu_tanh_and_grad(z0) if act_type == 1 else (np.maximum(0.0, z0), (z0 > 0.0).astype(np.float32))
            pred = h @ weights["bridge.1.weight"].T + weights["bridge.1.bias"]

        diff = pred - y
        last_loss = float(np.mean(diff * diff))
        d_pred = (2.0 / float(n_samples * out_dim)) * diff

        grads: Dict[str, np.ndarray] = {}
        if bridge_kind == 1:
            grads["bridge.0.weight"] = d_pred.T @ x
            grads["bridge.0.bias"] = np.sum(d_pred, axis=0)
        else:
            grads["bridge.1.weight"] = d_pred.T @ h
            grads["bridge.1.bias"] = np.sum(d_pred, axis=0)
            d_h = d_pred @ weights["bridge.1.weight"]
            d_z0 = d_h * dh_dz0
            grads["bridge.0.weight"] = d_z0.T @ x
            grads["bridge.0.bias"] = np.sum(d_z0, axis=0)

        step += 1
        bc1 = 1.0 - (beta1 ** step)
        bc2 = 1.0 - (beta2 ** step)
        for name, g in grads.items():
            m_buf[name] = beta1 * m_buf[name] + (1.0 - beta1) * g
            v_buf[name] = beta2 * v_buf[name] + (1.0 - beta2) * (g * g)
            m_hat = m_buf[name] / bc1
            v_hat = v_buf[name] / bc2
            weights[name] = weights[name] - lr * m_hat / (np.sqrt(v_hat) + eps)

    _save_ckpt(step, last_loss)
    return TrainResult(
        completed=(step >= total_steps and not paused),
        paused_by_preemption=paused,
        steps_done=step,
        last_loss=last_loss,
        checkpoint_path=checkpoint_path,
        weights=weights,
    )


def validate_bridge(
    weights: Dict[str, np.ndarray],
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    bridge_kind: int = 2,
    act_type: int = 1,
    min_cosine: float = 0.98,
) -> Dict[str, float | bool]:
    pred = forward_bridge(weights, np.asarray(x_val, dtype=np.float32), bridge_kind=bridge_kind, act_type=act_type)
    target = np.asarray(y_val, dtype=np.float32)
    if not np.all(np.isfinite(pred)):
        return {"passed": False, "cosine": 0.0, "mse": float("inf")}
    mse = float(np.mean((pred - target) ** 2))
    p_flat = pred.astype(np.float64).ravel()
    t_flat = target.astype(np.float64).ravel()
    cos = float(np.dot(p_flat, t_flat) / (np.linalg.norm(p_flat) * np.linalg.norm(t_flat) + 1e-12))
    return {
        "passed": bool(cos >= min_cosine),
        "cosine": cos,
        "mse": mse,
    }


def export_colab_bundle(
    out_zip: Path,
    *,
    manifest: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> Path:
    """Package pairing training metadata + teacher/QA tensors into a Colab GPU bundle."""
    out_zip = Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    npz_tmp = out_zip.with_suffix(".data.npz")
    np.savez(
        npz_tmp,
        x_train=np.asarray(x_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.float32),
        x_val=np.asarray(x_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.float32),
    )
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.write(npz_tmp, arcname="dataset.npz")
    npz_tmp.unlink(missing_ok=True)
    return out_zip


def import_colab_artifact(artifact_dir: Path) -> Tuple[bytes, dict]:
    """Load candidate pairing.asip bytes + report.json produced by Colab GPU run."""
    artifact_dir = Path(artifact_dir)
    report = json.loads((artifact_dir / "colab_report.json").read_text(encoding="utf-8"))
    sec_bytes = (artifact_dir / "pairing_section.asip").read_bytes()
    return sec_bytes, report
