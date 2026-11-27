#!/usr/bin/env python3
"""
ASI Neural Pairing Registry — 7-state fail-closed lifecycle tracker.

States:
  0: DISCOVERED
  1: COMPAT_CHECKED
  2: LEARNING_REQUIRED
  3: TRAINING
  4: VALIDATION
  5: APPROVED
  6: CONTINUOUS_IMPROVEMENT
  7: REJECTED

Invalid transitions (such as promoting directly to APPROVED without VALIDATION,
or promoting a failed candidate) raise ValueError so unvalidated pairings can
never silently enter inference.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional


class PairingState(IntEnum):
    DISCOVERED = 0
    COMPAT_CHECKED = 1
    LEARNING_REQUIRED = 2
    TRAINING = 3
    VALIDATION = 4
    APPROVED = 5
    CONTINUOUS_IMPROVEMENT = 6
    REJECTED = 7


ALLOWED_TRANSITIONS: Dict[PairingState, tuple[PairingState, ...]] = {
    PairingState.DISCOVERED: (PairingState.COMPAT_CHECKED, PairingState.REJECTED),
    PairingState.COMPAT_CHECKED: (
        PairingState.LEARNING_REQUIRED,
        PairingState.APPROVED,  # Native pairing verified compatible
        PairingState.REJECTED,
    ),
    PairingState.LEARNING_REQUIRED: (PairingState.TRAINING, PairingState.REJECTED),
    PairingState.TRAINING: (PairingState.TRAINING, PairingState.VALIDATION, PairingState.REJECTED),
    PairingState.VALIDATION: (PairingState.APPROVED, PairingState.REJECTED),
    PairingState.APPROVED: (PairingState.CONTINUOUS_IMPROVEMENT, PairingState.APPROVED),
    PairingState.CONTINUOUS_IMPROVEMENT: (
        PairingState.VALIDATION,
        PairingState.APPROVED,
        PairingState.REJECTED,
    ),
    PairingState.REJECTED: (PairingState.LEARNING_REQUIRED, PairingState.TRAINING),
}


@dataclass
class PairingRecord:
    pair_key: str
    vision_sha256: str
    llm_sha256: str
    target_llm_arch_id: int
    in_dim: int
    out_dim: int
    state: int = int(PairingState.DISCOVERED)
    native_compatible: bool = False
    staging_asip_path: str = ""
    sealed_cartridge_path: str = ""
    val_passed: bool = False
    val_cosine: float = 0.0
    val_loss: float = 0.0
    train_steps: int = 0
    evidence_source: str = ""
    updated_at: float = field(default_factory=time.time)
    history: List[dict] = field(default_factory=list)


class PairingRegistry:
    """Fail-closed persistent registry tracking discovered vision-LLM pairings."""

    def __init__(self, registry_path: Path) -> None:
        self.path = Path(registry_path)
        self.records: Dict[str, PairingRecord] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in raw.get("pairs", {}).items():
                self.records[k] = PairingRecord(**v)

    @staticmethod
    def make_key(vision_sha256: str, llm_sha256: str) -> str:
        return f"{vision_sha256.lower()}:{llm_sha256.lower()}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        payload = {"version": 1, "pairs": {k: asdict(v) for k, v in self.records.items()}}
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def discover(
        self,
        *,
        vision_sha256: str,
        llm_sha256: str,
        target_llm_arch_id: int,
        in_dim: int,
        out_dim: int,
        evidence_source: str = "",
    ) -> PairingRecord:
        key = self.make_key(vision_sha256, llm_sha256)
        if key in self.records:
            return self.records[key]
        rec = PairingRecord(
            pair_key=key,
            vision_sha256=vision_sha256.lower(),
            llm_sha256=llm_sha256.lower(),
            target_llm_arch_id=int(target_llm_arch_id),
            in_dim=int(in_dim),
            out_dim=int(out_dim),
            state=int(PairingState.DISCOVERED),
            evidence_source=evidence_source,
            history=[{"state": "DISCOVERED", "ts": time.time()}],
        )
        self.records[key] = rec
        self.save()
        return rec

    def transition(
        self,
        pair_key: str,
        new_state: PairingState,
        *,
        native_compatible: Optional[bool] = None,
        staging_asip_path: Optional[str] = None,
        sealed_cartridge_path: Optional[str] = None,
        val_passed: Optional[bool] = None,
        val_cosine: Optional[float] = None,
        val_loss: Optional[float] = None,
        train_steps: Optional[int] = None,
        note: str = "",
    ) -> PairingRecord:
        if pair_key not in self.records:
            raise KeyError(f"Unknown pair_key: {pair_key}")
        rec = self.records[pair_key]
        cur = PairingState(rec.state)
        if new_state not in ALLOWED_TRANSITIONS[cur]:
            raise ValueError(f"Invalid pairing transition: {cur.name} -> {new_state.name}")

        if new_state == PairingState.APPROVED:
            effective_val = rec.val_passed if val_passed is None else val_passed
            effective_native = rec.native_compatible if native_compatible is None else native_compatible
            if cur == PairingState.COMPAT_CHECKED and not effective_native:
                raise ValueError("Cannot jump from COMPAT_CHECKED to APPROVED unless native_compatible=True")
            if cur in (PairingState.VALIDATION, PairingState.CONTINUOUS_IMPROVEMENT) and not effective_val:
                raise ValueError("Cannot promote to APPROVED unless val_passed=True")

        rec.state = int(new_state)
        if native_compatible is not None:
            rec.native_compatible = bool(native_compatible)
        if staging_asip_path is not None:
            rec.staging_asip_path = staging_asip_path
        if sealed_cartridge_path is not None:
            rec.sealed_cartridge_path = sealed_cartridge_path
        if val_passed is not None:
            rec.val_passed = bool(val_passed)
        if val_cosine is not None:
            rec.val_cosine = float(val_cosine)
        if val_loss is not None:
            rec.val_loss = float(val_loss)
        if train_steps is not None:
            rec.train_steps = int(train_steps)
        rec.updated_at = time.time()
        rec.history.append({"state": new_state.name, "ts": rec.updated_at, "note": note})
        self.save()
        return rec

    def is_approved(self, vision_sha256: str, llm_sha256: str) -> bool:
        key = self.make_key(vision_sha256, llm_sha256)
        rec = self.records.get(key)
        if not rec:
            return False
        if PairingState(rec.state) not in (PairingState.APPROVED, PairingState.CONTINUOUS_IMPROVEMENT):
            return False
        return bool(rec.val_passed or rec.native_compatible)
