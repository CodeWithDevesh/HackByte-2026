from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable

from src.app.unified_vision import run_unified, should_use_unified
from src.models.face_recognition.model import build_default_face_model
from src.models.weapon_detection.model import build_default_weapon_model


@dataclass(frozen=True)
class OrchestratorConfig:
    enabled_models: list[str]

    @staticmethod
    def from_env() -> "OrchestratorConfig":
        raw = os.getenv("ENABLED_MODELS", "face_recognition,weapon_detection")
        enabled = [m.strip() for m in raw.split(",") if m.strip()]
        return OrchestratorConfig(enabled_models=enabled)


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        enabled = self.cfg.enabled_models

        # Integration decision: one camera + one window when both vision models run.
        if should_use_unified(enabled):
            run_unified(enabled)
            return

        runners: list[Callable[[], None]] = []

        if "face_recognition" in enabled:
            model = build_default_face_model()
            runners.append(lambda: model.run(list_cameras=False))

        if "weapon_detection" in enabled:
            model = build_default_weapon_model()
            runners.append(model.run)

        if not runners:
            raise RuntimeError(
                "No models enabled. Set ENABLED_MODELS=face_recognition,weapon_detection,..."
            )

        threads: list[threading.Thread] = []
        for fn in runners:
            t = threading.Thread(target=fn, daemon=False)
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

