from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable

from src.models.face_recognition.model import build_default_face_model


@dataclass(frozen=True)
class OrchestratorConfig:
    enabled_models: list[str]

    @staticmethod
    def from_env() -> "OrchestratorConfig":
        raw = os.getenv("ENABLED_MODELS", "face_recognition")
        enabled = [m.strip() for m in raw.split(",") if m.strip()]
        return OrchestratorConfig(enabled_models=enabled)


class Orchestrator:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        runners: list[Callable[[], None]] = []

        if "face_recognition" in self.cfg.enabled_models:
            model = build_default_face_model()
            runners.append(lambda: model.run(list_cameras=False))

        if not runners:
            raise RuntimeError("No models enabled. Set ENABLED_MODELS=face_recognition,...")

        threads: list[threading.Thread] = []
        for fn in runners:
            t = threading.Thread(target=fn, daemon=False)
            t.start()
            threads.append(t)

        # Wait for all model threads (typically face model blocks until quit).
        for t in threads:
            t.join()

