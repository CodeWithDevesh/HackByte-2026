from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class WeaponDetectionConfig:
    model_path: str = os.getenv("WEAPON_MODEL_PATH", "./assets/weapon_detection/best.pt")
    camera_index: int = int(os.getenv("WEAPON_CAMERA_INDEX", os.getenv("CAMERA_INDEX", "0")))
    conf_threshold: float = float(os.getenv("WEAPON_CONF_THRESHOLD", "0.5"))
    alert_cooldown_s: float = float(os.getenv("WEAPON_ALERT_COOLDOWN", "2.0"))
    frame_threshold: int = int(os.getenv("WEAPON_FRAME_THRESHOLD", "5"))
    max_camera_index: int = int(os.getenv("MAX_CAMERA_INDEX", "5"))
    router_url: str = os.getenv("SPEECH_ROUTER_URL", "http://127.0.0.1:8000")
    process_every_n_frames: int = 5