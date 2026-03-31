from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.core.events import EventPriority, ModelEvent
from src.models.face_recognition.camera import open_camera
from src.models.weapon_detection.config import WeaponDetectionConfig
from src.models.weapon_detection.utils import draw_alert, is_weapon
from src.speech.client import SpeechClient


@dataclass
class WeaponDetectionModel:
    cfg: WeaponDetectionConfig
    speech: SpeechClient
    window_name: str = "Weapon Detection"

    _yolo: Any = field(default=None, repr=False)
    _last_alert_time: float = field(default=0.0, repr=False)
    _weapon_frame_count: int = field(default=0, repr=False)
    _last_label: str = field(default="weapon", repr=False)

    def ensure_yolo(self) -> None:
        if self._yolo is None:
            self._yolo = YOLO(self.cfg.model_path)

    def process_frame(self, frame: np.ndarray, *, draw_ui_hint: bool = True) -> None:
        """Run weapon YOLO on one frame; draw overlays in place."""
        self.ensure_yolo()
        assert self._yolo is not None

        results = self._yolo(frame, verbose=False)
        weapon_found = False
        max_conf = 0.0

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = str(self._yolo.names[cls])
            conf = float(box.conf[0])
            if conf < self.cfg.conf_threshold or not is_weapon(label):
                continue

            weapon_found = True
            max_conf = max(max_conf, conf)
            self._last_label = label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        if weapon_found:
            self._weapon_frame_count += 1
        else:
            self._weapon_frame_count = 0

        if self._weapon_frame_count >= self.cfg.frame_threshold:
            draw_alert(frame)
            now = time.time()
            if (now - self._last_alert_time) > self.cfg.alert_cooldown_s:
                ev = ModelEvent(
                    source="weapon_detection",
                    type="weapon_alert",
                    message="Warning, weapon detected.",
                    priority=EventPriority.HIGH,
                    dedupe_key=f"weapon:{self._last_label}",
                    cooldown_s=self.cfg.alert_cooldown_s,
                    metadata={"label": self._last_label, "confidence": max_conf},
                )
                if not self.speech.post_event(ev):
                    print("[Warning] Speech router not reachable.")
                self._last_alert_time = now
            self._weapon_frame_count = 0

        if draw_ui_hint:
            cv2.putText(
                frame,
                "q=quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    def run(self) -> None:
        self.ensure_yolo()
        cap = open_camera(self.cfg.camera_index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(frame, draw_ui_hint=True)
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def build_default_weapon_model() -> WeaponDetectionModel:
    cfg = WeaponDetectionConfig()
    speech = SpeechClient(base_url=cfg.router_url, timeout_s=0.2)
    return WeaponDetectionModel(cfg=cfg, speech=speech)
