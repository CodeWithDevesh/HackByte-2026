from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from src.core.events import EventPriority, ModelEvent
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.weapon_detection.config import WeaponDetectionConfig
from src.models.weapon_detection.utils import draw_alert, is_weapon
from src.speech.client import SpeechClient


@dataclass
class WeaponDetectionModel:
    cfg: WeaponDetectionConfig
    speech: SpeechClient
    window_name: str = "Weapon Detection (Pi Optimized)"

    _vision_ready: bool = field(default=False, repr=False)
    _yolo: Any = field(default=None, repr=False)

    _frame_counter: int = field(default=0, repr=False)   # ✅ NEW

    _current_detections: list = field(default_factory=list, repr=False)

    _weapon_frame_count: int = field(default=0, repr=False)
    _last_alert_time: float = field(default=0.0, repr=False)
    _last_label: str = field(default="weapon", repr=False)

    # -------------------- LOAD MODEL --------------------
    def ensure_vision_resources(self) -> None:
        if self._vision_ready:
            return

        model_path = self.cfg.model_path
        if not model_path or not model_path.endswith(".pt"):
            model_path = "yolov8n.pt"

        self._yolo = YOLO(model_path)
        self._vision_ready = True

    # -------------------- PROCESS FRAME --------------------
    def process_frame(self, frame: np.ndarray, *, draw_camera_hint: bool = True) -> None:
        self.ensure_vision_resources()

        # 🔥 RUN EVERY N FRAMES
        self._frame_counter = (self._frame_counter + 1) % self.cfg.process_every_n_frames

        if self._frame_counter == 0:
            self._current_detections = []

            small_frame = cv2.resize(frame, (320, 240))
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = self._yolo(small_frame, verbose=False)

            weapon_found = False
            max_conf = 0.0

            if results and results[0].boxes is not None:
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

                    scale_x = frame.shape[1] / 320
                    scale_y = frame.shape[0] / 240

                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)

                    self._current_detections.append((x1, y1, x2, y2, label, conf))

            # -------- ALERT --------
            if weapon_found:
                self._weapon_frame_count += 1
            else:
                self._weapon_frame_count = 0

            if self._weapon_frame_count >= self.cfg.frame_threshold:
                now = time.time()

                if (now - self._last_alert_time) > self.cfg.alert_cooldown_s:
                    ev = ModelEvent(
                        source="weapon_detection",
                        type="weapon_alert",
                        message=f"Warning, {self._last_label} detected",
                        priority=EventPriority.HIGH,
                        dedupe_key=f"weapon:{self._last_label}",
                        cooldown_s=self.cfg.alert_cooldown_s,
                        metadata={
                            "label": self._last_label,
                            "confidence": max_conf,
                        },
                    )

                    if not self.speech.post_event(ev):
                        print("[Warning] Speech router not reachable.")

                    self._last_alert_time = now

                self._weapon_frame_count = 0

        # -------------------- DRAW (ALWAYS) --------------------
        for (x1, y1, x2, y2, label, conf) in self._current_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        if self._weapon_frame_count > 0:
            draw_alert(frame)

        if draw_camera_hint:
            cv2.putText(
                frame,
                f"Cam: (q=quit) | Skip: {self.cfg.process_every_n_frames}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    # -------------------- RUN --------------------
    def run(self, list_cameras: bool = False) -> None:
        camera_cycle = probe_cameras(
            max_index=self.cfg.max_camera_index
        ) or [self.cfg.camera_index]

        current_camera_index = camera_cycle[0]
        cap = open_camera(current_camera_index)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("Starting weapon detection (Frame Skipping Enabled)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            self.process_frame(frame, draw_camera_hint=False)

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


def build_default_weapon_model() -> WeaponDetectionModel:
    cfg = WeaponDetectionConfig()
    speech = SpeechClient(base_url=cfg.router_url, timeout_s=0.2)
    return WeaponDetectionModel(cfg=cfg, speech=speech)