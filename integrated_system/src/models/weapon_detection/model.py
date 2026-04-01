from __future__ import annotations

import time
import threading
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
    window_name: str = "Weapon Detection"

    _yolo: Any = field(default=None, repr=False)
    _vision_ready: bool = field(default=False, repr=False)

    _process_this_frame: bool = field(default=True, repr=False)
    _current_detections: list = field(default_factory=list, repr=False)

    _last_alert_time: float = field(default=0.0, repr=False)
    _weapon_frame_count: int = field(default=0, repr=False)
    _last_label: str = field(default="weapon", repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _detect_thread: threading.Thread | None = field(default=None, repr=False)

    # -------------------- LOAD MODEL --------------------
    def ensure_yolo(self) -> None:
        if self._vision_ready:
            return
        self._yolo = YOLO(self.cfg.model_path)
        self._vision_ready = True

    # -------------------- DETECTION LOGIC --------------------
    def _detect(self, frame: np.ndarray) -> None:
        """Runs YOLO detection in background thread."""
        detections = []
        weapon_found = False
        max_conf = 0.0
        last_label = "weapon"

        results = self._yolo(frame, verbose=False)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = str(self._yolo.names[cls])
            conf = float(box.conf[0])

            if conf < self.cfg.conf_threshold or not is_weapon(label):
                continue

            weapon_found = True
            max_conf = max(max_conf, conf)
            last_label = label

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append((x1, y1, x2, y2, label, conf))

        with self._lock:
            self._current_detections = detections

            if weapon_found:
                self._weapon_frame_count += 1
                self._last_label = last_label
            else:
                self._weapon_frame_count = 0

            # -------- ALERT --------
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

                    # ✅ Non-blocking TTS
                    threading.Thread(
                        target=self.speech.post_event,
                        args=(ev,),
                        daemon=True,
                    ).start()

                    self._last_alert_time = now

                self._weapon_frame_count = 0

    # -------------------- PROCESS FRAME --------------------
    def process_frame(self, frame: np.ndarray, *, draw_ui_hint: bool = True) -> None:
        self.ensure_yolo()
        assert self._yolo is not None

        # -------- CONTROLLED THREADING --------
        if self._process_this_frame:
            if self._detect_thread is None or not self._detect_thread.is_alive():
                frame_copy = frame.copy()  # ✅ Prevent race condition
                self._detect_thread = threading.Thread(
                    target=self._detect,
                    args=(frame_copy,),
                    daemon=True,
                )
                self._detect_thread.start()

        self._process_this_frame = not self._process_this_frame

        # -------- DRAW --------
        with self._lock:
            for (x1, y1, x2, y2, label, conf) in self._current_detections:
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

            if self._weapon_frame_count > 0:
                draw_alert(frame)

        # -------- UI --------
        if draw_ui_hint:
            cv2.putText(
                frame,
                "Cam: (c=switch, q=quit)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    # -------------------- RUN LOOP --------------------
    def run(self, list_cameras: bool = False) -> None:

        if list_cameras:
            cams = probe_cameras(max_index=self.cfg.max_camera_index)
            print("Available cameras:", cams)
            return

        camera_cycle = probe_cameras(
            max_index=self.cfg.max_camera_index
        ) or [self.cfg.camera_index]

        if self.cfg.camera_index in camera_cycle:
            cam_pos = camera_cycle.index(self.cfg.camera_index)
        else:
            camera_cycle = [self.cfg.camera_index] + camera_cycle
            cam_pos = 0

        current_camera_index = camera_cycle[cam_pos]
        cap = open_camera(current_camera_index)

        print("Starting weapon detection stream...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Camera frame not received")
                break

            self.process_frame(frame)

            cv2.putText(
                frame,
                f"Cam: {current_camera_index} (c=switch, q=quit)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c") and len(camera_cycle) > 1:
                cam_pos = (cam_pos + 1) % len(camera_cycle)
                next_index = camera_cycle[cam_pos]

                try:
                    cap.release()
                    cap = open_camera(next_index)
                    current_camera_index = next_index
                    print(f"Switched to camera {current_camera_index}")
                except Exception as e:
                    print(f"[Warning] Camera switch failed: {e}")

        cap.release()
        cv2.destroyAllWindows()


# -------------------- BUILDER --------------------
def build_default_weapon_model() -> WeaponDetectionModel:
    cfg = WeaponDetectionConfig()
    speech = SpeechClient(base_url=cfg.router_url, timeout_s=0.2)
    return WeaponDetectionModel(cfg=cfg, speech=speech)