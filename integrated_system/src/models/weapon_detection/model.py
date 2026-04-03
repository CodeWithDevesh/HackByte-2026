import time
from typing import Any

import cv2
from ultralytics import YOLO

from src.core.events import EventPriority, ModelEvent, RawFrameEvent, ModelResultEvent
from src.core.event_bus import shared_event_bus
from src.models.weapon_detection.config import WeaponDetectionConfig
from src.models.weapon_detection.utils import is_weapon
from src.speech.client import SpeechClient

class WeaponModelNode:
    """Subscribes to raw frames, runs optimized YOLO weapon detection, and triggers alerts."""
    
    def __init__(self, cfg: WeaponDetectionConfig, speech: SpeechClient):
        self.cfg = cfg
        self.speech = speech
        
        print("[Vision] Loading Weapon Detection (Ultra Optimized)...")
        self._yolo = YOLO(self.cfg.model_path or "yolov8n.pt")
        self._yolo.fuse()

        self._process_this_frame = True
        self._weapon_frame_count = 0
        self._last_alert_time = 0.0
        self._last_label = "weapon"

        # Subscribe to the shared video bus
        shared_event_bus.subscribe("raw_frame", self.on_raw_frame)

    def on_raw_frame(self, event: RawFrameEvent):
        frame = event.frame
        drawing_data = [] # Data to send to the Aggregator

        # 🔥 RUN ONLY EVERY OTHER FRAME
        if self._process_this_frame:
            small_frame = cv2.resize(frame, (256, 192))

            # YOLO inference (FAST SETTINGS)
            results = self._yolo(
                small_frame, imgsz=256, conf=self.cfg.conf_threshold, verbose=False, device="cpu"
            )

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

                    # 🔥 SCALE BACK TO ORIGINAL FRAME SIZE
                    scale_x = frame.shape[1] / 256
                    scale_y = frame.shape[0] / 192

                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

                    # Bundle for the Aggregator
                    drawing_data.append({
                        "box": (x1, y1, x2, y2),
                        "label": f"ALERT: {label} {conf:.2f}",
                        "color": (0, 0, 255)  # RED for weapons
                    })

            # -------- ALERT LOGIC --------
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
                        metadata={"label": self._last_label, "confidence": max_conf},
                    )
                    self.speech.post_event(ev)
                    self._last_alert_time = now
                self._weapon_frame_count = 0

        # Toggle for next frame
        self._process_this_frame = not self._process_this_frame

        # Publish results to the Aggregator!
        shared_event_bus.publish(
            "model_result",
            ModelResultEvent(event.frame_id, "WeaponModel", drawing_data),
            run_async=False
        )


def build_default_weapon_node() -> WeaponModelNode:
    cfg = WeaponDetectionConfig()
    speech = SpeechClient(base_url=cfg.router_url, timeout_s=0.2)
    return WeaponModelNode(cfg=cfg, speech=speech)
