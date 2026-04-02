from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO

# Imported from your existing project structure
from src.core.events import EventPriority, ModelEvent
from src.speech.client import SpeechClient
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.weapon_detection.config import WeaponDetectionConfig

@dataclass
class WeaponTracker:
    """Helper class to track a weapon's movement over time."""
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_announced_state: str = "none"
    last_event_time: float = 0.0
    is_active: bool = True
    has_announced_entrance: bool = False
    confidence: float = 0.0

@dataclass
class WeaponDetectionModel:
    cfg: WeaponDetectionConfig
    speech: SpeechClient
    window_name: str = "Weapon Detection System"

    _model: YOLO = field(init=False, repr=False)
    _current_weapon_data: list = field(default_factory=list, repr=False)
    
    # Threading and Throttling (Increased to 3 to prevent starvation during network alerts)
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=3), repr=False
    )
    _is_detecting: bool = field(default=False, repr=False)
    _last_detection_time: float = field(default=0.0, repr=False)

    # Intelligent Spatial Tracking
    _trackers: dict[str, WeaponTracker] = field(default_factory=dict, repr=False)
    _tracker_id_counter: int = field(default=0, repr=False)

    def __post_init__(self):
        print(f"Loading YOLO model from: {self.cfg.model_path}")
        self._model = YOLO(self.cfg.model_path)

    def _detect_weapons_worker(self, frame_copy: np.ndarray) -> None:
        """Runs in the background thread. Catches errors so they don't fail silently."""
        try:
            new_weapon_data = []
            results = self._model.predict(
                frame_copy, 
                conf=self.cfg.confidence_threshold, 
                verbose=False
            )
            
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = self._model.names[cls]
                    
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    new_weapon_data.append(((x, y, w, h), name, conf))

            self._current_weapon_data = new_weapon_data

        except Exception as e:
            print(f"\n[AI THREAD CRASHED]: {e}\n")
            
        finally:
            self._is_detecting = False

    def _analyze_and_alert(self, frame_width: int):
        current_time = time.time()

        for tracker_id, tracker in list(self._trackers.items()):
            # 1. Handle Expiration
            if not tracker.is_active:
                if len(tracker.history) > 0 and (current_time - tracker.history[-1][0] > 1.5):
                    del self._trackers[tracker_id]
                continue

            # 2. Handle Entrance & Alerting
            if not tracker.has_announced_entrance and len(tracker.history) > 3:
                tracker.has_announced_entrance = True
                tracker.last_announced_state = "entered"
                
                name_split = tracker_id.split('_')[0]
                alert_message = f"Warning. {name_split.replace('_', ' ')} detected."
                
                # Create the event object
                ev = ModelEvent(
                    source="weapon_detection",
                    type="weapon_spotted",
                    message=alert_message,
                    priority=EventPriority.HIGH,  
                    dedupe_key=f"weapon:{name_split}",
                    cooldown_s=15.0, 
                )
                
                # Send the network alert in the background thread to prevent camera freeze
                self._executor.submit(self.speech.post_event, ev)
                
                print(f"[HIGH PRIORITY ALERT] {alert_message}")
                
                tracker.last_event_time = current_time

    def process_frame(self, frame: np.ndarray, *, draw_camera_hint: bool = True) -> None:
        current_time = time.time()
        frame_width = frame.shape[1]

        for tracker in self._trackers.values():
            tracker.is_active = False

        display_data = []

        # 1. SMART SPATIAL TRACKING
        for (ox, oy, ow, oh), name, conf in self._current_weapon_data:
            cx, cy = ox + ow / 2, oy + oh / 2
            best_id = None
            min_dist = float("inf")

            for tracker_id, tracker in self._trackers.items():
                if tracker_id.startswith(name) and len(tracker.history) > 0:
                    _, _, last_cx, last_cy = tracker.history[-1]
                    spatial_dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                    if spatial_dist < (ow * 1.5) and spatial_dist < min_dist:
                        min_dist = spatial_dist
                        best_id = tracker_id

            if best_id is None:
                self._tracker_id_counter += 1
                best_id = f"{name}_{self._tracker_id_counter}"
                self._trackers[best_id] = WeaponTracker()

            self._trackers[best_id].is_active = True
            self._trackers[best_id].confidence = conf
            self._trackers[best_id].history.append((current_time, oh, cx, cy))
            display_data.append(((ox, oy, ow, oh), best_id, conf))

        # 2. RUN ALERTS
        self._analyze_and_alert(frame_width)

        # 3. ASYNC INFERENCE THROTTLING
        if not self._is_detecting and (current_time - self._last_detection_time > 0.1):
            self._is_detecting = True
            self._last_detection_time = current_time
            self._executor.submit(self._detect_weapons_worker, frame.copy())

        # 4. DRAW OVERLAYS
        if draw_camera_hint:
            cv2.putText(frame, "Weapon Monitor Active (c=switch, q=quit)", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for (x, y, w, h), tracker_id, conf in display_data:
            left, top, right, bottom = x, y, x + w, y + h
            name_label = tracker_id.split('_')[0]
            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, f"{name_label} {conf:.2f}", (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    def run(self, list_cameras: bool = False) -> None:
        if list_cameras:
            cams = probe_cameras(max_index=self.cfg.max_camera_index)
            print("Available cameras:", ", ".join(map(str, cams)) if cams else "(none found)")
            return

        try:
            if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass

        # Camera switching logic mirroring face recognition
        camera_cycle = probe_cameras(max_index=self.cfg.max_camera_index) or [self.cfg.camera_index]
        if self.cfg.camera_index in camera_cycle:
            cam_pos = camera_cycle.index(self.cfg.camera_index)
        else:
            camera_cycle = [self.cfg.camera_index] + camera_cycle
            cam_pos = 0

        current_camera_index = camera_cycle[cam_pos]
        video_capture = open_camera(current_camera_index)

        print("Starting weapon detection stream...")

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break

                self.process_frame(frame, draw_camera_hint=False)
                cv2.putText(
                    frame,
                    f"Cam: {current_camera_index} (c=switch, q=quit)",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
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
                        video_capture.release()
                    except Exception:
                        pass
                    try:
                        video_capture = open_camera(next_index)
                        current_camera_index = next_index
                        print(f"Switched to camera index {current_camera_index}")
                    except Exception as e:
                        print(f"[Warning] Failed to switch to camera {next_index}: {e}")
                        try:
                            video_capture = open_camera(current_camera_index)
                        except Exception:
                            break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            self._executor.shutdown(wait=False)

def build_default_weapon_model() -> WeaponDetectionModel:
    cfg = WeaponDetectionConfig()
    speech = SpeechClient(base_url=cfg.tts_router_url)
    return WeaponDetectionModel(cfg=cfg, speech=speech)