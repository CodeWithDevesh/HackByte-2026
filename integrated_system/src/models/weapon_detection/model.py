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
<<<<<<< Updated upstream
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.weapon_detection.config import WeaponDetectionConfig
from src.models.weapon_detection.utils import draw_alert, is_weapon
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
    window_name: str = "Weapon Detection (Pi Ultra Optimized)"

    _vision_ready: bool = field(default=False, repr=False)
    _yolo: Any = field(default=None, repr=False)

    _process_this_frame: bool = field(default=True, repr=False)   # ✅ like face model
    _current_detections: list = field(default_factory=list, repr=False)

    _weapon_frame_count: int = field(default=0, repr=False)
    _last_alert_time: float = field(default=0.0, repr=False)
    _last_label: str = field(default="weapon", repr=False)

    # -------------------- LOAD MODEL --------------------
    def ensure_vision_resources(self) -> None:
        if self._vision_ready:
            return

        model_path = self.cfg.model_path or "yolov8n.pt"

        # ✅ LOAD LIGHT MODEL + CPU OPT
        self._yolo = YOLO(model_path)
        self._yolo.fuse()   # 🔥 faster inference

        self._vision_ready = True

    # -------------------- PROCESS FRAME --------------------
    def process_frame(self, frame: np.ndarray, *, draw_camera_hint: bool = True) -> None:
        self.ensure_vision_resources()

        # 🔥 RUN ONLY EVERY OTHER FRAME (like face model)
        if self._process_this_frame:
            self._current_detections = []

            # 🔥 VERY IMPORTANT: reduce size
            small_frame = cv2.resize(frame, (256, 192))

            # 🔥 YOLO inference (FAST SETTINGS)
            results = self._yolo(
                small_frame,
                imgsz=256,
                conf=self.cfg.conf_threshold,
                verbose=False,
                device="cpu"
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

                    # 🔥 SCALE BACK
                    scale_x = frame.shape[1] / 256
                    scale_y = frame.shape[0] / 192

                    x1 = int(x1 * scale_x)
                    x2 = int(x2 * scale_x)
                    y1 = int(y1 * scale_y)
                    y2 = int(y2 * scale_y)

                    self._current_detections.append((x1, y1, x2, y2, label, conf))

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
                        metadata={
                            "label": self._last_label,
                            "confidence": max_conf,
                        },
                    )

                    if not self.speech.post_event(ev):
                        print("[Warning] Speech router not reachable.")

                    self._last_alert_time = now

                self._weapon_frame_count = 0

        # 🔁 TOGGLE FRAME (same as face model)
        self._process_this_frame = not self._process_this_frame

        # -------------------- DRAW ALWAYS --------------------
        for (x1, y1, x2, y2, label, conf) in self._current_detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

        if self._weapon_frame_count > 0:
            draw_alert(frame)

        if draw_camera_hint:
            cv2.putText(
                frame,
                "Cam: (q=quit)",
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

        cap = open_camera(camera_cycle[0])

        # 🔥 LOW RES CAMERA (BIG BOOST)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        print("🚀 Ultra Fast Weapon Detection Started")
=======
    window_name: str = "Weapon Detection System"

    _model: YOLO = field(init=False, repr=False)
    _current_weapon_data: list = field(default_factory=list, repr=False)
    
    # Threading and Throttling
    # Threading and Throttling
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
                
                # [CRITICAL FIX]: Send the network alert in the background thread!
                # This prevents the camera from freezing while talking to the Speech Server.
                self._executor.submit(self.speech.post_event, ev)
                
                print(f"[CRITICAL ALERT] {alert_message}")
                
                tracker.last_event_time = current_time

    def process_frame(self, frame: np.ndarray, *, draw_camera_hint: bool = True) -> None:
        current_time = time.time()
        frame_width = frame.shape[1]

        for tracker in self._trackers.values():
            tracker.is_active = False
>>>>>>> Stashed changes

        display_data = []

<<<<<<< Updated upstream
            self.process_frame(frame, draw_camera_hint=False)

            cv2.imshow(self.window_name, frame)
=======
        # 1. SMART SPATIAL TRACKING
        for (ox, oy, ow, oh), name, conf in self._current_weapon_data:
            cx, cy = ox + ow / 2, oy + oh / 2
            best_id = None
            min_dist = float("inf")
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
    speech = SpeechClient(base_url=cfg.router_url, timeout_s=0.2)
=======
    speech = SpeechClient(base_url=cfg.tts_router_url)
>>>>>>> Stashed changes
    return WeaponDetectionModel(cfg=cfg, speech=speech)