from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import face_recognition

from src.core.events import EventPriority, ModelEvent
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.face_recognition.config import FaceRecognitionConfig
from src.speech.client import SpeechClient
from src.core.event_bus import shared_event_bus  # <-- Imported Event Bus


@dataclass
class FaceTracker:
    """Helper class to track a person's movement over time."""

    history: deque = field(
        default_factory=lambda: deque(maxlen=20)  # Increased memory to 20 frames
    )
    last_announced_state: str = "none"
    last_event_time: float = 0.0
    is_active: bool = True
    has_announced_entrance: bool = (
        False  # NEW: Tracks if we've officially recognized them
    )


@dataclass
class FaceRecognitionModel:
    cfg: FaceRecognitionConfig
    speech: SpeechClient
    window_name: str = "Face Recognition"

    _vision_ready: bool = field(default=False, repr=False)
    _face_cascade: Any = field(default=None, repr=False)
    _current_face_data: list = field(default_factory=list, repr=False)

    # Face Database
    _known_face_encodings: list = field(default_factory=list, repr=False)
    _known_face_names: list = field(default_factory=list, repr=False)

    # Threading and Throttling
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1), repr=False
    )
    _is_recognizing: bool = field(default=False, repr=False)
    _last_recognition_time: float = field(default=0.0, repr=False)

    # Intelligent Spatial Tracking
    _trackers: dict[str, FaceTracker] = field(default_factory=dict, repr=False)

    # --- Event Bus Subscription ---
    def __post_init__(self):
        shared_event_bus.subscribe("voice_command", self._on_voice_command)

    def _on_voice_command(self, transcript: str):
        if "who" in transcript or "people" in transcript or "describe" in transcript:
            print("[Face Model] Intercepted relevant voice command!")
            response = self.describe_scene()
            self.speech.speak_text(response)

    def describe_scene(self) -> str:
        """Returns a natural-sounding, human-readable summary of who is currently in the frame."""
        active_people = []

        for name, tracker in self._trackers.items():
            # Only describe people who are actively in the frame and have been tracked
            if (
                tracker.is_active
                and tracker.has_announced_entrance
                and name != "Scanning..."
            ):

                # 1. Translate the technical state into conversational English
                state = tracker.last_announced_state
                if state in ["stationary", "entered", "none"]:
                    action = "standing nearby"
                elif state == "very_close":
                    action = "right in front of you"
                elif state == "approaching":
                    action = "approaching you"
                elif state == "leaving":
                    action = "walking away"
                else:
                    action = state

                # 2. Make "Unknown person" sound more natural
                display_name = (
                    "someone I don't recognize" if name == "Unknown person" else name
                )

                active_people.append(f"{display_name} {action}")

        # 3. Format the final sentence based on how many people are there
        if not active_people:
            return "I don't see anyone around right now."

        if len(active_people) == 1:
            return f"I see {active_people[0]}."

        if len(active_people) == 2:
            return f"I see {active_people[0]} and {active_people[1]}."

        # Oxford comma for 3 or more people
        return f"I see {', '.join(active_people[:-1])}, and {active_people[-1]}."

    # --- Core Vision ---
    def _load_known_faces(self) -> None:
        if not os.path.exists(self.cfg.face_db_path):
            print(f"[Warning] Face database path not found: {self.cfg.face_db_path}")
            return

        print("Loading known faces into memory...")
        for filename in os.listdir(self.cfg.face_db_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(self.cfg.face_db_path, filename)
                name = os.path.splitext(filename)[0]
                try:
                    image = face_recognition.load_image_file(filepath)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        self._known_face_encodings.append(encodings[0])
                        self._known_face_names.append(name.replace("_", " "))
                        print(f"Loaded: {name}")
                except Exception as e:
                    print(f"[Error] Failed to load {filename}: {e}")

    def ensure_vision_resources(self) -> None:
        if self._vision_ready:
            return
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._load_known_faces()
        self._vision_ready = True

    def _recognize_faces_worker(
        self, frame_copy: np.ndarray, faces: np.ndarray
    ) -> None:
        new_face_data = []
        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

        css_locations = []
        for x, y, w, h in faces:
            top, right = max(0, y - 20), min(rgb_frame.shape[1], x + w + 20)
            bottom, left = min(rgb_frame.shape[0], y + h + 20), max(0, x - 20)
            css_locations.append((top, right, bottom, left))

        try:
            encodings = face_recognition.face_encodings(
                rgb_frame, known_face_locations=css_locations
            )
        except Exception:
            encodings = []

        for i, (x, y, w, h) in enumerate(faces):
            name = "Unknown person"
            if i < len(encodings) and len(self._known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(
                    self._known_face_encodings, encodings[i]
                )
                if len(face_distances) > 0:
                    best_idx = np.argmin(face_distances)
                    if face_distances[best_idx] < 0.6:
                        name = self._known_face_names[best_idx]

            new_face_data.append(((x, y, w, h), name))

        self._current_face_data = new_face_data
        self._is_recognizing = False

    def _get_spatial_description(self, cx: float, frame_width: int) -> str:
        third = frame_width / 3
        if cx < third:
            return "on your left"
        elif cx > 2 * third:
            return "on your right"
        return "in front of you"

    def _analyze_and_announce_intent(self, frame_width: int):
        current_time = time.time()

        for name, tracker in list(self._trackers.items()):

            # 1. HANDLE EXPIRATION (Grace period of 2 seconds off-camera)
            if not tracker.is_active:
                if len(tracker.history) > 0 and (
                    current_time - tracker.history[-1][0] > 2.0
                ):
                    # Only announce "left" if we previously announced they "entered"
                    if tracker.has_announced_entrance and name not in [
                        "Scanning...",
                        "Unknown person",
                    ]:
                        ev = ModelEvent(
                            source="face_recognition",
                            type="person_left",
                            message=f"{name} left.",
                            priority=EventPriority.NORMAL,
                            dedupe_key=f"left:{name}",
                            cooldown_s=15.0,
                        )
                        self.speech.post_event(ev)
                    # Safe to delete them now
                    del self._trackers[name]
                continue  # Skip all other logic if they aren't in the current frame

            # 2. IGNORE FLICKERS & NOISE
            if name == "Scanning..." or len(tracker.history) < 10:
                continue

            # Get their most recent position
            newest_cx = tracker.history[-1][2]
            direction = self._get_spatial_description(newest_cx, frame_width)

            # 3. HANDLE ENTRANCE (Debounced)
            # We now only announce someone AFTER they've been solid for 10 frames
            if not tracker.has_announced_entrance:
                tracker.has_announced_entrance = True
                tracker.last_announced_state = "entered"

                if name != "Unknown person":  # Avoid spamming unknown entries
                    ev = ModelEvent(
                        source="face_recognition",
                        type="person_entered",
                        message=f"{name} appeared {direction}.",
                        priority=EventPriority.NORMAL,
                        dedupe_key=f"entered:{name}",
                        cooldown_s=10.0,
                    )
                    self.speech.post_event(ev)
                    tracker.last_event_time = current_time
                continue  # Skip intent analysis for the first few frames

            # 4. INTENT SMOOTHING (Median Filtering)
            # This mathematically kills the wild bounding box jumps
            history_list = list(tracker.history)

            # Sort the oldest 5 and newest 5 heights to find the stable "middle" value
            old_heights = sorted([h for _, h, _ in history_list[:5]])
            new_heights = sorted([h for _, h, _ in history_list[-5:]])

            oldest_h = old_heights[len(old_heights) // 2]
            newest_h = new_heights[len(new_heights) // 2]

            height_diff = newest_h - oldest_h

            # 5. HIGHER THRESHOLD (18% change required to trigger walking intent)
            threshold = max(45, oldest_h * 0.18)

            current_state = "stationary"
            if newest_h > 260:
                current_state = "very_close"
            elif height_diff > threshold:
                current_state = "approaching"
            elif height_diff < -threshold:
                current_state = "leaving"

            # 6. ANNOUNCE STATE CHANGES
            time_since_last_event = current_time - tracker.last_event_time
            state_changed = current_state != tracker.last_announced_state

            if (state_changed and time_since_last_event > 5.0) or (
                time_since_last_event > 15.0
            ):
                message = ""
                priority = EventPriority.NORMAL

                if current_state == "approaching":
                    message = f"{name} is approaching from {direction}."
                elif current_state == "leaving":
                    message = f"{name} is walking away."
                elif current_state == "very_close":
                    message = f"{name} is standing right {direction}."
                    priority = EventPriority.HIGH
                elif (
                    current_state == "stationary"
                    and tracker.last_announced_state == "entered"
                ):
                    message = f"{name} is standing {direction}."

                if message and name != "Unknown person":
                    ev = ModelEvent(
                        source="face_recognition",
                        type="person_intent",
                        message=message,
                        priority=priority,
                        dedupe_key=f"intent:{name}:{current_state}",
                        cooldown_s=5.0,
                    )
                    self.speech.post_event(ev)
                    tracker.last_announced_state = current_state
                    tracker.last_event_time = current_time

    def process_frame(
        self, frame: np.ndarray, *, draw_camera_hint: bool = True
    ) -> None:
        self.ensure_vision_resources()
        assert self._face_cascade is not None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        display_data = []
        current_time = time.time()
        frame_width = frame.shape[1]

        # Mark all trackers as inactive initially
        for tracker in self._trackers.values():
            tracker.is_active = False

        # 1. SMART TRACKING & DATA COLLECTION
        for x, y, w, h in faces:
            cx, cy = x + w / 2, y + h / 2
            best_name = "Scanning..."
            min_dist = float("inf")

            for (ox, oy, ow, oh), name in self._current_face_data:
                ocx, ocy = ox + ow / 2, oy + oh / 2
                spatial_dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
                if spatial_dist < (w * 1.5) and spatial_dist < min_dist:
                    min_dist = spatial_dist
                    best_name = name

            display_data.append(((x, y, w, h), best_name))

            if best_name not in self._trackers:
                self._trackers[best_name] = FaceTracker()

            self._trackers[best_name].is_active = True
            self._trackers[best_name].history.append((current_time, h, cx))

        # 2. RUN INTELLIGENT ANALYSIS
        self._analyze_and_announce_intent(frame_width)

        # 3. THROTTLE RECOGNITION
        if (
            not self._is_recognizing
            and len(faces) > 0
            and (current_time - self._last_recognition_time > 1.0)
        ):
            self._is_recognizing = True
            self._last_recognition_time = current_time
            self._executor.submit(self._recognize_faces_worker, frame.copy(), faces)

        # 4. DRAW OVERLAYS
        if draw_camera_hint:
            cv2.putText(
                frame,
                "Cam: (c=switch, q=quit)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        for (x, y, w, h), name in display_data:
            left, top, right, bottom = x, y, x + w, y + h
            tracker_state = (
                self._trackers.get(name, FaceTracker()).last_announced_state
                if name in self._trackers
                else ""
            )

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
            )
            cv2.putText(
                frame,
                f"{name} ({tracker_state})",
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

    def run(self, list_cameras: bool = False) -> None:
        if list_cameras:
            cams = probe_cameras(max_index=self.cfg.max_camera_index)
            print(
                "Available cameras:",
                ", ".join(map(str, cams)) if cams else "(none found)",
            )
            return

        try:
            if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass

        camera_cycle = probe_cameras(max_index=self.cfg.max_camera_index) or [
            self.cfg.camera_index
        ]
        if self.cfg.camera_index in camera_cycle:
            cam_pos = camera_cycle.index(self.cfg.camera_index)
        else:
            camera_cycle = [self.cfg.camera_index] + camera_cycle
            cam_pos = 0

        current_camera_index = camera_cycle[cam_pos]
        video_capture = open_camera(current_camera_index)

        print("Starting video stream...")

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


def build_default_face_model() -> FaceRecognitionModel:
    cfg = FaceRecognitionConfig()
    speech = SpeechClient(base_url=cfg.tts_router_url)
    return FaceRecognitionModel(cfg=cfg, speech=speech)
