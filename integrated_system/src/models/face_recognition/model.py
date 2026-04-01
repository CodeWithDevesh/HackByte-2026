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
from ultralytics import YOLO  # <-- NEW: YOLO Import

from src.core.events import EventPriority, ModelEvent
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.face_recognition.config import FaceRecognitionConfig
from src.speech.client import SpeechClient
from src.core.event_bus import shared_event_bus


@dataclass
class PersonTracker:
    """Tracks a unique body ID provided by YOLO."""

    yolo_id: int
    name: str = "Scanning..."
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    last_announced_state: str = "none"
    last_event_time: float = 0.0
    is_active: bool = True
    has_announced_entrance: bool = False


@dataclass
class FaceRecognitionModel:
    cfg: FaceRecognitionConfig
    speech: SpeechClient
    window_name: str = "Unified Tracking"

    _vision_ready: bool = field(default=False, repr=False)
    _yolo_model: Any = field(default=None, repr=False)

    # Face Database
    _known_face_encodings: list = field(default_factory=list, repr=False)
    _known_face_names: list = field(default_factory=list, repr=False)

    # Threading and Throttling
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1), repr=False
    )
    _is_recognizing: bool = field(default=False, repr=False)

    # Trackers keyed by YOLO ID, not name!
    _trackers: dict[int, PersonTracker] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        shared_event_bus.subscribe("voice_command", self._on_voice_command)

    def _on_voice_command(self, transcript: str):
        if "who" in transcript or "people" in transcript or "describe" in transcript:
            print("[Vision Model] Intercepted relevant voice command!")
            response = self.describe_scene()
            self.speech.speak_text(response)

    def describe_scene(self) -> str:
        active_people = []
        for tracker in self._trackers.values():
            if (
                tracker.is_active
                and tracker.has_announced_entrance
                and tracker.name != "Scanning..."
            ):
                state = tracker.last_announced_state.replace("_", " ")

                # Natural language formatting
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

                display_name = (
                    "someone I don't recognize"
                    if tracker.name == "Unknown person"
                    else tracker.name
                )
                active_people.append(f"{display_name} {action}")

        if not active_people:
            return "I don't see anyone around right now."
        if len(active_people) == 1:
            return f"I see {active_people[0]}."
        if len(active_people) == 2:
            return f"I see {active_people[0]} and {active_people[1]}."

        return f"I see {', '.join(active_people[:-1])}, and {active_people[-1]}."

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

        # Initialize YOLOv8 Nano for extremely fast body tracking
        print("Loading YOLOv8 Nano...")
        self._yolo_model = YOLO("yolov8n.pt")
        self._load_known_faces()
        self._vision_ready = True

    def _recognize_faces_worker(self, pending_identifications: list) -> None:
        """
        Takes a list of tuples: (yolo_id, rgb_image_crop)
        """
        for yolo_id, face_crop_rgb in pending_identifications:
            # We only process if they are still tracked
            if yolo_id not in self._trackers:
                continue

            try:
                encodings = face_recognition.face_encodings(face_crop_rgb)

                if encodings and len(self._known_face_encodings) > 0:
                    face_distances = face_recognition.face_distance(
                        self._known_face_encodings, encodings[0]
                    )
                    best_idx = np.argmin(face_distances)

                    if face_distances[best_idx] < 0.6:
                        self._trackers[yolo_id].name = self._known_face_names[best_idx]
                    else:
                        self._trackers[yolo_id].name = "Unknown person"
                else:
                    # If we couldn't find a face in the crop (they turned around), leave as Scanning
                    # so we can try again on the next frame.
                    pass
            except Exception:
                pass

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

        for yolo_id, tracker in list(self._trackers.items()):
            name = tracker.name

            # 1. HANDLE EXPIRATION (Object Permanence)
            if not tracker.is_active:
                if len(tracker.history) > 0 and (
                    current_time - tracker.history[-1][0] > 10.0
                ):
                    # Silently remove them from memory after 10 seconds off-screen
                    del self._trackers[yolo_id]
                continue

            # 2. IGNORE FLICKERS
            if name == "Scanning..." or len(tracker.history) < 10:
                continue

            newest_cx = tracker.history[-1][2]
            direction = self._get_spatial_description(newest_cx, frame_width)

            # 3. HANDLE ENTRANCE
            if not tracker.has_announced_entrance:
                tracker.has_announced_entrance = True
                tracker.last_announced_state = "entered"

                if name != "Unknown person":
                    ev = ModelEvent(
                        source="vision_tracking",
                        type="person_entered",
                        message=f"{name} is here, {direction}.",
                        priority=EventPriority.NORMAL,
                        dedupe_key=f"entered:{name}",
                        cooldown_s=30.0,
                    )
                    self.speech.post_event(ev)
                    tracker.last_event_time = current_time
                continue

            # 4. INTENT SMOOTHING (Median Filtering)
            history_list = list(tracker.history)
            old_heights = sorted([h for _, h, _ in history_list[:5]])
            new_heights = sorted([h for _, h, _ in history_list[-5:]])

            oldest_h = old_heights[len(old_heights) // 2]
            newest_h = new_heights[len(new_heights) // 2]
            height_diff = newest_h - oldest_h

            # INCREASED THRESHOLD: Require a 20% size change (or 80 pixels)
            # This prevents minor body swaying from triggering a "walking away" alert.
            threshold = max(80, oldest_h * 0.20)

            current_state = "stationary"
            if newest_h > 400:
                current_state = "very_close"
            elif height_diff > threshold:
                current_state = "approaching"
            elif height_diff < -threshold:
                current_state = "leaving"

            # 5. STRICT ACTION-BASED ANNOUNCEMENTS
            time_since_last_event = current_time - tracker.last_event_time
            state_changed = current_state != tracker.last_announced_state

            # THE FIX: Removed the 15-second forced repeat timer.
            # It will NOW ONLY speak if their physical state actually changes.
            # (e.g., from "stationary" -> "leaving").
            # We keep the 5-second cooldown to prevent rapid toggling if they dance on the threshold.
            if state_changed and time_since_last_event > 5.0:
                message = ""
                priority = EventPriority.NORMAL

                if current_state == "approaching":
                    message = f"{name} is approaching."
                elif current_state == "leaving":
                    message = f"{name} is walking away."
                elif current_state == "very_close":
                    message = f"{name} is right in front of you."
                    priority = EventPriority.HIGH

                if message and name != "Unknown person":
                    ev = ModelEvent(
                        source="vision_tracking",
                        type="person_intent",
                        message=message,
                        priority=priority,
                        dedupe_key=f"intent:{name}:{current_state}",
                        # Push the global dedupe cooldown high so it strictly only says it once
                        cooldown_s=15.0,
                    )
                    self.speech.post_event(ev)

                    # Update the state so it knows it already announced this action
                    tracker.last_announced_state = current_state
                    tracker.last_event_time = current_time

    def process_frame(
        self, frame: np.ndarray, *, draw_camera_hint: bool = True
    ) -> None:
        self.ensure_vision_resources()

        # 1. RUN YOLO TRACKING
        # class=0 ensures it only looks for people. persist=True tells it to remember IDs between frames.
        results = self._yolo_model.track(
            frame, classes=[0], persist=True, verbose=False
        )

        current_time = time.time()
        frame_width = frame.shape[1]

        # Mark all trackers as inactive initially
        for tracker in self._trackers.values():
            tracker.is_active = False

        pending_identifications = []

        # 2. PROCESS TRACKED BODIES
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                cx = x1 + (w / 2)

                # Initialize tracker if we haven't seen this ID before
                if track_id not in self._trackers:
                    self._trackers[track_id] = PersonTracker(yolo_id=track_id)

                tracker = self._trackers[track_id]
                tracker.is_active = True
                tracker.history.append((current_time, h, cx))

                # Draw the body box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (255, 165, 0), cv2.FILLED)

                tracker_state = (
                    tracker.last_announced_state
                    if tracker.has_announced_entrance
                    else ""
                )
                label = f"{tracker.name} ({tracker_state})"
                cv2.putText(
                    frame,
                    label,
                    (x1 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                # If we don't know who this is yet, queue them for background Face Recognition
                if tracker.name == "Scanning...":
                    # THE FIX: Stop guessing where the head is!
                    # Pass the ENTIRE padded YOLO box to the background thread.
                    pad = 20
                    y1_pad = max(0, y1 - pad)
                    y2_pad = min(
                        frame.shape[0], y2 + pad
                    )  # Changed from 60% math to full y2
                    x1_pad = max(0, x1 - pad)
                    x2_pad = min(frame.shape[1], x2 + pad)

                    full_body_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    if full_body_crop.shape[0] > 0 and full_body_crop.shape[1] > 0:
                        rgb_crop = cv2.cvtColor(full_body_crop, cv2.COLOR_BGR2RGB)
                        pending_identifications.append((track_id, rgb_crop))

        # 3. RUN INTELLIGENT ANALYSIS
        self._analyze_and_announce_intent(frame_width)

        # 4. THROTTLE BACKGROUND RECOGNITION
        if not self._is_recognizing and len(pending_identifications) > 0:
            self._is_recognizing = True
            self._executor.submit(self._recognize_faces_worker, pending_identifications)

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

    def run(self, list_cameras: bool = False) -> None:
        if list_cameras:
            cams = probe_cameras(max_index=self.cfg.max_camera_index)
            print(
                "Available cameras:",
                ", ".join(map(str, cams)) if cams else "(none found)",
            )
            return

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
                    except Exception as e:
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
