from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2

from src.core.events import EventPriority, ModelEvent
from src.models.face_recognition.camera import open_camera, probe_cameras
from src.models.face_recognition.config import FaceRecognitionConfig
from src.speech.client import SpeechClient


@dataclass
class FaceRecognitionModel:
    cfg: FaceRecognitionConfig
    speech: SpeechClient
    window_name: str = "Face Recognition"

    def run(self, list_cameras: bool = False) -> None:
        if list_cameras:
            cams = probe_cameras(max_index=self.cfg.max_camera_index)
            print("Available cameras:", ", ".join(map(str, cams)) if cams else "(none found)")
            return

        # Import DeepFace lazily to avoid TF logs for `--list-cameras` usage.
        from deepface import DeepFace

        close_announced: dict[str, bool] = {}

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Reduce noisy OpenCV logs (best effort).
        try:
            if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
                cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except Exception:
            pass

        camera_cycle = probe_cameras(max_index=self.cfg.max_camera_index) or [self.cfg.camera_index]
        if self.cfg.camera_index in camera_cycle:
            cam_pos = camera_cycle.index(self.cfg.camera_index)
        else:
            camera_cycle = [self.cfg.camera_index] + camera_cycle
            cam_pos = 0

        current_camera_index = camera_cycle[cam_pos]
        video_capture = open_camera(current_camera_index)

        process_this_frame = True
        current_face_data = []

        print("Starting video stream...")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            if process_this_frame:
                current_face_data = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
                )

                for (x, y, w, h) in faces:
                    y_min, y_max = max(0, y - 20), min(frame.shape[0], y + h + 20)
                    x_min, x_max = max(0, x - 20), min(frame.shape[1], x + w + 20)
                    face_crop = frame[y_min:y_max, x_min:x_max]

                    name = "Unknown"

                    if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        try:
                            dfs = DeepFace.find(
                                img_path=face_crop,
                                db_path=self.cfg.face_db_path,
                                model_name="Dlib",
                                enforce_detection=False,
                                silent=True,
                            )

                            if dfs and not dfs[0].empty:
                                identity_path = dfs[0]["identity"].iloc[0]
                                name = os.path.splitext(os.path.basename(identity_path))[0]
                        except Exception:
                            pass

                    # Smart Distance & Speech Logic
                    if h > 250:
                        dist_text = "very close"
                        if name != "Unknown" and not close_announced.get(name, False):
                            # Send structured event to speech router (dedupe/cooldown handled there)
                            ev = ModelEvent(
                                source="face_recognition",
                                type="person_distance",
                                message=f"{name} is very close",
                                priority=EventPriority.HIGH,
                                dedupe_key=f"face:{name}:very_close",
                                cooldown_s=10.0,
                                metadata={"distance_bucket": "very_close"},
                            )
                            if not self.speech.post_event(ev):
                                print("[Warning] Speech router not reachable.")
                            close_announced[name] = True

                    elif h > 150:
                        dist_text = "a few meters away"

                    else:
                        dist_text = "far away"
                        if name != "Unknown":
                            close_announced[name] = False

                    current_face_data.append(((x, y, w, h), name, dist_text))

            process_this_frame = not process_this_frame

            cv2.putText(
                frame,
                f"Cam: {current_camera_index} (c=switch, q=quit)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            for (x, y, w, h), name, dist in current_face_data:
                left, top, right, bottom = x, y, x + w, y + h

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(
                    frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
                )
                cv2.putText(
                    frame,
                    f"{name}: {dist}",
                    (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
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

        video_capture.release()
        cv2.destroyAllWindows()


def build_default_face_model() -> FaceRecognitionModel:
    cfg = FaceRecognitionConfig()
    speech = SpeechClient(base_url=cfg.tts_router_url)
    return FaceRecognitionModel(cfg=cfg, speech=speech)

