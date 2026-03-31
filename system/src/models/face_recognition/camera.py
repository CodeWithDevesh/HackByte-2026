from __future__ import annotations

import cv2


def probe_cameras(max_index: int = 10) -> list[int]:
    """Return a list of camera indices that can be opened."""
    available: list[int] = []
    for idx in range(max_index + 1):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
        cap = cv2.VideoCapture(idx, backend)
        try:
            if cap is not None and cap.isOpened():
                available.append(idx)
        finally:
            try:
                cap.release()
            except Exception:
                pass
    return available


def open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {index}")
    return cap

