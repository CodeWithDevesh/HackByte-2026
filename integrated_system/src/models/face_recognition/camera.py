from __future__ import annotations

import sys

import cv2


def _capture_backends() -> list[int]:
    """
    Platform-appropriate capture API order.
    Windows: default backend was often failing; DirectShow (DSHOW) is most reliable for USB webcams.
    Linux: V4L2 when available.
    """
    backends: list[int] = []
    if sys.platform == "win32":
        for attr in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            if hasattr(cv2, attr):
                b = int(getattr(cv2, attr))
                if b not in backends:
                    backends.append(b)
        if 0 not in backends:
            backends.append(0)
        return backends
    if hasattr(cv2, "CAP_V4L2"):
        backends.append(int(cv2.CAP_V4L2))
    if 0 not in backends:
        backends.append(0)
    return backends


def _try_open(index: int, backend: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def open_camera(index: int) -> cv2.VideoCapture:
    tried: list[int] = []
    for api in _capture_backends():
        tried.append(api)
        cap = _try_open(index, api)
        if cap is not None:
            return cap
    hint = ""
    if sys.platform == "win32":
        hint = (
            " On Windows, allow camera access for Python in Settings > Privacy, "
            "close other apps using the camera, or try a different CAMERA_INDEX."
        )
    raise RuntimeError(
        f"Could not open camera index {index} (tried backends {tried}).{hint}"
    )


def probe_cameras(max_index: int = 10) -> list[int]:
    """Return camera indices that can be opened (tries each platform backend per index)."""
    available: list[int] = []
    backends = _capture_backends()
    for idx in range(max_index + 1):
        opened = False
        for api in backends:
            cap = _try_open(idx, api)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
                opened = True
                break
        if opened:
            available.append(idx)
    return available
