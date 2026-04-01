import threading
from typing import Callable, Dict, List


class EventBus:
    """A simple Pub/Sub message broker to keep models decoupled."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: Callable):
        """Models call this to listen for specific events."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: any = None):
        """The Voice Assistant calls this to broadcast an event."""
        with self._lock:
            if event_type not in self._subscribers:
                return
            callbacks = self._subscribers[event_type].copy()

        # Fire all callbacks in background threads so we don't block the publisher
        for callback in callbacks:
            threading.Thread(target=callback, args=(data,), daemon=True).start()


# Create a single global instance that all your models can import and share
shared_event_bus = EventBus()
