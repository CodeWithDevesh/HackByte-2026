from __future__ import annotations

import os
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.core.events import EventIngestResponse, ModelEvent, SpeakRequest
from src.speech.router import SpeechRouter
from src.speech.tts_piper import PiperTTS, default_piper_voice_config


app = FastAPI(title="Speech Router (Offline TTS)")

router = SpeechRouter()
_tts = None


def _playback_worker() -> None:
    global _tts
    tmp_wav = Path(os.getenv("SPEECH_TMP_WAV", "temp_speech.wav"))
    while True:
        event = router.get_next(timeout_s=0.25)
        if event is None:
            continue
        try:
            if _tts is None:
                _tts = PiperTTS(default_piper_voice_config())
            _tts.speak(event.message, tmp_wav=tmp_wav)
        except Exception as e:
            # Keep worker alive even if playback fails.
            print(f"[SpeechRouter] playback error: {e}")
        finally:
            try:
                if tmp_wav.exists():
                    tmp_wav.unlink()
            except Exception:
                pass


if os.getenv("SPEECH_DISABLE_PLAYBACK", "0") != "1":
    threading.Thread(target=_playback_worker, daemon=True).start()


@app.post("/event", response_model=EventIngestResponse)
def ingest_event(event: ModelEvent) -> EventIngestResponse:
    ok, reason = router.should_accept(event)
    if not ok:
        return EventIngestResponse(accepted=False, reason=reason, queued=False)

    router.enqueue(event)
    return EventIngestResponse(accepted=True, queued=True)


@app.post("/speak", response_model=EventIngestResponse)
def speak(req: SpeakRequest) -> EventIngestResponse:
    event = SpeechRouter.from_text(req.text, priority=req.priority)
    router.enqueue(event)
    return EventIngestResponse(accepted=True, queued=True)


def main() -> None:
    host = os.getenv("SPEECH_HOST", "0.0.0.0")
    port = int(os.getenv("SPEECH_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

