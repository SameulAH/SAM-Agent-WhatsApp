"""
Coqui XTTS v2 TTS backend.

Model:   tts_models/multilingual/multi-dataset/xtts_v2
Install: pip install TTS

Features:
- Neural voice synthesis (far higher quality than gTTS / glow-tts)
- 17 languages, multilingual in a single model
- Voice cloning: pass a ≥6-second WAV speaker sample via XTTS_SPEAKER_WAV
- Built-in speaker fallback when no sample is provided ("Ana Florence")
- Lazy model loading (first call only; ~1-2 GB download on first run)

Environment variables:
  XTTS_SPEAKER_WAV  Path to a WAV file used for voice cloning (optional)
  XTTS_LANGUAGE     BCP-47 language code, e.g. "en", "pt", "ar" (default: "en")
"""

import os
import tempfile
import subprocess
from typing import Optional
from .base import TTSBackend, TTSRequest, TTSResponse

try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# XTTS v2 model identifier
_XTTS_V2_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
# Built-in speaker used when no WAV sample is supplied
_DEFAULT_SPEAKER = "Ana Florence"


class CoquiTTSBackend(TTSBackend):
    """
    Coqui XTTS v2 TTS backend.

    Produces high-quality, natural-sounding speech locally.
    Supports voice cloning via a speaker WAV file, with a built-in
    speaker as the default fallback.

    Requires: pip install TTS
    Model is downloaded once (~1.8 GB) and cached locally.
    """

    def __init__(
        self,
        model_name: str = _XTTS_V2_MODEL,
        device: str = "cpu",
        speaker_wav: Optional[str] = None,
        language: str = "en",
    ):
        """
        Initialise the XTTS v2 backend.

        Args:
            model_name:  TTS model ID — defaults to XTTS v2.
            device:      "cpu" or "cuda".
            speaker_wav: Path to a reference WAV for voice cloning.
                         Falls back to XTTS_SPEAKER_WAV env var, then
                         to the built-in speaker if neither is set.
            language:    BCP-47 language code (default "en").
        """
        if not COQUI_AVAILABLE:
            raise ImportError(
                "TTS package not installed. Run: pip install TTS"
            )

        self.model_name = model_name
        self.device = device
        self.language = language or os.getenv("XTTS_LANGUAGE", "en")
        # Resolve speaker WAV: constructor arg → env var → None (use built-in)
        self.speaker_wav = (
            speaker_wav
            or os.getenv("XTTS_SPEAKER_WAV")
            or None
        )
        self._model: Optional[TTS] = None

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load XTTS v2 on the first synthesis call."""
        if self._model is None:
            self._model = TTS(
                model_name=self.model_name,
                gpu=(self.device == "cuda"),
                progress_bar=False,
            )

    def _wav_to_ogg(self, wav_path: str) -> bytes:
        """
        Convert a WAV file to OGG Opus using ffmpeg.

        Returns raw OGG bytes.  Raises on ffmpeg failure.
        """
        ogg_path = wav_path.replace(".wav", ".ogg")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-c:a", "libopus", "-b:a", "32k", ogg_path,
            ],
            check=True,
            capture_output=True,
        )
        with open(ogg_path, "rb") as f:
            return f.read()

    # ── Public API ─────────────────────────────────────────────────────────

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """
        Synthesize text to OGG Opus audio using XTTS v2.

        XTTS v2 accepts either:
          • speaker_wav  — a reference WAV for voice cloning, OR
          • speaker      — a named built-in speaker ("Ana Florence" etc.)

        The method prefers speaker_wav when available; otherwise it falls
        back to the built-in speaker so synthesis always succeeds.

        Args:
            request: TTSRequest with text and optional language override.

        Returns:
            TTSResponse with OGG audio_data on success, or typed error.
        """
        if not request.text:
            return TTSResponse(
                status="recoverable_error",
                error_type="invalid_text",
                metadata={"backend": "coqui_xtts_v2", "model": self.model_name,
                           "trace_id": request.trace_id},
            )

        try:
            self._load_model()

            language = request.language or self.language

            with tempfile.TemporaryDirectory() as tmp_dir:
                wav_path = os.path.join(tmp_dir, "tts_output.wav")

                if self.speaker_wav and os.path.isfile(self.speaker_wav):
                    # Voice-cloning mode
                    self._model.tts_to_file(
                        text=request.text,
                        file_path=wav_path,
                        speaker_wav=self.speaker_wav,
                        language=language,
                    )
                else:
                    # Built-in speaker mode
                    self._model.tts_to_file(
                        text=request.text,
                        file_path=wav_path,
                        speaker=_DEFAULT_SPEAKER,
                        language=language,
                    )

                # Convert WAV → OGG Opus (Telegram requires Opus in OGG container)
                try:
                    ogg_bytes = self._wav_to_ogg(wav_path)
                    audio_data = ogg_bytes
                    audio_format = "ogg"
                except Exception:
                    # ffmpeg not available — return raw WAV
                    with open(wav_path, "rb") as f:
                        audio_data = f.read()
                    audio_format = "wav"

            if not audio_data:
                return TTSResponse(
                    status="recoverable_error",
                    error_type="invalid_text",
                    metadata={"backend": "coqui_xtts_v2", "model": self.model_name,
                               "trace_id": request.trace_id},
                )

            return TTSResponse(
                status="success",
                audio_data=audio_data,
                audio_format=audio_format,
                metadata={
                    "backend": "coqui_xtts_v2",
                    "model": self.model_name,
                    "language": language,
                    "voice_cloning": bool(self.speaker_wav),
                    "trace_id": request.trace_id,
                    "text_length": len(request.text),
                },
            )

        except Exception as e:
            return TTSResponse(
                status="fatal_error",
                error_type="backend_unavailable",
                metadata={
                    "backend": "coqui_xtts_v2",
                    "model": self.model_name,
                    "error": str(e),
                    "trace_id": request.trace_id,
                },
            )
