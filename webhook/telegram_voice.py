"""
Telegram Voice Message Handler

Processes voice messages from Telegram:
1. Download voice file from Telegram
2. Normalize audio (WAV 16kHz mono)
3. Transcribe to text (STT)
4. Process through agent
5. Send text response back

Integrates Phase 8 (audio) with Phase 7 (Telegram transport).

Observability:
- Unified trace propagation from transport → agent → LangSmith
- Each voice message gets a unique trace_id
- Transport-level events (download, STT) are first spans in trace
- All downstream nodes use the same trace_id for complete lineage
"""

import logging
import io
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
import httpx

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, BackgroundTasks

from config import Config
from transport.telegram.transport import (
    create_telegram_transport,
    NormalizedMessage,
)
from services.audio import (
    get_audio_normalizer,
    get_stt_processor,
)
from agent.tracing import TraceMetadata
from webhook.telegram import TelegramUpdate

# Setup logging
logger = logging.getLogger(__name__)

# Create router
voice_router = APIRouter(prefix="/webhook", tags=["voice"])


def clean_agent_response(text: str) -> str:
    """
    Strip chain-of-thought / reasoning from the agent's raw output.

    The phi model sometimes wraps reasoning in <think>...</think> tags,
    or emits "Let me think..." preambles before the actual answer.
    We keep only the final answer portion.
    """
    import re

    if not text:
        return text

    # 1. Remove <think>...</think> blocks (including multi-line)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

    # 2. Remove <|im_start|> / <|im_end|> tokens
    text = re.sub(r'<\|im_start\|>\w*\n?', '', text)
    text = re.sub(r'<\|im_end\|>', '', text)

    # 3. If "Answer:" or "Response:" header exists, keep only what follows
    for marker in (r'Answer:', r'Response:', r'Final answer:', r'My answer:'):
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[match.end():]
            break

    # 4. Strip leading filler phrases
    filler_patterns = [
        r'^(Sure[,!.]?\s*)', r'^(Of course[,!.]?\s*)', r'^(Certainly[,!.]?\s*)',
        r'^(Let me (think|explain|help)[^.]*\.\s*)', r'^(Here\'s my (answer|response)[^:]*:\s*)',
        r'^(I\'d be (happy|glad) to[^.]*\.\s*)',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    return text.strip()


def text_to_ogg_voice(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Convert text to OGG Opus audio using gTTS + ffmpeg.

    Returns raw OGG bytes, or None if synthesis fails.
    """
    import tempfile
    import os
    import subprocess

    try:
        from gtts import gTTS
    except ImportError:
        logger.warning("gTTS not installed, cannot synthesize voice")
        return None

    try:
        # Generate MP3 from text
        tts = gTTS(text=text, lang=lang, slow=False)
        tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp_mp3.name)
        tmp_mp3.close()

        # Convert MP3 → OGG Opus with ffmpeg
        tmp_ogg_path = tmp_mp3.name.replace(".mp3", ".ogg")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3.name,
                 "-c:a", "libopus", "-b:a", "32k", tmp_ogg_path],
                check=True, capture_output=True,
            )
            with open(tmp_ogg_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_mp3.name)
            if os.path.exists(tmp_ogg_path):
                os.unlink(tmp_ogg_path)

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}", exc_info=True)
        return None


# Telegram voice message structure
class TelegramVoice(BaseModel):
    """Voice object from Telegram."""
    file_id: str
    file_unique_id: str
    duration: int  # seconds
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramVoiceMessage(BaseModel):
    """Message containing voice from Telegram."""
    message_id: int
    date: int
    chat_id: int
    user_id: int
    voice: TelegramVoice
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    caption: Optional[str] = None


class VoiceMessageResult(BaseModel):
    """Result of voice message processing."""
    message_id: int
    transcribed_text: str
    confidence: Optional[float] = None
    processing_time_ms: float
    status: str  # "success", "error", "skipped"
    error: Optional[str] = None


class TelegramVoiceTransport:
    """
    Telegram voice transport layer.
    
    Handles:
    - Downloading voice files from Telegram
    - Normalizing audio
    - Transcribing with STT
    - Sending responses
    - Trace propagation for observability
    """
    
    def __init__(self):
        """
        Initialize with Telegram transport, services, and tracer.
        
        Tracer is injected for unified observability from transport → agent.
        """
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.api_url = f"https://api.telegram.org/bot{self.token}"
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")
        
        # Initialize audio services
        self.normalizer = get_audio_normalizer()
        self.stt_processor = get_stt_processor()
        
        # 🔍 INITIALIZE TRACER FOR OBSERVABILITY
        # Import here to avoid circular dependency
        from agent.tracing import tracer_factory
        try:
            self.tracer = tracer_factory.create_tracer()
        except Exception as e:
            logger.warning(f"Failed to initialize tracer: {e}. Using no-op tracer.")
            from agent.tracing import NoOpTracer
            self.tracer = NoOpTracer()
        
        # HTTP client for file downloads
        self.http_client = None
    
    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        return self.http_client
    
    async def download_voice_file(self, file_id: str) -> Optional[bytes]:
        """
        Download voice file from Telegram.
        
        Args:
            file_id: Telegram file_id from voice message
            
        Returns:
            Raw audio bytes, or None if download failed
            
        Raises:
            HTTPException: If download fails
        """
        try:
            client = await self._get_http_client()
            
            # Step 1: Get file path from Telegram
            file_info_url = f"{self.api_url}/getFile"
            file_info_response = await client.get(
                file_info_url,
                params={"file_id": file_id},
                timeout=30,
            )
            
            if file_info_response.status_code != 200:
                logger.error(
                    f"Failed to get file info for {file_id}: "
                    f"{file_info_response.text}"
                )
                return None
            
            file_info = file_info_response.json()
            if not file_info.get("ok"):
                logger.error(f"Telegram error: {file_info.get('description')}")
                return None
            
            file_path = file_info["result"]["file_path"]
            
            # Step 2: Download the actual file
            download_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            file_response = await client.get(download_url, timeout=60)
            
            if file_response.status_code != 200:
                logger.error(
                    f"Failed to download file {file_id}: "
                    f"{file_response.status_code}"
                )
                return None
            
            return file_response.content
        
        except Exception as e:
            logger.error(f"Error downloading voice file: {str(e)}", exc_info=True)
            return None
    
    async def process_voice_message(
        self,
        file_id: str,
        message_id: int,
        duration: int,
    ) -> VoiceMessageResult:
        """
        Process voice message: download → normalize → transcribe.
        
        Args:
            file_id: Telegram file_id
            message_id: Telegram message_id (for tracking)
            duration: Voice duration in seconds
            
        Returns:
            VoiceMessageResult with transcribed text
        """
        import time
        start_time = time.time()
        
        try:
            # Skip very short/long messages
            if duration < 1:
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text="",
                    status="skipped",
                    error="Voice message too short (< 1s)",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            if duration > 300:  # 5 minutes
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text="",
                    status="skipped",
                    error="Voice message too long (> 5 min)",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Step 1: Download voice file
            logger.debug(f"Downloading voice file {file_id}...")
            voice_bytes = await self.download_voice_file(file_id)
            
            if not voice_bytes:
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text="",
                    status="error",
                    error="Failed to download voice file",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            logger.debug(
                f"Downloaded {len(voice_bytes)} bytes for voice message {message_id}"
            )
            
            # Step 2: Normalize audio
            logger.debug("Normalizing audio...")
            try:
                # Telegram voice is OGG format
                normalized = self.normalizer.normalize_bytes(voice_bytes, "ogg")
                logger.debug(
                    f"Normalized audio: {len(normalized.data)} bytes, "
                    f"{normalized.duration_seconds:.1f}s"
                )
            except Exception as e:
                logger.error(f"Audio normalization failed: {str(e)}")
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text="",
                    status="error",
                    error=f"Audio normalization failed: {str(e)}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Step 3: Transcribe audio (STT)
            logger.debug("Transcribing audio...")
            try:
                stt_result = self.stt_processor.transcribe(normalized.data)
                
                if not stt_result.text:
                    return VoiceMessageResult(
                        message_id=message_id,
                        transcribed_text="",
                        status="skipped",
                        error="No speech detected in audio",
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )
                
                logger.info(
                    f"Transcribed voice message {message_id}: "
                    f"{stt_result.text[:100]}"
                )
                
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text=stt_result.text,
                    confidence=stt_result.confidence,
                    status="success",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
            
            except Exception as e:
                logger.error(f"Transcription failed: {str(e)}")
                return VoiceMessageResult(
                    message_id=message_id,
                    transcribed_text="",
                    status="error",
                    error=f"Transcription failed: {str(e)}",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )
        
        except Exception as e:
            logger.error(
                f"Unexpected error processing voice message: {str(e)}",
                exc_info=True,
            )
            return VoiceMessageResult(
                message_id=message_id,
                transcribed_text="",
                status="error",
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    async def send_message(self, chat_id: int, text: str) -> bool:
        """
        Send text response via Telegram API.
        
        Args:
            chat_id: Telegram chat ID
            text: Message text to send
            
        Returns:
            True if successful
        """
        try:
            client = await self._get_http_client()
            
            response = await client.post(
                f"{self.api_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=10,
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send message: {response.text}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}", exc_info=True)
            return False
    
    async def send_voice_message(self, chat_id: int, audio_bytes: bytes, caption: Optional[str] = None) -> bool:
        """
        Send a voice message (OGG Opus) via Telegram sendVoice API.

        Args:
            chat_id: Telegram chat ID
            audio_bytes: OGG Opus encoded audio
            caption: Optional text caption

        Returns:
            True if successful
        """
        try:
            client = await self._get_http_client()
            data = {"chat_id": str(chat_id)}
            if caption:
                data["caption"] = caption[:1024]
            files = {"voice": ("voice.ogg", audio_bytes, "audio/ogg")}
            response = await client.post(
                f"{self.api_url}/sendVoice",
                data=data,
                files=files,
                timeout=30,
            )
            if response.status_code != 200:
                logger.error(f"Failed to send voice: {response.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error sending voice message: {str(e)}", exc_info=True)
            return False

    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()


# Storage for transport (initialized once)
_voice_transport = None


def get_voice_transport() -> TelegramVoiceTransport:
    """Get or create Telegram voice transport (singleton)."""
    global _voice_transport
    if _voice_transport is None:
        _voice_transport = TelegramVoiceTransport()
    return _voice_transport


@voice_router.post("/telegram/voice")
async def handle_voice_message(
    update: TelegramUpdate,
    background_tasks: BackgroundTasks,
):
    """
    Handle incoming Telegram voice message.
    
    Observability:
    - Generate trace_id at transport layer (FIRST STEP)
    - Emit transport-level event (FIRST SPAN)
    - Pass trace_id to background task
    - Agent uses same trace_id (UNIFIED TRACE)
    
    Workflow:
    1. Generate trace_id (unified observability)
    2. Download voice file from Telegram
    3. Normalize audio to 16kHz mono WAV
    4. Transcribe using Whisper STT
    5. Process transcribed text through agent (async background task)
    6. Send response back to user
    
    Args:
        update: Telegram update with voice message
        background_tasks: FastAPI background tasks
        
    Returns:
        {"status": "ok"} immediately
        Response will be sent asynchronously after processing
    """
    try:
        # Validate we have a message
        if not update.message:
            logger.warning(f"Update {update.update_id} has no message")
            return {"status": "ok"}  # Still return 200 to Telegram
        
        message = update.message
        
        # Only handle voice messages
        if not message.voice:
            logger.debug(f"Message {message.message_id} has no voice, skipping")
            return {"status": "ok"}
        
        transport = get_voice_transport()
        
        # 🔍 GENERATE TRACE_ID AT TRANSPORT LAYER
        trace_id = str(uuid4())
        chat_id = message.chat.id
        user_id = message.from_.id
        conversation_id = f"telegram:{chat_id}"
        
        # 📊 CREATE TRANSPORT METADATA
        transport_metadata = {
            "platform": "telegram",
            "modality": "voice",
            "telegram_chat_id": chat_id,
            "telegram_user_id": user_id,
            "telegram_user_name": message.from_.username or message.from_.first_name,
            "voice_duration": message.voice.duration,
            "voice_file_id": message.voice.file_id,
        }
        
        # 📝 EMIT TRANSPORT-LEVEL EVENT (FIRST SPAN)
        trace_metadata = TraceMetadata(
            trace_id=trace_id,
            conversation_id=conversation_id,
            user_id=str(user_id),
        )
        
        if hasattr(transport, 'tracer') and transport.tracer:
            transport.tracer.record_event(
                name="telegram_voice_received",
                metadata=transport_metadata,
                trace_metadata=trace_metadata,
            )
        
        print(f"[VOICE] Received voice from user {user_id} chat {chat_id}, duration={message.voice.duration}s, file_id={message.voice.file_id[:20]}... [trace={trace_id[:8]}]")
        logger.info(
            f"Received voice message from user {user_id} "
            f"(chat {chat_id}): {message.voice.duration}s "
            f"[trace_id={trace_id}]"
        )
        
        # Process voice asynchronously (don't block webhook response)
        # PASS trace_id to background task for unified propagation
        background_tasks.add_task(
            _process_voice_async,
            message_id=message.message_id,
            chat_id=chat_id,
            user_id=user_id,
            file_id=message.voice.file_id,
            duration=message.voice.duration,
            user_name=message.from_.username or message.from_.first_name,
            trace_id=trace_id,
            conversation_id=conversation_id,
        )
        
        # Return immediately to Telegram
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Error handling voice message: {str(e)}", exc_info=True)
        # Still return 200 to acknowledge receipt
        return {"status": "ok"}


async def _process_voice_async(
    message_id: int,
    chat_id: int,
    user_id: int,
    file_id: str,
    duration: int,
    user_name: Optional[str] = None,
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
):
    """
    Background task to process voice message.
    
    Observability:
    - Receives trace_id from transport layer
    - Uses trace_id for all downstream operations
    - Passes trace_id to agent.invoke() for unified tracing
    
    After transcription, integrates with agent and sends response.
    """
    transport = get_voice_transport()
    
    # Use provided trace_id or generate new one
    if not trace_id:
        trace_id = str(uuid4())
    if not conversation_id:
        conversation_id = f"telegram:{chat_id}"
    
    print(f"[VOICE TASK] Starting processing for message {message_id}, file_id={file_id[:20]}...")
    try:
        # Process voice (download + normalize + transcribe)
        result = await transport.process_voice_message(
            file_id=file_id,
            message_id=message_id,
            duration=duration,
        )
        print(f"[VOICE TASK] process_voice_message result: status={result.status}, text={repr(result.transcribed_text)}, error={result.error}")
        
        logger.info(
            f"Voice processing result for message {message_id}: "
            f"status={result.status}, text_len={len(result.transcribed_text or '')} "
            f"[trace_id={trace_id}]"
        )
        
        # Send status message
        if result.status == "success" and result.transcribed_text:
            # Default fallback: show transcribed text
            confidence_str = f" ({result.confidence:.1%})" if result.confidence is not None else ""
            response_msg = f"🎤 You said: {result.transcribed_text}{confidence_str}"

            # Invoke agent with transcribed text
            try:
                from agent.orchestrator import SAMOrchestrator
                agent = SAMOrchestrator()
                agent_result = await agent.invoke(
                    raw_input=result.transcribed_text,
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                )
                raw_response = agent_result.get("output") or agent_result.get("final_output") or ""
                response_msg = clean_agent_response(raw_response) or response_msg
                print(f"[VOICE TASK] Agent response (cleaned): {response_msg[:120]}")
            except Exception as agent_error:
                logger.error(f"Agent invocation failed: {agent_error}", exc_info=True)

            # --- Reply with VOICE (input was voice, reply with voice) ---
            lines = [l for l in response_msg.splitlines() if l.strip()]
            use_voice_reply = True  # always reply with voice when input was voice

            if use_voice_reply:
                import asyncio
                loop = asyncio.get_event_loop()
                ogg_bytes = await loop.run_in_executor(None, text_to_ogg_voice, response_msg)
                if ogg_bytes:
                    sent = await transport.send_voice_message(chat_id, ogg_bytes)
                    if sent:
                        print(f"[VOICE TASK] Sent voice reply to chat {chat_id}")
                        return
                    else:
                        print(f"[VOICE TASK] Voice send failed, falling back to text")
                else:
                    print(f"[VOICE TASK] TTS synthesis failed, falling back to text")

            # Fallback: send as text
            await transport.send_message(chat_id, response_msg)

        elif result.status == "success" and not result.transcribed_text:
            response_msg = "⚠️ Transcription succeeded but no text was detected"
        elif result.status == "skipped":
            response_msg = f"⏭️ Skipped: {result.error}"
        else:
            response_msg = f"❌ Error: {result.error}"
        
        # Send response
        await transport.send_message(chat_id, response_msg)
        logger.info(f"Sent response to chat {chat_id} [trace_id={trace_id}]")
    
    except Exception as e:
        import traceback
        print(f"[VOICE TASK ERROR] message_id={message_id}: {type(e).__name__}: {e}")
        print(traceback.format_exc())
        logger.error(
            f"Error in voice processing task: {str(e)}", exc_info=True
        )
        try:
            transport = get_voice_transport()
            await transport.send_message(
                chat_id,
                f"❌ Processing failed: {type(e).__name__}: {str(e)}"
            )
        except Exception as send_error:
            print(f"[VOICE TASK ERROR] Failed to send error: {send_error}")
            logger.error(f"Failed to send error response: {str(send_error)}")


@voice_router.get("/telegram/voice/health")
async def voice_handler_health():
    """Health check for voice handler."""
    try:
        transport = get_voice_transport()
        return {
            "status": "ok",
            "voice_handler": "ready",
            "audio_normalizer": "active" if transport.normalizer else "inactive",
            "stt_processor": "active" if transport.stt_processor else "inactive",
        }
    except Exception as e:
        logger.error(f"Voice handler health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
