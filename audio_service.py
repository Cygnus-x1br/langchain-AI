"""
Serviço de Transcrição e Reconhecimento de Fala (STT) com Cascata de Fallback.
Estratégia de Prioridade:
  Tier 1 (Zero Custo & Ultra Rápido): Groq Whisper (whisper-large-v3-turbo -> whisper-large-v3)
  Tier 2 (Gratuito / Google): Gemini Multimodal / Transcribe (gemini-3.5-transcribe / gemini-2.5-flash)
  Tier 3 (Fallback Oficial de Alta Resiliência): OpenAI Whisper (whisper-1)
"""

import os
import io
import time
import httpx
from typing import Optional, Tuple
from config import LITELLM_BASE_URL, LITELLM_API_KEY, ORCHESTRATOR_DEBUG

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Fallback models ordered by priority
AUDIO_CASCADE_TIERS = [
    # Tier 1: Groq Whisper (Gratuito, super rápido)
    {"provider": "groq", "model": "groq/whisper-large-v3-turbo"},
    {"provider": "groq", "model": "groq/whisper-large-v3"},
    # Tier 2: Google Gemini (Gratuito, ótima acurácia em PT-BR)
    {"provider": "gemini", "model": "gemini/gemini-3.5-transcribe"},
    # Tier 3: OpenAI (Oficial / Pago com alta disponibilidade)
    {"provider": "openai", "model": "openai/whisper-1"}
]


async def _transcribe_via_litellm(
    file_bytes: bytes,
    filename: str,
    model: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0
) -> Tuple[bool, str, dict]:
    """Tenta transcrever o arquivo de áudio utilizando o LiteLLM Proxy."""
    try:
        url = f"{LITELLM_BASE_URL.rstrip('/')}/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {LITELLM_API_KEY}"
        }
        
        # Monta os dados multipart/form-data
        data = {
            "model": model,
            "temperature": str(temperature)
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        files = {
            "file": (filename, file_bytes, "audio/wav")
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json.get("text", "").strip()
                return True, text, res_json
            else:
                if ORCHESTRATOR_DEBUG:
                    print(f"[STT LiteLLM {model}] Erro HTTP {response.status_code}: {response.text}")
                return False, "", {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        if ORCHESTRATOR_DEBUG:
            print(f"[STT LiteLLM {model}] Exceção: {e}")
        return False, "", {"error": str(e)}


async def _transcribe_via_groq_direct(
    file_bytes: bytes,
    filename: str,
    model: str = "whisper-large-v3-turbo",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0
) -> Tuple[bool, str, dict]:
    """Tenta transcrever o áudio diretamente na API da Groq Cloud caso o LiteLLM falhe."""
    if not GROQ_API_KEY:
        return False, "", {"error": "GROQ_API_KEY não configurada"}

    try:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        data = {
            "model": model,
            "temperature": str(temperature)
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        files = {
            "file": (filename, file_bytes, "audio/wav")
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json.get("text", "").strip()
                return True, text, res_json
            else:
                return False, "", {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return False, "", {"error": str(e)}


async def _transcribe_via_openai_direct(
    file_bytes: bytes,
    filename: str,
    model: str = "whisper-1",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0
) -> Tuple[bool, str, dict]:
    """Fallback direto na OpenAI API."""
    if not OPENAI_API_KEY:
        return False, "", {"error": "OPENAI_API_KEY não configurada"}

    try:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data = {
            "model": model,
            "temperature": str(temperature)
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        files = {
            "file": (filename, file_bytes, "audio/wav")
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=headers, data=data, files=files)
            if response.status_code == 200:
                res_json = response.json()
                text = res_json.get("text", "").strip()
                return True, text, res_json
            else:
                return False, "", {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return False, "", {"error": str(e)}


async def transcribe_audio_cascade(
    file_bytes: bytes,
    filename: str = "audio.wav",
    requested_model: Optional[str] = None,
    language: Optional[str] = "pt",
    prompt: Optional[str] = None,
    temperature: float = 0.0
) -> dict:
    """
    Executa a transcrição com cascata de fallback automática:
      1. Tenta Groq Turbo -> Groq Large via LiteLLM ou direto
      2. Tenta Google Gemini Transcribe
      3. Tenta OpenAI Whisper-1
    Retorna o dicionário no formato padrão OpenAI: {"text": str, ...}
    """
    start_time = time.time()
    errors = []

    # Se o cliente solicitou um modelo específico que não é genérico, tentamos primeiro
    candidates = list(AUDIO_CASCADE_TIERS)
    if requested_model and requested_model not in ("whisper-1", "default", "Audio_Transcribe", ""):
        # Se solicitou algo específico, prioriza
        if "groq" in requested_model or "turbo" in requested_model:
            candidates = [c for c in AUDIO_CASCADE_TIERS if c["provider"] == "groq"] + [c for c in AUDIO_CASCADE_TIERS if c["provider"] != "groq"]
        elif "gemini" in requested_model:
            candidates = [c for c in AUDIO_CASCADE_TIERS if c["provider"] == "gemini"] + [c for c in AUDIO_CASCADE_TIERS if c["provider"] != "gemini"]
        elif "openai" in requested_model or "whisper" in requested_model:
            candidates = [c for c in AUDIO_CASCADE_TIERS if c["provider"] == "openai"] + [c for c in AUDIO_CASCADE_TIERS if c["provider"] != "openai"]

    for tier in candidates:
        provider = tier["provider"]
        model = tier["model"]
        
        if ORCHESTRATOR_DEBUG:
            print(f"[STT Cascade] Tentando {provider} com modelo {model}...")

        # 1ª Tentativa: Via LiteLLM Proxy
        success, text, details = await _transcribe_via_litellm(
            file_bytes=file_bytes,
            filename=filename,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature
        )

        if success and (text or len(file_bytes) < 5000):
            duration = round(time.time() - start_time, 3)
            if ORCHESTRATOR_DEBUG:
                print(f"[STT Cascade SUCESSO] Provedor: {provider} ({model}) em {duration}s -> \"{text}\"")
            return {
                "text": text,
                "model_used": model,
                "provider": provider,
                "duration_seconds": duration,
                "details": details
            }

        # Se LiteLLM falhou para Groq, tenta direto com a API Groq
        if provider == "groq":
            groq_model = model.replace("groq/", "")
            succ_dir, text_dir, details_dir = await _transcribe_via_groq_direct(
                file_bytes=file_bytes,
                filename=filename,
                model=groq_model,
                language=language,
                prompt=prompt,
                temperature=temperature
            )
            if succ_dir and (text_dir or len(file_bytes) < 5000):
                duration = round(time.time() - start_time, 3)
                if ORCHESTRATOR_DEBUG:
                    print(f"[STT Groq Direto SUCESSO] Modelo {groq_model} em {duration}s -> \"{text_dir}\"")
                return {
                    "text": text_dir,
                    "model_used": groq_model,
                    "provider": "groq-direct",
                    "duration_seconds": duration,
                    "details": details_dir
                }
            errors.append(f"Groq Direct: {details_dir.get('error')}")

        # Se LiteLLM falhou para OpenAI, tenta direto com a API OpenAI
        if provider == "openai":
            succ_dir, text_dir, details_dir = await _transcribe_via_openai_direct(
                file_bytes=file_bytes,
                filename=filename,
                model="whisper-1",
                language=language,
                prompt=prompt,
                temperature=temperature
            )
            if succ_dir and (text_dir or len(file_bytes) < 5000):
                duration = round(time.time() - start_time, 3)
                if ORCHESTRATOR_DEBUG:
                    print(f"[STT OpenAI Direto SUCESSO] whisper-1 em {duration}s -> \"{text_dir}\"")
                return {
                    "text": text_dir,
                    "model_used": "whisper-1",
                    "provider": "openai-direct",
                    "duration_seconds": duration,
                    "details": details_dir
                }
            errors.append(f"OpenAI Direct: {details_dir.get('error')}")

        errors.append(f"{provider}/{model}: {details.get('error')}")

    # Se todos falharam
    duration = round(time.time() - start_time, 3)
    return {
        "text": "",
        "error": "Todos os provedores de transcrição em cascata falharam.",
        "details": errors,
        "duration_seconds": duration
    }


# ---------------------------------------------------------
# Síntese de Voz Ultra-Realista (TTS - Text to Speech)
# Compatível 100% com OpenAI POST /v1/audio/speech
# ---------------------------------------------------------
async def synthesize_speech_openai(
    text: str,
    voice: str = "nova",
    model: str = "tts-1",
    response_format: str = "mp3",
    speed: float = 1.0
) -> Tuple[bool, bytes, str]:
    """
    Gera áudio falado ultra-realista utilizando a API oficial da OpenAI.
    Suporta vozes: alloy, echo, fable, onyx, nova, shimmer.
    Retorna: (sucesso: bool, audio_bytes: bytes, content_type: str)
    """
    if not OPENAI_API_KEY:
        return False, b"", "OPENAI_API_KEY não configurada"

    # Vozes válidas da OpenAI
    valid_voices = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
    chosen_voice = voice.lower() if voice and voice.lower() in valid_voices else "nova"
    chosen_model = "tts-1-hd" if "hd" in (model or "").lower() else "tts-1"

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": chosen_model,
        "voice": chosen_voice,
        "input": text,
        "response_format": response_format or "mp3",
        "speed": float(speed) if speed else 1.0
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "audio/mpeg")
                return True, resp.content, content_type
            else:
                err_msg = f"HTTP {resp.status_code}: {resp.text}"
                print(f"[TTS OpenAI Erro]: {err_msg}")
                return False, b"", err_msg
    except Exception as e:
        print(f"[TTS OpenAI Exceção]: {e}")
        return False, b"", str(e)

