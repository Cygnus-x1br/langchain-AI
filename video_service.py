import os
import re
import json
import time
import uuid
import asyncio
import subprocess
from typing import Tuple, List, Dict, Any, Optional
import httpx

from config import (
    IMAGE_STORAGE_DIR,
    SERVER_PUBLIC_URL,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    get_now_local,
    sync_image_to_librechat
)
from image_service import save_image_locally

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Template de Sistema para Criação de Vídeos Curtos (Reels / Shorts / TikTok)
SHORT_VIDEO_DIRECTOR_PROMPT = """Você é um Diretor Executivo de Vídeos Curtos Verticais (Instagram Reels, YouTube Shorts, TikTok).
Sua missão é transformar a ideia, tema ou notícia do usuário em um roteiro dinâmico, envolvente e de alta retenção (gancho magnético nos primeiros 3s, ritmo acelerado, conclusão forte com CTA).

ESTRUTURA OBRIGATÓRIA (Duração total ideal: 10 a 25 segundos):
1. 'title': Título curto e impactante para o vídeo.
2. 'theme': Tema principal resumido.
3. 'total_duration_sec': Duração total estimada em segundos (ex: 12 a 20).
4. 'voice_tone': Tom da narração ('animado', 'misterioso', 'inspirador', 'didático', 'urgente').
5. 'narration_script': O texto EXATO que será falado pela voz ultra-realista em áudio (em português natural, sem rubricas ou parênteses, fluído e direto).
6. 'scenes': Lista de 2 a 4 cenas visuais sequenciais. Para cada cena forneça:
   - 'scene_id': Número da cena (1, 2, 3...)
   - 'duration_sec': Tempo de exibição da cena (ex: 4 a 6 segundos)
   - 'caption_text': Frase de legenda curta que ficará em destaque na tela (máximo 7 palavras)
   - 'visual_prompt_en': Prompt descritivo detalhado em INGLÊS para gerar o fundo visual fotorrealista daquela cena (estilo cinematográfico 9:16 vertical, iluminação dramática, alta qualidade, sem textos).

Responda ESTRITAMENTE em formato JSON válido:
{
  "title": "Título do Vídeo",
  "theme": "Tema do Vídeo",
  "total_duration_sec": 15,
  "voice_tone": "urgente",
  "narration_script": "Você sabia que um dia em Vênus é mais longo que um ano inteiro? Isso acontece porque o planeta gira tão devagar que leva 243 dias terrestres para dar uma volta sobre si mesmo. O universo é bizarro!",
  "scenes": [
    {
      "scene_id": 1,
      "duration_sec": 5,
      "caption_text": "Vênus é o planeta mais bizarro!",
      "visual_prompt_en": "Cinematic vertical 9:16 shot of planet Venus spinning in deep space, glowing atmosphere, stars, photorealistic 8k"
    },
    {
      "scene_id": 2,
      "duration_sec": 5,
      "caption_text": "1 dia lá dura mais que 1 ano!",
      "visual_prompt_en": "Dramatic cosmic view of planetary orbit with time lapse trails, surreal golden lighting, vertical 9:16"
    },
    {
      "scene_id": 3,
      "duration_sec": 5,
      "caption_text": "Siga para mais mistérios do cosmos!",
      "visual_prompt_en": "Glowing galaxy spiral with nebula dust, hyper realistic universe, vertical 9:16 mobile wallpaper aesthetic"
    }
  ]
}"""


async def plan_short_video_storyboard(user_prompt: str) -> Dict[str, Any]:
    """
    Gera o roteiro completo, narração e prompts de cenas para o vídeo curto utilizando OpenAI / LiteLLM.
    """
    url = "https://api.openai.com/v1/chat/completions" if OPENAI_API_KEY else f"{LITELLM_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY or LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }
    model = "gpt-4o-mini" if OPENAI_API_KEY else "ComboFree_Smart"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHORT_VIDEO_DIRECTOR_PROMPT},
            {"role": "user", "content": f"Crie um roteiro completo de vídeo curto vertical (Reels/Shorts/TikTok) para o seguinte tema:\n{user_prompt}"}
        ],
        "temperature": 0.6
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                raw_txt = resp.json()["choices"][0]["message"]["content"]
                # Extrai JSON
                match = re.search(r"\{.*\}", raw_txt, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
    except Exception as e:
        print(f"[Erro no Planejador de Vídeo Curto]: {e}")

    # Fallback seguro caso a IA não retorne JSON estrito
    return {
        "title": "Vídeo Curto Exclusivo",
        "theme": user_prompt[:60],
        "total_duration_sec": 12,
        "voice_tone": "dinâmico",
        "narration_script": f"Confira este conteúdo especial sobre {user_prompt[:80]}. Informações essenciais direto ao ponto para você!",
        "scenes": [
            {
                "scene_id": 1,
                "duration_sec": 6,
                "caption_text": user_prompt[:35],
                "visual_prompt_en": f"Cinematic vertical 9:16 scene about {user_prompt[:60]}, dramatic lighting, 4k ultra realistic, high detail"
            },
            {
                "scene_id": 2,
                "duration_sec": 6,
                "caption_text": "Compartilhe e siga para mais!",
                "visual_prompt_en": "Futuristic abstract background with vibrant colors, vertical 9:16 mobile framing, studio lighting"
            }
        ]
    }


async def generate_narration_audio(narration_text: str, voice: str = "nova") -> Optional[str]:
    """
    Gera o áudio de locução falado via OpenAI TTS oficial e salva como arquivo .mp3 temporário.
    """
    if not OPENAI_API_KEY:
        print("[Vídeo Curto] OPENAI_API_KEY ausente para TTS.")
        return None

    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "voice": voice if voice in ["nova", "alloy", "echo", "fable", "onyx", "shimmer"] else "nova",
        "input": narration_text,
        "response_format": "mp3",
        "speed": 1.05  # Ritmo ágil para vídeos curtos
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                audio_id = uuid.uuid4().hex[:6]
                audio_path = os.path.join(IMAGE_STORAGE_DIR, f"narration_{audio_id}.mp3")
                with open(audio_path, "wb") as f:
                    f.write(resp.content)
                return audio_path
    except Exception as e:
        print(f"[Erro TTS Vídeo Curto]: {e}")
    return None


async def generate_scene_image(prompt_en: str, scene_idx: int) -> Optional[str]:
    """
    Gera a imagem de fundo de uma cena específica e salva localmente.
    """
    # 1. Tenta gerar via LiteLLM com modelos configurados (Gemini Flash Lite ou OpenAI GPT Image ou Together SDXL)
    from config import LITELLM_BASE_URL, LITELLM_API_KEY
    url = f"{LITELLM_BASE_URL}/images/generations"
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json"
    }

    # Modelos candidatos: Fast (Gemini Flash Lite / SDXL) ou Pro (OpenAI GPT Image 2.5 Flare / Together FLUX)
    candidate_models = ["ImageGen_Fast", "ImageGen_Fast_SDXL", "ImageGen_Pro_OpenAI"]

    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "prompt": f"{prompt_en}, 9:16 vertical composition, mobile wallpaper aspect, dramatic lighting, photorealistic cinematic 8k, without text",
            "n": 1
        }
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json().get("data", [])
                    if data:
                        item = data[0]
                        uid = uuid.uuid4().hex[:6]
                        img_filename = f"scene_{scene_idx}_{uid}.png"
                        target_path = os.path.join(IMAGE_STORAGE_DIR, img_filename)
                        if "b64_json" in item:
                            import base64
                            b64_str = item["b64_json"].split(",")[-1]
                            with open(target_path, "wb") as f:
                                f.write(base64.b64decode(b64_str))
                            sync_image_to_librechat(target_path)
                            return target_path
                        elif "url" in item:
                            r_img = await client.get(item["url"])
                            if r_img.status_code == 200:
                                with open(target_path, "wb") as f:
                                    f.write(r_img.content)
                                sync_image_to_librechat(target_path)
                                return target_path
        except Exception as e:
            print(f"[Aviso Geração Cena {scene_idx} com {model_name}]: {e}")

    # Fallback: Cria uma imagem base gráfica vertical em Pillow com gradiente moderno
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (1080, 1920), color=(15, 23, 42))
        draw = ImageDraw.Draw(img)
        # Gradiente suave
        for y in range(1920):
            r = int(15 + (scene_idx * 25) * (y / 1920)) % 80
            g = int(23 + (50 * (1 - y / 1920))) % 100
            b = int(42 + (120 * (y / 1920))) % 180
            draw.line([(0, y), (1080, y)], fill=(r, g, b))
        fallback_path = os.path.join(IMAGE_STORAGE_DIR, f"fallback_scene_{scene_idx}_{uuid.uuid4().hex[:4]}.png")
        img.save(fallback_path)
        sync_image_to_librechat(fallback_path)
        return fallback_path
    except Exception:
        return None


def assemble_video_ffmpeg(
    scene_items: List[Dict[str, Any]],
    audio_path: Optional[str],
    output_filename: str
) -> Tuple[bool, str]:
    """
    Monta e renderiza o vídeo curto vertical (1080x1920) utilizando FFmpeg com:
    - Efeito de movimento dinâmico (Ken Burns / zoompan)
    - Legendas estilizadas com caixa de fundo
    - Trilha de locução sincronizada
    """
    output_path = os.path.join(IMAGE_STORAGE_DIR, output_filename)
    concat_txt_path = os.path.join(IMAGE_STORAGE_DIR, f"concat_{uuid.uuid4().hex[:6]}.txt")

    rendered_clips = []
    try:
        # 1. Renderiza cada cena individualmente em MP4 vertical de alta resolução
        for idx, scene in enumerate(scene_items):
            img_p = scene.get("image_path")
            dur = max(2, int(scene.get("duration_sec", 4)))
            caption = scene.get("caption_text", "").replace("'", "").replace("\"", "").replace(":", "-")

            clip_out = os.path.join(IMAGE_STORAGE_DIR, f"clip_{idx}_{uuid.uuid4().hex[:4]}.mp4")

            # Filtro de vídeo: Redimensiona para 1080x1920, aplica zoom suave (Ken Burns) e renderiza legenda com faixa escura
            vf_filter = (
                f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
                f"zoompan=z='min(zoom+0.0015,1.15)':d={dur*25}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920,"
                f"drawbox=y=ih-420:color=black@0.65:width=iw:height=180:t=fill,"
                f"drawtext=text='{caption}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=h-360:shadowcolor=black@0.8:shadowx=2:shadowy=2"
            )

            cmd_clip = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", img_p,
                "-t", str(dur),
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-r", "25",
                clip_out
            ]
            subprocess.run(cmd_clip, capture_output=True, text=True, check=True)
            rendered_clips.append(clip_out)

        # 2. Concatena os clips
        with open(concat_txt_path, "w", encoding="utf-8") as f:
            for c in rendered_clips:
                f.write(f"file '{c}'\n")

        # 3. Une vídeo concatenado com áudio da narração (ou áudio silencioso se ausente)
        cmd_final = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_txt_path
        ]

        if audio_path and os.path.exists(audio_path):
            cmd_final.extend([
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest"
            ])
        else:
            cmd_final.extend([
                "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest"
            ])

        cmd_final.append(output_path)
        subprocess.run(cmd_final, capture_output=True, text=True, check=True)

        sync_image_to_librechat(output_path)
        return True, output_path

    except Exception as e:
        print(f"[Erro FFmpeg Montagem de Vídeo]: {e}")
        return False, str(e)
    finally:
        # Limpeza de arquivos temporários
        if os.path.exists(concat_txt_path):
            try: os.remove(concat_txt_path)
            except: pass
        for c in rendered_clips:
            if os.path.exists(c):
                try: os.remove(c)
                except: pass


async def execute_short_video_flow(user_prompt: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pipeline mestre de orquestração do Vídeo Curto:
    1. Roteirização e storyboard estruturado (OpenAI).
    2. Síntese de áudio de locução falada (OpenAI TTS-1).
    3. Geração visual das cenas (IA Generativa de Imagens).
    4. Renderização e encoding vertical 9:16 (FFmpeg).
    5. Disponibilização pública e retorno Markdown rico no chat.
    """
    flow_start = time.time()
    tools_called = []

    print(f"[Vídeo Curto] Iniciando pipeline de vídeo vertical para: '{user_prompt[:80]}'...")

    # 1. Roteirização e Storyboard
    storyboard = await plan_short_video_storyboard(user_prompt)
    tools_called.append({
        "action": "short_video_storyboard",
        "title": storyboard.get("title"),
        "scenes_count": len(storyboard.get("scenes", [])),
        "total_duration_sec": storyboard.get("total_duration_sec", 15),
        "duration_seconds": round(time.time() - flow_start, 2)
    })

    narration_script = storyboard.get("narration_script", "")
    scenes = storyboard.get("scenes", [])

    # 2. Narração em Áudio (OpenAI TTS)
    t_audio_start = time.time()
    audio_path = await generate_narration_audio(narration_script)
    tools_called.append({
        "action": "short_video_tts",
        "voice": "nova",
        "provider": "OpenAI TTS-1",
        "status": "success" if audio_path else "fallback_muted",
        "duration_seconds": round(time.time() - t_audio_start, 2)
    })

    # 3. Geração das Cenas Visuais
    t_scenes_start = time.time()
    scene_items = []
    for sc in scenes:
        prompt_en = sc.get("visual_prompt_en", "")
        sc_id = sc.get("scene_id", 1)
        img_path = await generate_scene_image(prompt_en, sc_id)
        if img_path:
            scene_items.append({
                "scene_id": sc_id,
                "duration_sec": sc.get("duration_sec", 4),
                "caption_text": sc.get("caption_text", ""),
                "image_path": img_path
            })

    tools_called.append({
        "action": "short_video_visual_scenes",
        "scenes_rendered": len(scene_items),
        "duration_seconds": round(time.time() - t_scenes_start, 2)
    })

    # Se nenhuma imagem foi gerada, aborta
    if not scene_items:
        fail_msg = "⚠️ Não foi possível compor as cenas visuais para o vídeo curto no momento. Por favor, tente novamente."
        return fail_msg, tools_called

    # 4. Renderização Final do Vídeo com FFmpeg
    t_render_start = time.time()
    vid_uid = uuid.uuid4().hex[:6]
    vid_filename = f"short_{get_now_local().strftime('%Y%m%d_%H%M%S')}_{vid_uid}.mp4"
    success, result_out = assemble_video_ffmpeg(scene_items, audio_path, vid_filename)

    render_duration = round(time.time() - t_render_start, 2)
    total_duration = round(time.time() - flow_start, 2)

    if not success:
        fail_msg = f"⚠️ Ocorreu um erro na renderização do vídeo com FFmpeg: {result_out}"
        return fail_msg, tools_called

    public_video_url = f"{SERVER_PUBLIC_URL}/{vid_filename}?t={int(time.time())}"

    tools_called.append({
        "action": "short_video_assembly",
        "engine": "FFmpeg (H.264 / AAC 1080x1920 Vertical)",
        "duration_seconds": render_duration,
        "video_url": public_video_url,
        "status": "success"
    })

    # 5. Formatação da Resposta com Markdown e Player HTML5
    scenes_bullets = "\n".join([
        f"- **Cena {s['scene_id']}** ({s['duration_sec']}s): _{s['caption_text']}_"
        for s in scene_items
    ])

    markdown_response = f"""### 🎬 Vídeo Curto Vertical Gerado com Sucesso!

#### 📱 **{storyboard.get('title', 'Vídeo Curto')}**
- **Formato:** Vertical 9:16 (1080x1920) ideal para **Instagram Reels**, **TikTok** e **YouTube Shorts**
- **Duração Total:** ~{storyboard.get('total_duration_sec', 15)} segundos
- **Narração:** Voz Ultra-Realista OpenAI (*nova*)
- **Efeitos:** Movimento Dinâmico Ken Burns + Legendas de Retenção

---

<video width="320" height="568" controls style="border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); margin: 12px 0;">
  <source src="{public_video_url}" type="video/mp4">
  Seu navegador não suporta a tag de vídeo.
</video>

📥 **[Clique aqui para baixar o vídeo em MP4]({public_video_url})**

---

#### 🎙️ **Roteiro & Narração:**
> "{narration_script}"

#### 🎞️ **Cenas Produzidas:**
{scenes_bullets}

⏱️ *Tempo total de criação e renderização: {total_duration}s*"""

    return markdown_response, tools_called
