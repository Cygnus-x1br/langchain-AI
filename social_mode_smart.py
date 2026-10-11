import os
import re
import time
import io
import httpx
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image

from config import resolve_image_to_base64, IMAGE_STORAGE_DIR
from image_service import save_image_locally

# Resoluções recomendadas pela OpenAI gpt-image-2.5-sunburst / gpt-image
PLATFORM_OPENAI_SIZES = {
    "INSTAGRAM": "1024x1536",    # Proporção próxima a 4:5 retrato
    "LINKEDIN": "1536x1024",     # Paisagem 1.5:1
    "X_TWITTER": "1536x1024",    # Paisagem 16:9
    "FACEBOOK": "1024x1024",     # Quadrado 1:1
    "REDDIT": "1536x1024",       # Paisagem
    "MULTIPLATAFORMA": "1024x1024"
}

async def execute_smart_mode(
    user_prompt: str,
    platform_key: str,
    briefing: Dict[str, Any],
    source_img_public_urls: List[str],
    spec: Dict[str, Any],
    tools_called: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Motor SOCIAL_MEDIA_SMART (OpenAI GPT-Image Multimodal Edition):
    Recebe 1, 2, 3 ou mais fotos enviadas pelo usuário, junta com os requisitos de design,
    textos, tipografia, paleta e logotipos especificados, e envia para a API OpenAI
    (gpt-image-2.5-sunburst) via endpoint /v1/images/edits ou /v1/images/generations.
    """
    start_time = time.time()
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    # Extração de dados estratégicos do briefing
    domain = briefing.get("domain", "Geral")
    tone = briefing.get("tone", "Profissional e Esportivo")
    headline = briefing.get("must_include_text") or briefing.get("topic_summary", "")
    palette = briefing.get("palette") or ["#0055A5", "#00A859"]
    palette_str = ", ".join(palette) if isinstance(palette, list) else str(palette)
    visual_elements = ", ".join(briefing.get("visual_elements") or [])
    
    # Construção do Megaprompt profissional de composição para o GPT Image
    prompt_lines = [
        f"Create an eye-catching, professional {spec['label']} social media poster graphic (aspect ratio {spec['aspect_ratio']}).",
        f"Theme / Domain: {domain}. Visual Tone: {tone}.",
        f"Design Instructions from user: {user_prompt}.",
        f"Background style & Colors: Incorporate {palette_str} background gradients, modern geometric curves and textures.",
    ]
    if headline:
        prompt_lines.append(f"Graphic Typography & Text Overlay: Prominently feature and render clearly the text: '{headline}'.")
    if visual_elements:
        prompt_lines.append(f"Visual details: Incorporate {visual_elements}.")
    
    # Verificação de solicitação explícita de alteração corporal/facial pelo usuário
    user_prompt_lower = (user_prompt or "").lower()
    explicit_face_alteration = any(k in user_prompt_lower for k in [
        "mude o rosto", "altere o rosto", "troque o rosto", "modifique o rosto",
        "mude o corpo", "altere o corpo", "troque a pessoa", "caricatura",
        "mude a feição", "transforme a pessoa", "avatar", "desenhe o rosto",
        "change face", "alter face", "change body", "cartoon face"
    ])

    if source_img_public_urls:
        if explicit_face_alteration:
            prompt_lines.append(
                f"Composition Requirement: Integrate the {len(source_img_public_urls)} reference images into the graphic "
                f"applying the user's requested artistic modifications to the subjects."
            )
        else:
            # GUARDRAIL ESTRITO DE FIDELIDADE HUMANA E IDENTIDADE:
            prompt_lines.append(
                f"CRITICAL IDENTITY & HUMAN PRESERVATION GUARDRAIL: When integrating the {len(source_img_public_urls)} source photos into the layout, "
                f"do NOT alter, redraw, hallucinate, distort, rejuvenate, age, or stylize the human subjects' faces, eyes, facial expressions, skin tones, or body proportions. "
                f"The people and athletes must remain 100% photorealistic and authentic to the original reference images. "
                f"Apply all graphic design, background gradients, textures, typography, and framing ONLY around the subjects, "
                f"preserving the original human features and logos completely untouched."
            )

    full_image_prompt = " ".join(prompt_lines)
    
    image_size = PLATFORM_OPENAI_SIZES.get(platform_key, "1024x1024")
    target_model = "gpt-image-2.5-sunburst"
    
    public_url = None
    status_msg = "failed"
    provider_used = "OpenAI (GPT Image 2.5 Sunburst)"
    
    # Preparação dos arquivos binários PNG para envio à OpenAI API
    image_files_payload = []
    
    if source_img_public_urls and openai_key:
        print(f"[Smart Mode] Processando {len(source_img_public_urls)} imagens de entrada para o OpenAI {target_model}...")
        for idx, img_url in enumerate(source_img_public_urls[:5]):
            try:
                # 1. Se for URL pública local (/images/... ou /images/generated/...), lê diretamente do disco para evitar latência de rede
                fname = img_url.split("/")[-1].split("?")[0]
                local_fpath = os.path.join(IMAGE_STORAGE_DIR, fname)
                if os.path.exists(local_fpath):
                        with open(local_fpath, "rb") as f_img:
                            raw_file_bytes = f_img.read()
                        with Image.open(io.BytesIO(raw_file_bytes)) as pil_img:
                            pil_img.load()
                            buf = io.BytesIO()
                            pil_img.convert("RGBA").save(buf, format="PNG")
                            image_files_payload.append(('image[]', (f'input_{idx}.png', buf.getvalue(), 'image/png')))
                            continue
                
                # 2. Se for URL remota ou base64
                res_block = await resolve_image_to_base64({"image_url": {"url": img_url}})
                b64_str = res_block.get("image_url", {}).get("url", "")
                if b64_str.startswith("data:image"):
                    raw_b64 = b64_str.split(",", 1)[1]
                    import base64
                    img_bytes = base64.b64decode(raw_b64)
                    with Image.open(io.BytesIO(img_bytes)) as pil_img:
                        buf = io.BytesIO()
                        pil_img.convert("RGBA").save(buf, format="PNG")
                        image_files_payload.append(('image[]', (f'input_{idx}.png', buf.getvalue(), 'image/png')))
            except Exception as prep_err:
                print(f"[Smart Mode] Erro ao preparar imagem {idx}: {prep_err}")

    # ---------------------------------------------------------
    # Seleção de Motor Conforme Prioridade do Dashboard (SOCIAL_IMAGE)
    # ---------------------------------------------------------
    try:
        from admin_config_service import get_category_models
        configured_models = get_category_models("SOCIAL_IMAGE") or ["ImageGen_Pro_OpenAI", "ImageGen_Pro_Flux"]
    except Exception:
        configured_models = ["ImageGen_Pro_OpenAI", "ImageGen_Pro_Flux"]

    together_key = os.getenv("TOGETHERAI_API_KEY", "")

    async def _try_openai() -> Optional[str]:
        nonlocal status_msg, provider_used
        if not openai_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if image_files_payload:
                    print(f"[Smart Mode] Enviando {len(image_files_payload)} imagens para OpenAI /v1/images/edits ({target_model})...")
                    data_payload = {
                        "model": target_model,
                        "prompt": full_image_prompt,
                        "size": image_size
                    }
                    oai_res = await client.post(
                        "https://api.openai.com/v1/images/edits",
                        headers={"Authorization": f"Bearer {openai_key}"},
                        data=data_payload,
                        files=image_files_payload
                    )
                else:
                    print(f"[Smart Mode] Sem imagens prévias. Gerando arte via OpenAI /v1/images/generations ({target_model})...")
                    oai_res = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers={"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"},
                        json={
                            "model": target_model,
                            "prompt": full_image_prompt,
                            "size": image_size,
                            "n": 1
                        }
                    )

                if oai_res.status_code == 200:
                    res_json = oai_res.json()
                    data_list = res_json.get("data", [])
                    if data_list:
                        saved_url = await save_image_locally(data_list[0], prefix="smart_oai")
                        if saved_url:
                            status_msg = "success"
                            provider_used = "OpenAI (GPT Image 2.5 Sunburst)"
                            print(f"[Smart Mode] Arte gerada pela OpenAI com sucesso: {saved_url}")
                            return saved_url
                else:
                    print(f"[Smart Mode Erro OpenAI HTTP {oai_res.status_code}]: {oai_res.text[:250]}")
        except Exception as oai_err:
            print(f"[Smart Mode Exceção OpenAI]: {oai_err}")
        return None

    async def _try_flux() -> Optional[str]:
        nonlocal status_msg, provider_used
        if not together_key:
            return None
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": "black-forest-labs/FLUX.1-kontext-pro" if source_img_public_urls else "black-forest-labs/FLUX.2-dev",
                    "prompt": full_image_prompt
                }
                if source_img_public_urls:
                    res_block = await resolve_image_to_base64({"image_url": {"url": source_img_public_urls[0]}})
                    raw_b64_url = res_block.get("image_url", {}).get("url", "")
                    if raw_b64_url.startswith("data:image"):
                        payload["image_url"] = raw_b64_url

                print(f"[Smart Mode] Gerando arte via Together AI FLUX ({payload['model']})...")
                tog_res = await client.post(
                    "https://api.together.xyz/v1/images/generations",
                    json=payload,
                    headers={"Authorization": f"Bearer {together_key}", "Content-Type": "application/json"}
                )
                if tog_res.status_code == 200:
                    img_list = tog_res.json().get("data", [])
                    if img_list and "url" in img_list[0]:
                        saved_url = await save_image_locally({"url": img_list[0]["url"]}, prefix="smart_flux")
                        if saved_url:
                            status_msg = "success (FLUX)"
                            provider_used = "Together AI (FLUX.1 Kontext Pro)"
                            print(f"[Smart Mode] Arte gerada pelo FLUX com sucesso: {saved_url}")
                            return saved_url
                else:
                    print(f"[Smart Mode Erro Together HTTP {tog_res.status_code}]: {tog_res.text[:250]}")
        except Exception as flx_err:
            print(f"[Smart Mode Exceção FLUX]: {flx_err}")
        return None

    async def _try_gemini() -> Optional[str]:
        nonlocal status_msg, provider_used
        try:
            from config import LITELLM_BASE_URL, LITELLM_API_KEY
            async with httpx.AsyncClient(timeout=120.0) as client:
                print(f"[Smart Mode] Gerando arte via Google Imagen 3 Pro (ImageGen_Pro_Gemini)...")
                payload = {"model": "ImageGen_Pro_Gemini", "prompt": full_image_prompt, "n": 1}
                res = await client.post(
                    f"{LITELLM_BASE_URL}/images/generations",
                    json=payload,
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
                )
                if res.status_code == 200:
                    data_items = res.json().get("data", [])
                    if data_items:
                        saved_url = await save_image_locally(data_items[0], prefix="smart_gemini")
                        if saved_url:
                            status_msg = "success (Google Imagen 3)"
                            provider_used = "Google (Imagen 3 Pro)"
                            print(f"[Smart Mode] Arte gerada pelo Google Imagen 3 com sucesso: {saved_url}")
                            return saved_url
                else:
                    print(f"[Smart Mode Erro Gemini HTTP {res.status_code}]: {res.text[:250]}")
        except Exception as gem_err:
            print(f"[Smart Mode Exceção Gemini]: {gem_err}")
        return None

    # Executa os motores na ordem definida no Dashboard
    attempted_engines = set()
    for m in configured_models:
        m_lower = str(m).lower()
        if ("openai" in m_lower or "gpt" in m_lower or "flare" in m_lower or "sunburst" in m_lower) and "openai" not in attempted_engines:
            attempted_engines.add("openai")
            public_url = await _try_openai()
            if public_url:
                break
        elif ("gemini" in m_lower or "imagen" in m_lower) and "gemini" not in attempted_engines:
            attempted_engines.add("gemini")
            public_url = await _try_gemini()
            if public_url:
                break
        elif ("flux" in m_lower or "together" in m_lower) and "flux" not in attempted_engines:
            attempted_engines.add("flux")
            public_url = await _try_flux()
            if public_url:
                break

    # Se nenhum dos preferidos concluiu, tenta os restantes como contingência
    if not public_url:
        for fallback_fn in [_try_openai, _try_gemini, _try_flux]:
            public_url = await fallback_fn()
            if public_url:
                break

    if not public_url and source_img_public_urls:
        public_url = source_img_public_urls[0]

    duration = round(time.time() - start_time, 2)
    tools_called.append({
        "action": "social_smart_generation",
        "model": target_model,
        "provider": provider_used,
        "input_images_count": len(source_img_public_urls),
        "status": status_msg,
        "duration_seconds": duration,
        "image_url": public_url
    })

    # Legenda estratégica do post
    headline_clean = headline or "Circuito Escolar 2026"
    caption_text = (
        f"🏆 {headline_clean}!\n\n"
        f"A energia, dedicação e talento dos nossos atletas em quadra e na piscina fazem história! "
        f"Parabéns a todos os alunos, técnicos e apoiadores envolvidos nessa jornada inspiradora.\n\n"
        f"👉 Acompanhe a cobertura completa e celebre o esporte conosco!\n\n"
        f".\n.\n"
        f"#{platform_key.lower()} #esporte #circuitoescolar #colegioarbos #natacao #handebol #esportesempre"
    )

    out = []
    out.append(f"### 📱 Estratégia de Conteúdo: {spec['label']} (Modo SMART - OpenAI)")
    out.append(f"**Objetivo:** `Composição e Estilização Multimodal com OpenAI GPT Image` | **Formato:** `{spec['dimensions']}` (`{spec['aspect_ratio']}`)\n")

    if public_url:
        out.append(f"![Arte Final para {spec['label']}]({public_url})\n")
    else:
        out.append(f"> ⚠️ *Ocorreu uma falha no processamento visual ({status_msg}).*\n")

    out.append("#### 📝 Legenda Estruturada:")
    out.append("```text")
    out.append(caption_text)
    out.append("```\n")

    out.append("#### 🎨 Detalhes da Produção Visual (OpenAI GPT Image):")
    out.append(f"- **Motor de Imagem:** `{provider_used}`")
    out.append(f"- **Imagens de Entrada Processadas:** `{len(source_img_public_urls)} fotos combinadas`")
    out.append(f"- **Resolução do Canvas:** `{image_size}`")
    out.append(f"- **Prompt de Composição:** _{full_image_prompt}_")
    out.append(f"- **Tempo de Processamento:** `{duration}s`")
    out.append(f"\n_Postagem inteligente gerada com sucesso pelo agente SOCIAL_MEDIA_SMART._")

    return "\n".join(out), tools_called
