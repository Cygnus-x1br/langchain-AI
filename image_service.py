import os
import re
import time
import uuid
import base64
import io
import httpx
from typing import Tuple, List, Dict, Any, Optional
from PIL import Image

from config import (
    IMAGE_STORAGE_DIR,
    SERVER_PUBLIC_URL,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    get_now_local,
    resolve_image_to_base64,
    sync_image_to_librechat
)
from phrase_extractor_guardrail import extract_phrase_intent, is_instruction_command, is_no_text_intent

_pro_provider_turn = 0

async def save_image_locally(data_item: dict, prefix: str = "img") -> str:
    """
    Decodifica base64 ou faz download da URL remota temporária (xAI/Together),
    salvando a imagem permanentemente em IMAGE_STORAGE_DIR e retornando a URL pública local.
    """
    timestamp = get_now_local().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    filename = f"{prefix}_{timestamp}_{uid}.png"
    target_path = os.path.join(IMAGE_STORAGE_DIR, filename)
    public_url = f"{SERVER_PUBLIC_URL}/{filename}?t={int(time.time())}"

    b64_data = data_item.get("b64_json")
    remote_url = data_item.get("url")

    if b64_data:
        try:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_data)
            with open(target_path, "wb") as f:
                f.write(raw_bytes)
            sync_image_to_librechat(target_path)
            print(f"[Imagem Salva] Base64 decodificado e salvo em: {target_path} ({len(raw_bytes)} bytes)")
            return public_url
        except Exception as e:
            print(f"[Erro Salvar Imagem Base64]: {e}")

    if remote_url:
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                resp = await client.get(remote_url)
                if resp.status_code == 200:
                    with open(target_path, "wb") as f:
                        f.write(resp.content)
                    sync_image_to_librechat(target_path)
                    print(f"[Imagem Baixada] Salva localmente em: {target_path} ({len(resp.content)} bytes)")
                    return public_url
                else:
                    print(f"[Aviso Download Imagem] HTTP {resp.status_code}. Retornando URL remota original.")
                    return remote_url
        except Exception as e:
            print(f"[Erro Download Imagem Remota]: {e}. Retornando URL remota original.")
            return remote_url

    return None

def detect_requested_model_in_prompt(user_prompt: str) -> Optional[Tuple[str, str]]:
    """
    Detecta se o usuário solicitou explicitamente um modelo de imagem no prompt.
    Retorna (provider_code, display_label).
    """
    p_low = user_prompt.lower()
    if any(k in p_low for k in ["sunburst", "gpt-image", "gpt-2.5", "gpt 2.5", "openai", "flare", "dall-e", "dalle"]):
        return ("openai", "OpenAI (GPT Image 2.5 Sunburst)")
    if any(k in p_low for k in ["flux", "kontext", "together", "black forest"]):
        return ("flux", "Together AI (FLUX.1-kontext-pro)")
    if any(k in p_low for k in ["gemini", "imagen", "flash lite"]):
        return ("gemini", "Google Gemini (Flash Lite Image)")
    if any(k in p_low for k in ["grok", "xai"]):
        return ("grok", "xAI (Grok Imagine 2.0)")
    return None

def detect_image_context_from_history(messages: list) -> Dict[str, Any]:
    """
    Percorre o histórico de mensagens para identificar:
    - O modelo que gerou a imagem anterior (OpenAI gpt-image-2.5-sunburst, Together FLUX, Gemini, Pillow)
    - A URL pública e o filename da imagem anterior
    - Se é uma composição Pillow com arquivo base disponível
    """
    ctx = {
        "original_creator": None,
        "image_url": None,
        "filename": None,
        "is_composed": False,
        "raw_base_filename": None,
        "creator_label": "Desconhecido"
    }
    if not messages:
        return ctx

    for msg in reversed(messages):
        content = msg.get("content", "")
        # Se for resposta do assistente (geralmente contém a URL e detalhes do motor)
        if isinstance(content, str):
            m_img = re.search(r"!\[.*?\]\((https?://[^\s\)]+)\)", content)
            if m_img:
                img_url = m_img.group(1)
                ctx["image_url"] = img_url
                fname = img_url.split("/")[-1].split("?")[0]
                ctx["filename"] = fname
                if fname.startswith("composed_"):
                    ctx["is_composed"] = True
                    ctx["raw_base_filename"] = fname.replace("composed_", "", 1)

                c_low = content.lower()
                if "smart_oai" in fname or "sunburst" in c_low or "gpt image" in c_low or ("openai" in c_low and "provedor" in c_low):
                    ctx["original_creator"] = "openai"
                    ctx["creator_label"] = "OpenAI (GPT Image 2.5 Sunburst)"
                elif "flux" in c_low or "together" in c_low:
                    ctx["original_creator"] = "flux"
                    ctx["creator_label"] = "Together AI (FLUX.1-kontext-pro)"
                elif "gemini" in c_low or "imagen" in c_low or "flash lite" in c_low:
                    ctx["original_creator"] = "gemini"
                    ctx["creator_label"] = "Google Gemini (Flash Lite Image)"
                elif ctx["is_composed"] or "pillow" in c_low:
                    ctx["original_creator"] = "pillow"
                    ctx["creator_label"] = "Composição Gráfica Pillow"
                return ctx

        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                    url = part.get("image_url", {}).get("url", "")
                    if url:
                        ctx["image_url"] = url
                        fname = url.split("/")[-1].split("?")[0]
                        ctx["filename"] = fname
                        if fname.startswith("composed_"):
                            ctx["is_composed"] = True
                            ctx["raw_base_filename"] = fname.replace("composed_", "", 1)
                        if "smart_oai" in fname:
                            ctx["original_creator"] = "openai"
                            ctx["creator_label"] = "OpenAI (GPT Image 2.5 Sunburst)"
                        return ctx
    return ctx

async def _extract_png_bytes(source_block: dict, resolved_block: dict) -> Tuple[Optional[bytes], str]:
    """Extrai bytes PNG e URL da imagem (seja local, base64 ou remota)."""
    url = ""
    if isinstance(source_block, dict):
        url = source_block.get("image_url", {}).get("url") or source_block.get("url") or ""

    # 1. Se for arquivo local em nosso diretório de armazenamento
    if url and ("images/generated" in url or "/images/" in url):
        fname = url.split("/")[-1].split("?")[0]
        local_path = os.path.join(IMAGE_STORAGE_DIR, fname)
        if os.path.exists(local_path):
            try:
                with Image.open(local_path) as pil_img:
                    buf = io.BytesIO()
                    pil_img.convert("RGBA").save(buf, format="PNG")
                    return buf.getvalue(), url
            except Exception as e:
                print(f"[image_service] Erro ao ler imagem local {local_path}: {e}")

    # 2. Se for base64 ou URL remota
    raw_b64 = resolved_block.get("image_url", {}).get("url", "")
    if raw_b64.startswith("data:image"):
        try:
            b64_clean = raw_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_clean)
            with Image.open(io.BytesIO(img_bytes)) as pil_img:
                buf = io.BytesIO()
                pil_img.convert("RGBA").save(buf, format="PNG")
                return buf.getvalue(), url or raw_b64[:30]
        except Exception as e:
            print(f"[image_service] Erro ao decodificar base64 para PNG: {e}")

    return None, url

async def execute_image_flow(category: str, user_prompt: str, last_user_msg: dict, messages: list) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executa a geração ou edição de imagem multi-provedor com balanceamento de carga e fallbacks.
    Tiers suportados:
      - ImageGen_Fast (Gemini Flash Lite - Grátis) com failover para ImageGen_Fast_SDXL (Together - $0.002)
      - ImageGen_Pro (Grok Imagine 2.0 via xAI), ImageGen_Pro_Flux (FLUX.2-dev via Together - $0.015) e ImageGen_Pro_OpenAI (GPT Image 2.5 Flare via OpenAI - $0.013)
      - ImageEdit (OpenAI GPT Image 2.5 Sunburst / FLUX.1 Kontext Pro / Gemini Flash Lite Image) para edição multimodal/inpainting
    """
    global _pro_provider_turn
    flow_start = time.time()
    tools_called = []

    # -------------------------------------------------------------
    # 1. FLUXO: EDIÇÃO DE IMAGEM (IMAGE_EDIT)
    # -------------------------------------------------------------
    if category == "IMAGE_EDIT":
        print(f"[Fluxo Imagem] Executando IMAGE_EDIT...")
        source_image_block = None

        # Procura imagem na mensagem atual
        content = last_user_msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                    source_image_block = part
                    break

        # Se não estiver na atual, procura no histórico de mensagens
        if not source_image_block:
            for msg in reversed(messages[:-1]):
                m_content = msg.get("content")
                if isinstance(m_content, list):
                    for part in m_content:
                        if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                            source_image_block = part
                            break
                    if source_image_block:
                        break
                elif isinstance(m_content, str):
                    m_match = re.search(r"!\[.*?\]\((http[^\)]+)\)", m_content)
                    if m_match:
                        source_image_block = {"type": "image_url", "image_url": {"url": m_match.group(1)}}
                        break

        if source_image_block:
            # Converte a imagem para base64 data URL
            resolved_block = await resolve_image_to_base64(source_image_block)
            raw_b64_url = resolved_block.get("image_url", {}).get("url", "")
            png_bytes, source_url = await _extract_png_bytes(source_image_block, resolved_block)

            # Análise semântica e determinística da intenção do usuário
            phrase_intent = extract_phrase_intent(user_prompt)
            req_model_info = detect_requested_model_in_prompt(user_prompt)
            history_ctx = detect_image_context_from_history(messages)

            # Caso especial ultra-rápido: restauração autêntica de foto se for Composed Pillow e pediu tirar tudo
            if phrase_intent.get("remove_all_text") and history_ctx.get("is_composed") and history_ctx.get("raw_base_filename"):
                raw_base_path = os.path.join(IMAGE_STORAGE_DIR, history_ctx["raw_base_filename"])
                if os.path.exists(raw_base_path):
                    clean_url = f"{SERVER_PUBLIC_URL}/{history_ctx['raw_base_filename']}?t={int(time.time())}"
                    duration = round(time.time() - flow_start, 2)
                    md_clean = (
                        f"![Foto Original Restaurada]({clean_url})\n\n"
                        f"✨ **Textos removidos com sucesso!**\n\n"
                        f"- **Ação:** Restauração instantânea da fotografia original limpa (100% autêntica, sem degradação por IA)\n"
                        f"- **Tempo de Processamento:** `{duration}s`\n"
                        f"- **Instrução:** _{user_prompt}_\n"
                        f"- **Resultado:** Foto preservada com os sujeitos e enquadramento originais sem nenhum texto gráfico sobreposto."
                    )
                    tools_called.append({
                        "action": "image_edit_clean_restore",
                        "status": "success",
                        "duration_seconds": duration,
                        "image_url": clean_url
                    })
                    print(f"[Fluxo Imagem] Restauração instantânea de imagem original limpa concluída em {duration}s!")
                    return md_clean, tools_called

            # Construção de prompt direcionado cirurgicamente para os modelos de edição
            if phrase_intent.get("remove_all_text"):
                tailored_prompt = (
                    "Remove all text, typography, letters, words, and titles from the image. "
                    "Keep all people, human subjects, faces, facial expressions, background, colors, framing, and visual composition "
                    "completely identical and pristine, leaving only the clean photo without any text."
                )
            elif phrase_intent.get("action") == "replace_phrase" and phrase_intent.get("old_phrase") and phrase_intent.get("headline"):
                tailored_prompt = (
                    f"Edit the text on this image. Replace the old phrase '{phrase_intent['old_phrase']}' with the exact new text: '{phrase_intent['headline']}'. "
                    f"Do NOT write any other words or instructions. Keep all subjects, people, faces, identities, background, colors, and layout completely identical."
                )
            elif phrase_intent.get("action") == "remove_word" and phrase_intent.get("remove_word"):
                tailored_prompt = (
                    f"Remove only the word '{phrase_intent['remove_word']}' from the text/typography on the image. "
                    f"Keep all other words intact, and preserve all subjects, faces, colors, framing, and background completely untouched."
                )
            elif phrase_intent.get("action") == "set_phrase" and phrase_intent.get("headline"):
                tailored_prompt = (
                    f"Change the text on this image to read exactly: '{phrase_intent['headline']}'. "
                    f"Do NOT add any extra text or instruction words. Keep all subjects, faces, identities, styling, background, and visual composition completely identical."
                )
            else:
                tailored_prompt = f"Altere a imagem conforme a instrução mantendo todo o restante da cena e sujeitos originais idênticos: {user_prompt}"

            # Determinação da ordem de execução e prioridade
            is_text_edit = (
                phrase_intent.get("remove_all_text") or
                phrase_intent.get("action") in ("replace_phrase", "remove_word", "set_phrase") or
                any(k in user_prompt.lower() for k in ["frase", "texto", "palavra", "escrito", "escreva", "letra", "tirar", "remover", "substituir", "trocar"])
            )

            if req_model_info:
                req_code, req_label = req_model_info
                priority_reason = f"Modelo expressamente solicitado pelo usuário no prompt ('{req_label}')."
                if req_code == "openai":
                    execution_order = ["openai", "flux", "gemini"]
                elif req_code == "flux":
                    execution_order = ["flux", "openai", "gemini"]
                elif req_code == "gemini":
                    execution_order = ["gemini", "openai", "flux"]
                else:
                    execution_order = ["openai", "flux", "gemini"]
            elif history_ctx.get("original_creator") == "openai":
                execution_order = ["openai", "flux", "gemini"]
                priority_reason = "Modelo original (OpenAI GPT Image 2.5 Sunburst) mantido como prioritário para continuidade da criação anterior."
            elif is_text_edit:
                # Vantagem técnica evidente da OpenAI para renderização e edição tipográfica em português
                execution_order = ["openai", "flux", "gemini"]
                priority_reason = "OpenAI (GPT Image 2.5 Sunburst) priorizado por vantagem técnica superior em precisão tipográfica e manipulação textual."
            elif history_ctx.get("original_creator") == "flux":
                execution_order = ["flux", "openai", "gemini"]
                priority_reason = "Modelo original (Together FLUX.1-kontext-pro) mantido como prioritário para inpainting contextual."
            elif history_ctx.get("original_creator") == "gemini":
                execution_order = ["gemini", "openai", "flux"]
                priority_reason = "Modelo original (Google Gemini Flash Lite Image) mantido como prioritário."
            else:
                execution_order = ["openai", "flux", "gemini"] if is_text_edit else ["flux", "openai", "gemini"]
                priority_reason = "Prioridade definida por adequação técnica da tarefa."

            print(f"[Fluxo Imagem] Ordem de execução para edição: {execution_order}. Motivo: {priority_reason}")

            openai_key = os.getenv("OPENAI_API_KEY", "")
            together_key = os.getenv("TOGETHERAI_API_KEY", "")

            # -------------------------------------------------------------
            # Provedor 1: OpenAI (GPT Image 2.5 Sunburst)
            # -------------------------------------------------------------
            async def _try_edit_openai() -> Optional[Tuple[str, Dict[str, Any]]]:
                if not openai_key or not png_bytes:
                    return None
                try:
                    print(f"[Fluxo Imagem] Enviando edição para OpenAI /v1/images/edits (gpt-image-2.5-sunburst)...")
                    # Detecta proporção aproximada da imagem original
                    size_str = "1024x1024"
                    try:
                        with Image.open(io.BytesIO(png_bytes)) as p_img:
                            w, h = p_img.size
                            ratio = h / w if w > 0 else 1.0
                            if ratio >= 1.25:
                                size_str = "1024x1536"
                            elif ratio <= 0.8:
                                size_str = "1536x1024"
                    except Exception:
                        size_str = "1024x1024"

                    data_payload = {
                        "model": "gpt-image-2.5-sunburst",
                        "prompt": tailored_prompt,
                        "size": size_str
                    }
                    files_payload = [('image[]', ('edit_input.png', png_bytes, 'image/png'))]

                    async with httpx.AsyncClient(timeout=120.0) as client:
                        oai_res = await client.post(
                            "https://api.openai.com/v1/images/edits",
                            headers={"Authorization": f"Bearer {openai_key}"},
                            data=data_payload,
                            files=files_payload
                        )
                        if oai_res.status_code == 200:
                            oai_data = oai_res.json()
                            img_list = oai_data.get("data", [])
                            if img_list:
                                saved_url = await save_image_locally(img_list[0], prefix="edit_oai")
                                if saved_url:
                                    duration = round(time.time() - flow_start, 2)
                                    md = (
                                        f"![Imagem Editada]({saved_url})\n\n"
                                        f"✨ **Edição de imagem concluída com sucesso!**\n\n"
                                        f"- **Provedor:** `OpenAI`\n"
                                        f"- **Modelo:** `gpt-image-2.5-sunburst`\n"
                                        f"- **Critério de Seleção:** {priority_reason}\n"
                                        f"- **Tempo de Processamento:** `{duration}s`\n"
                                        f"- **Instrução Aplicada:** _{user_prompt}_\n"
                                        f"- **Modo:** Edição contextual de alta precisão com preservação de identidades e tipografia."
                                    )
                                    tool_info = {
                                        "action": "image_edit",
                                        "model": "gpt-image-2.5-sunburst",
                                        "provider": "OpenAI (GPT Image 2.5 Sunburst)",
                                        "status": "success",
                                        "duration_seconds": duration,
                                        "image_url": saved_url,
                                        "priority_reason": priority_reason
                                    }
                                    return md, tool_info
                        else:
                            print(f"[Erro OpenAI Edit HTTP {oai_res.status_code}]: {oai_res.text[:250]}")
                except Exception as oai_err:
                    print(f"[Aviso OpenAI Edit Falhou]: {oai_err}. Prosseguindo para próximo modelo...")
                return None

            # -------------------------------------------------------------
            # Provedor 2: Together AI (FLUX.1-kontext-pro)
            # -------------------------------------------------------------
            async def _try_edit_flux() -> Optional[Tuple[str, Dict[str, Any]]]:
                if not together_key or not raw_b64_url.startswith("data:image"):
                    return None
                try:
                    print(f"[Fluxo Imagem] Enviando edição para Together AI (FLUX.1-kontext-pro)...")
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        tog_res = await client.post(
                            "https://api.together.xyz/v1/images/generations",
                            json={
                                "model": "black-forest-labs/FLUX.1-kontext-pro",
                                "prompt": tailored_prompt,
                                "image_url": raw_b64_url
                            },
                            headers={
                                "Authorization": f"Bearer {together_key}",
                                "Content-Type": "application/json"
                            }
                        )
                        if tog_res.status_code == 200:
                            tog_data = tog_res.json()
                            img_list = tog_data.get("data", [])
                            if img_list and "url" in img_list[0]:
                                saved_url = await save_image_locally({"url": img_list[0]["url"]}, prefix="edit")
                                if saved_url:
                                    duration = round(time.time() - flow_start, 2)
                                    md = (
                                        f"![Imagem Editada]({saved_url})\n\n"
                                        f"✨ **Edição de imagem concluída com sucesso!**\n\n"
                                        f"- **Provedor:** `Together AI (Black Forest Labs)`\n"
                                        f"- **Modelo:** `FLUX.1-kontext-pro`\n"
                                        f"- **Critério de Seleção:** {priority_reason}\n"
                                        f"- **Tempo de Processamento:** `{duration}s`\n"
                                        f"- **Instrução Aplicada:** _{user_prompt}_\n"
                                        f"- **Modo:** Edição contextual preservando imagem base original."
                                    )
                                    tool_info = {
                                        "action": "image_edit",
                                        "model": "black-forest-labs/FLUX.1-kontext-pro",
                                        "provider": "Together AI",
                                        "status": "success",
                                        "duration_seconds": duration,
                                        "image_url": saved_url,
                                        "priority_reason": priority_reason
                                    }
                                    return md, tool_info
                        else:
                            print(f"[Erro Together Kontext HTTP {tog_res.status_code}]: {tog_res.text[:250]}")
                except Exception as tog_err:
                    print(f"[Aviso Together Kontext Falhou]: {tog_err}. Prosseguindo para próximo modelo...")
                return None

            # -------------------------------------------------------------
            # Provedor 3: Google Gemini Flash Lite Image via LiteLLM
            # -------------------------------------------------------------
            async def _try_edit_gemini() -> Optional[Tuple[str, Dict[str, Any]]]:
                target_model = "ImageEdit"
                provider_name = "Google Gemini (Flash Lite Image)"
                edit_messages = []
                user_blocks = [
                    {"type": "text", "text": f"{tailored_prompt}. Mantenha a imagem original inalterada exceto pelas alterações solicitadas e gere a imagem final."},
                    resolved_block
                ]
                edit_messages.append({"role": "user", "content": user_blocks})

                try:
                    print(f"[Fluxo Imagem] Enviando edição com imagem base64 para '{target_model}'...")
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        res = await client.post(
                            f"{LITELLM_BASE_URL}/chat/completions",
                            json={"model": target_model, "messages": edit_messages},
                            headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
                        )
                        if res.status_code == 200:
                            res_json = res.json()
                            choice_msg = res_json.get("choices", [{}])[0].get("message", {})
                            img_data = None
                            if "images" in choice_msg and choice_msg["images"]:
                                raw_img = choice_msg["images"][0]
                                if isinstance(raw_img, dict):
                                    img_data = {"b64_json": raw_img.get("image_url", {}).get("url") or raw_img.get("b64_json")}
                                elif isinstance(raw_img, str):
                                    img_data = {"b64_json": raw_img}

                            if img_data:
                                saved_url = await save_image_locally(img_data, prefix="edit")
                                if saved_url:
                                    duration = round(time.time() - flow_start, 2)
                                    md = (
                                        f"![Imagem Editada]({saved_url})\n\n"
                                        f"✨ **Edição de imagem concluída com sucesso!**\n\n"
                                        f"- **Provedor:** `{provider_name}`\n"
                                        f"- **Modelo:** `{target_model}`\n"
                                        f"- **Critério de Seleção:** {priority_reason}\n"
                                        f"- **Tempo de Processamento:** `{duration}s`\n"
                                        f"- **Instrução Aplicada:** _{user_prompt}_"
                                    )
                                    tool_info = {
                                        "action": "image_edit",
                                        "model": target_model,
                                        "provider": provider_name,
                                        "status": "success",
                                        "duration_seconds": duration,
                                        "image_url": saved_url,
                                        "priority_reason": priority_reason
                                    }
                                    return md, tool_info
                            else:
                                print(f"[Aviso ImageEdit] Mensagem retornada sem bloco de imagem: {choice_msg.get('content')[:150]}")
                        else:
                            print(f"[Erro ImageEdit HTTP {res.status_code}]: {res.text[:250]}")
                except Exception as edit_err:
                    print(f"[Aviso ImageEdit Falhou]: {edit_err}. Prosseguindo para fallback...")
                return None

            provider_dispatch = {
                "openai": _try_edit_openai,
                "flux": _try_edit_flux,
                "gemini": _try_edit_gemini
            }

            for p_key in execution_order:
                fn = provider_dispatch.get(p_key)
                if fn:
                    res_tuple = await fn()
                    if res_tuple:
                        md_res, tool_res = res_tuple
                        tools_called.append(tool_res)
                        return md_res, tools_called

        # Fallback de edição: transformar em geração Pro com o contexto
        print(f"[Fluxo Imagem] Fallback: Gerando nova imagem baseada nas modificações solicitadas...")
        user_prompt = f"Gere uma imagem com as seguintes características e alterações: {user_prompt}"
        user_prompt = f"Gere uma imagem com as seguintes características e alterações: {user_prompt}"

    # -------------------------------------------------------------
    # 2. FLUXO: GERAÇÃO DE IMAGEM (IMAGE_GENERATION)
    # -------------------------------------------------------------
    pro_keywords = [
        "pro", "flux", "grok", "fotorrealista", "fotorealista", "fotorrealismo",
        "8k", "4k", "ultra-realista", "hiper-realista", "photorealistic",
        "estúdio", "studio", "cinema", "cinematográfico", "alta fidelidade",
        "detalhada", "obra de arte", "masterpiece", "high resolution", "profissional"
    ]
    is_pro = any(k in user_prompt.lower() for k in pro_keywords)
    tier_label = "Pro (Alta Fidelidade)" if is_pro else "Fast (Econômico / Rápido)"

    if is_pro:
        # Alterna entre Together FLUX.2-dev, OpenAI GPT Image 2.5 Flare e xAI Grok Imagine 2.0 para balancear tokens e custos
        _pro_provider_turn += 1
        turn = _pro_provider_turn % 3
        if turn == 1:
            candidate_chain = [
                ("ImageGen_Pro_Flux", "Together AI (FLUX.2-dev)"),
                ("ImageGen_Pro_OpenAI", "OpenAI (GPT Image 2.5 Flare)"),
                ("ImageGen_Pro_Sunburst", "OpenAI (GPT Image 2.5 Sunburst)"),
                ("ImageGen_Pro_Gemini", "Google (Imagen 3 Pro)"),
                ("ImageGen_Pro", "xAI (Grok Imagine 2.0)"),
                ("ImageGen_Fast_SDXL", "Together AI (SDXL Fallback)"),
                ("ImageGen_Fast", "Google Gemini (Flash Lite)")
            ]
        elif turn == 2:
            candidate_chain = [
                ("ImageGen_Pro_OpenAI", "OpenAI (GPT Image 2.5 Flare)"),
                ("ImageGen_Pro_Sunburst", "OpenAI (GPT Image 2.5 Sunburst)"),
                ("ImageGen_Pro_Gemini", "Google (Imagen 3 Pro)"),
                ("ImageGen_Pro", "xAI (Grok Imagine 2.0)"),
                ("ImageGen_Pro_Flux", "Together AI (FLUX.2-dev)"),
                ("ImageGen_Fast_SDXL", "Together AI (SDXL Fallback)"),
                ("ImageGen_Fast", "Google Gemini (Flash Lite)")
            ]
        else:
            candidate_chain = [
                ("ImageGen_Pro", "xAI (Grok Imagine 2.0)"),
                ("ImageGen_Pro_Gemini", "Google (Imagen 3 Pro)"),
                ("ImageGen_Pro_Flux", "Together AI (FLUX.2-dev)"),
                ("ImageGen_Pro_OpenAI", "OpenAI (GPT Image 2.5 Flare)"),
                ("ImageGen_Pro_Sunburst", "OpenAI (GPT Image 2.5 Sunburst)"),
                ("ImageGen_Fast_SDXL", "Together AI (SDXL Fallback)"),
                ("ImageGen_Fast", "Google Gemini (Flash Lite)")
            ]
    else:
        # Fast Tier: Tenta Gemini Flash Lite (grátis), fallback para SDXL ($0.002), fallback para Flare/FLUX/Grok
        candidate_chain = [
            ("ImageGen_Fast", "Google Gemini (Flash Lite)"),
            ("ImageGen_Fast_SDXL", "Together AI (SDXL 1.0)"),
            ("ImageGen_Pro_OpenAI", "OpenAI (GPT Image 2.5 Flare)"),
            ("ImageGen_Pro_Sunburst", "OpenAI (GPT Image 2.5 Sunburst)"),
            ("ImageGen_Pro_Gemini", "Google (Imagen 3 Pro)"),
            ("ImageGen_Pro_Flux", "Together AI (FLUX.2-dev)"),
            ("ImageGen_Pro", "xAI (Grok Imagine 2.0)")
        ]

    try:
        from admin_config_service import apply_model_priority
        cat_key = "IMAGE_GENERATION_PRO" if is_pro else "IMAGE_GENERATION_FAST"
        candidate_chain = apply_model_priority(cat_key, candidate_chain, name_index=0)
    except Exception as pri_err:
        pass

    for model_name, provider_label in candidate_chain:
        call_start = time.time()
        print(f"[Fluxo Imagem] Tentando gerar via '{model_name}' ({provider_label})...")
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                res = await client.post(
                    f"{LITELLM_BASE_URL}/images/generations",
                    json={"model": model_name, "prompt": user_prompt, "n": 1},
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
                )

                if res.status_code == 200:
                    res_json = res.json()
                    data_items = res_json.get("data", [])
                    if data_items:
                        public_url = await save_image_locally(data_items[0], prefix="gen")
                        if public_url:
                            duration = round(time.time() - flow_start, 2)
                            md_output = (
                                f"![Imagem Gerada]({public_url})\n\n"
                                f"🎨 **Imagem gerada com sucesso!**\n\n"
                                f"- **Provedor:** `{provider_label}`\n"
                                f"- **Modelo:** `{model_name}`\n"
                                f"- **Tier:** `{tier_label}`\n"
                                f"- **Tempo de Criação:** `{duration}s`\n"
                                f"- **Prompt:** _{user_prompt}_"
                            )
                            tools_called.append({
                                "action": "image_generation",
                                "model": model_name,
                                "provider": provider_label,
                                "tier": tier_label,
                                "status": "success",
                                "duration_seconds": duration,
                                "image_url": public_url
                            })
                            print(f"[Fluxo Imagem] Sucesso com {model_name} em {duration}s!")
                            return md_output, tools_called
                else:
                    err_body = res.text[:200]
                    print(f"[Fluxo Imagem] Falha em {model_name} (HTTP {res.status_code}): {err_body}. Tentando próximo provedor...")
                    tools_called.append({
                        "action": "image_generation",
                        "model": model_name,
                        "status": f"failed_http_{res.status_code}",
                        "duration_seconds": round(time.time() - call_start, 2)
                    })
        except Exception as gen_err:
            print(f"[Fluxo Imagem] Exceção em {model_name}: {gen_err}. Tentando próximo provedor...")
            tools_called.append({
                "action": "image_generation",
                "model": model_name,
                "status": f"exception: {str(gen_err)}",
                "duration_seconds": round(time.time() - call_start, 2)
            })

    # Caso todos os provedores falhem
    fail_msg = "⚠️ Desculpe, não foi possível gerar a imagem no momento. Todos os provedores retornaram erro temporário. Por favor, tente novamente em instantes."
    return fail_msg, tools_called
