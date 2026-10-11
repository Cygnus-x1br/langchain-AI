import os
import re
import time
import uuid
import base64
import httpx
from typing import Tuple, List, Dict, Any

from config import (
    IMAGE_STORAGE_DIR,
    SERVER_PUBLIC_URL,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    get_now_local,
    resolve_image_to_base64,
    sync_image_to_librechat
)

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

async def execute_image_flow(category: str, user_prompt: str, last_user_msg: dict, messages: list) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executa a geração ou edição de imagem multi-provedor com balanceamento de carga e fallbacks.
    Tiers suportados:
      - ImageGen_Fast (Gemini Flash Lite - Grátis) com failover para ImageGen_Fast_SDXL (Together - $0.002)
      - ImageGen_Pro (Grok Imagine 2.0 via xAI), ImageGen_Pro_Flux (FLUX.2-dev via Together - $0.015) e ImageGen_Pro_OpenAI (GPT Image 2.5 Flare via OpenAI - $0.013)
      - ImageEdit (FLUX.1 Kontext Pro via Together / Gemini Flash Lite Image) para edição multimodal/inpainting
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

            # 1. Tentativa Principal: FLUX.1 Kontext Pro (Together AI - Especialista em Edição Preservativa de Imagem)
            together_key = os.getenv("TOGETHERAI_API_KEY", "")
            if together_key and raw_b64_url.startswith("data:image"):
                print(f"[Fluxo Imagem] Enviando pedido de edição para FLUX.1 Kontext Pro (Together AI)...")
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        tog_res = await client.post(
                            "https://api.together.xyz/v1/images/generations",
                            json={
                                "model": "black-forest-labs/FLUX.1-kontext-pro",
                                "prompt": f"Altere a imagem conforme a instrução mantendo todo o restante da cena e sujeitos originais idênticos: {user_prompt}",
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
                                public_url = await save_image_locally({"url": img_list[0]["url"]}, prefix="edit")
                                if public_url:
                                    duration = round(time.time() - flow_start, 2)
                                    md_output = (
                                        f"![Imagem Editada]({public_url})\n\n"
                                        f"✨ **Edição de imagem concluída com sucesso!**\n\n"
                                        f"- **Provedor:** `Together AI (Black Forest Labs)`\n"
                                        f"- **Modelo:** `FLUX.1-kontext-pro`\n"
                                        f"- **Tempo de Processamento:** `{duration}s`\n"
                                        f"- **Instrução:** _{user_prompt}_\n"
                                        f"- **Modo:** Edição contextual preservando imagem base original."
                                    )
                                    tools_called.append({
                                        "action": "image_edit",
                                        "model": "black-forest-labs/FLUX.1-kontext-pro",
                                        "provider": "Together AI",
                                        "status": "success",
                                        "duration_seconds": duration,
                                        "image_url": public_url
                                    })
                                    print(f"[Fluxo Imagem] Edição com FLUX.1 Kontext Pro bem-sucedida em {duration}s!")
                                    return md_output, tools_called
                        else:
                            print(f"[Erro Together Kontext HTTP {tog_res.status_code}]: {tog_res.text[:250]}")
                except Exception as tog_err:
                    print(f"[Aviso Together Kontext Falhou]: {tog_err}. Prosseguindo para próximo modelo...")

            # 2. Tentativa Secundária: Google Gemini Flash Lite Image via LiteLLM
            edit_messages = []
            user_blocks = [
                {"type": "text", "text": f"{user_prompt}. Mantenha a imagem original inalterada exceto pelas alterações solicitadas e gere a imagem final."},
                resolved_block
            ]
            edit_messages.append({"role": "user", "content": user_blocks})

            target_model = "ImageEdit"
            provider_name = "Google Gemini (Flash Lite Image)"
            call_start = time.time()
            try:
                print(f"[Fluxo Imagem] Enviando pedido de edição com imagem base64 para '{target_model}'...")
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
                            public_url = await save_image_locally(img_data, prefix="edit")
                            if public_url:
                                duration = round(time.time() - flow_start, 2)
                                md_output = (
                                    f"![Imagem Editada]({public_url})\n\n"
                                    f"✨ **Edição de imagem concluída com sucesso!**\n\n"
                                    f"- **Provedor:** `{provider_name}`\n"
                                    f"- **Modelo:** `{target_model}`\n"
                                    f"- **Tempo de Processamento:** `{duration}s`\n"
                                    f"- **Instrução:** _{user_prompt}_"
                                )
                                tools_called.append({
                                    "action": "image_edit",
                                    "model": target_model,
                                    "provider": provider_name,
                                    "status": "success",
                                    "duration_seconds": duration,
                                    "image_url": public_url
                                })
                                print(f"[Fluxo Imagem] Edição bem-sucedida em {duration}s!")
                                return md_output, tools_called
                        else:
                            print(f"[Aviso ImageEdit] Mensagem retornada sem bloco de imagem: {choice_msg.get('content')[:150]}")
                    else:
                        print(f"[Erro ImageEdit HTTP {res.status_code}]: {res.text[:250]}")
            except Exception as edit_err:
                print(f"[Aviso ImageEdit Falhou]: {edit_err}. Prosseguindo para fallback...")

        # Fallback de edição: transformar em geração Pro com o contexto
        print(f"[Fluxo Imagem] Fallback: Gerando nova imagem baseada nas modificações solicitadas...")
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
