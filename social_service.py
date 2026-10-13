import os
import re
import json
import time
import asyncio
import httpx
from typing import Tuple, List, Dict, Any, Optional

import agent_service
from config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    get_now_local,
    resolve_image_to_base64
)
from prompts import (
    SOCIAL_PLANNER_PROMPT,
    SOCIAL_RESEARCH_PROMPT,
    SOCIAL_CREATOR_PROMPT
)
from image_service import save_image_locally
from config import IMAGE_STORAGE_DIR, SERVER_PUBLIC_URL, sync_image_to_librechat

# ---------------------------------------------------------
# Cadeias de modelos por agente
# ---------------------------------------------------------
PLANNER_MODELS = ["Classifier_Fast", "Combo_Free_Chat", "ComboFree_Smart"]
RESEARCH_MODELS_DEFAULT = ["Coding_DeepSeek_V3", "ComboFree_Smart", "Combo_Free_Chat"]
RESEARCH_MODELS_COMPLEX = ["Reasoning_DeepSeek_R1", "Coding_DeepSeek_V3", "ComboFree_Smart"]
CREATOR_MODELS = ["ComboFree_Smart", "Coding_DeepSeek_V3", "Combo_Free_Chat"]
VISION_MODELS = ["ComboVision"]

# Mapeamento técnico de enquadramento e resolução padrão por rede social
PLATFORM_SPECS = {
    "INSTAGRAM": {
        "label": "Instagram",
        "aspect_ratio": "4:5",
        "image_size": "1024x1792",
        "dimensions": "1080x1350 (Retrato Feed)",
        "tone": "Visual, envolvente, storytelling dinâmico e espaçamento arejado",
        "hashtags_count": "5 a 8 hashtags nichadas"
    },
    "LINKEDIN": {
        "label": "LinkedIn",
        "aspect_ratio": "1.91:1",
        "image_size": "1792x1024",
        "dimensions": "1200x627 (Banner de Artigo/Feed)",
        "tone": "Profissional, liderança de pensamento, orientado a dados e carreira",
        "hashtags_count": "3 a 5 hashtags corporativas"
    },
    "X_TWITTER": {
        "label": "X (Twitter)",
        "aspect_ratio": "16:9",
        "image_size": "1792x1024",
        "dimensions": "1200x675 (Landscape)",
        "tone": "Concisão máxima, gancho afiado e impacto imediato",
        "hashtags_count": "1 a 2 hashtags estratégicas"
    },
    "FACEBOOK": {
        "label": "Facebook",
        "aspect_ratio": "1:1",
        "image_size": "1024x1024",
        "dimensions": "1080x1080 (Feed Quadrado)",
        "tone": "Comunitário, acolhedor, estimulando debate nos comentários",
        "hashtags_count": "2 a 4 hashtags"
    },
    "REDDIT": {
        "label": "Reddit",
        "aspect_ratio": "16:9",
        "image_size": "1792x1024",
        "dimensions": "1200x675 (Autêntico)",
        "tone": "Anti-marketing, transparente, debate de alto valor e informal",
        "hashtags_count": "Sem hashtags (cultura do Reddit rejeita hashtags)"
    },
    "MULTIPLATAFORMA": {
        "label": "Multiplataforma (Adaptável)",
        "aspect_ratio": "1:1",
        "image_size": "1024x1024",
        "dimensions": "1080x1080 (Universal)",
        "tone": "Balanceado para engajamento e captação multicanal",
        "hashtags_count": "3 a 5 hashtags versáteis"
    }
}


def detect_social_platform(user_prompt: str) -> str:
    """Identifica a rede social desejada com base no prompt do usuário."""
    p = user_prompt.lower()
    if re.search(r"\b(insta|instagram|stories|reels)\b", p):
        return "INSTAGRAM"
    elif re.search(r"\b(linkedin|linke)\b", p):
        return "LINKEDIN"
    elif re.search(r"\b(twitter|x\b|tweet|fio|thread)\b", p):
        return "X_TWITTER"
    elif re.search(r"\b(facebook|face|fb)\b", p):
        return "FACEBOOK"
    elif re.search(r"\b(reddit|subreddit)\b", p):
        return "REDDIT"
    return "MULTIPLATAFORMA"


# ---------------------------------------------------------
# Utilitários de LLM e parsing
# ---------------------------------------------------------
def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Extrai um objeto JSON de uma resposta de LLM (remove <think>, cercas markdown e texto extra)."""
    if not raw:
        return None
    clean = re.sub(r"(?s)<think>.*?</think>", "", raw).strip()
    if "```json" in clean:
        clean = clean.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in clean:
        parts = clean.split("```")
        if len(parts) >= 2:
            clean = parts[1].strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    start, end = clean.find("{"), clean.rfind("}")
    if start != -1 and end > start:
        snippet = clean[start:end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            # Tenta reparar JSON com hashtags ou aspas não fechadas
            repaired = re.sub(r',\s*([\]}])', r'\1', snippet)
            repaired = re.sub(r'\[\s*#([^",\]\s]+)', r'["#\1', repaired)
            repaired = re.sub(r',\s*#([^",\]\s]+)', r', "#\1"', repaired)
            try:
                return json.loads(repaired)
            except Exception:
                pass
            # Fallback cirúrgico por regex para campos essenciais do post
            fields = {}
            for k in ["hook", "body_copy", "call_to_action", "objective", "image_prompt_en", "visual_style", "strategic_tip"]:
                m = re.search(rf'"{k}"\s*:\s*"((?:\\.|[^"\\])*)"', snippet)
                if m:
                    fields[k] = m.group(1).encode().decode('unicode_escape', 'ignore')
            if "hashtags" not in fields:
                ht = re.findall(r'#\w+', snippet)
                if ht:
                    fields["hashtags"] = ht
            if fields.get("body_copy") or fields.get("hook"):
                return fields
    return None


from phrase_extractor_guardrail import extract_user_phrases_guardrail


async def extract_user_requested_phrases(user_prompt: str, messages: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Encaminha para o guardrail determinístico de frases (phrase_extractor_guardrail).
    Suporta aspas, tópicos, triggers 'inclua a frase...', histórico e fallback LLM.
    """
    return await extract_user_phrases_guardrail(user_prompt, messages)


async def _call_llm_chain(
    models: List[str],
    messages: List[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    timeout: float = 60.0
) -> Tuple[Optional[str], Optional[str]]:
    """Chama a cadeia de modelos via LiteLLM com fallback. Retorna (texto, modelo_usado)."""
    for m in models:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(
                    f"{LITELLM_BASE_URL}/chat/completions",
                    json={"model": m, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
                )
                if res.status_code == 200:
                    content = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        return content, m
                else:
                    print(f"[Social LLM] '{m}' retornou HTTP {res.status_code}: {res.text[:120]}")
        except Exception as e:
            print(f"[Social LLM] Erro em '{m}': {e}. Tentando próximo...")
    return None, None


# ---------------------------------------------------------
# Tratamento de imagens (anexos base64 -> URL pública local)
# ---------------------------------------------------------
def _extract_image_url_from_part(part: Any) -> Optional[str]:
    if not isinstance(part, dict):
        return None
    img = part.get("image_url")
    if isinstance(img, dict):
        return img.get("url")
    if isinstance(img, str):
        return img
    return part.get("url")


def extract_all_images(messages: list, last_user_msg: dict) -> List[str]:
    """
    Extrai todas as imagens (URLs públicas ou data URIs) presentes na mensagem atual do usuário
    e no histórico recente da conversa.
    """
    found_urls = []
    
    # 1. Mensagem atual
    content = last_user_msg.get("content")
    if isinstance(content, list):
        for part in content:
            url = _extract_image_url_from_part(part)
            if url and url not in found_urls:
                found_urls.append(url)

    # 2. Histórico recente de mensagens (se houver)
    if messages:
        for msg in reversed(messages[:-1]):
            m_content = msg.get("content")
            if isinstance(m_content, list):
                for part in m_content:
                    url = _extract_image_url_from_part(part)
                    if url and url not in found_urls:
                        found_urls.append(url)
            elif isinstance(m_content, str):
                md_matches = re.findall(r"!\[.*?\]\((https?://[^\s\)]+|/images/[^\s\)]+)\)", m_content)
                for u in md_matches:
                    if u not in found_urls:
                        found_urls.append(u)
                raw_matches = re.findall(r"(https?://[^\s]+\.(png|jpg|jpeg|webp))", m_content, re.IGNORECASE)
                for u, _ in raw_matches:
                    if u not in found_urls:
                        found_urls.append(u)
                        
    return found_urls


def find_existing_image_in_history(messages: list, last_user_msg: dict) -> Optional[str]:
    """
    Busca a imagem anexada pelo usuário (mensagem atual) ou gerada/enviada anteriormente no histórico.
    Suporta blocos image_url multimodais (URL ou data URI) e links Markdown.
    """
    all_imgs = extract_all_images(messages, last_user_msg)
    return all_imgs[0] if all_imgs else None


async def ensure_public_image_url(image_url: str) -> str:
    """
    Anexos do LibreChat chegam como 'data:image/...;base64,...', que o chat não renderiza em Markdown
    (e que inflariam logs). Converte para arquivo local servido em /images/.
    """
    if image_url and image_url.startswith("data:image/"):
        saved = await save_image_locally({"b64_json": image_url}, prefix="social_src")
        if saved:
            print(f"[Social Flow] Imagem anexa (base64) salva como URL pública: {saved}")
            return saved
    return image_url


async def describe_source_image(image_url: str) -> Optional[str]:
    """Agente de visão: descreve objetivamente a imagem que será usada na postagem."""
    try:
        block = await resolve_image_to_base64({"type": "image_url", "image_url": {"url": image_url}})
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Descreva objetivamente esta imagem em português para um diretor de arte de social media: "
                    "cena, pessoas/objetos, esporte ou atividade, cores dominantes, composição, "
                    "texto visível e onde há espaço livre para sobrepor texto. Máximo 120 palavras."
                )},
                block
            ]
        }]
        try:
            from admin_config_service import apply_model_priority
            vision_models = apply_model_priority("SOCIAL_VISION", VISION_MODELS)
        except Exception:
            vision_models = VISION_MODELS
        text, used = await _call_llm_chain(vision_models, messages, temperature=0.2, max_tokens=500, timeout=60.0)
        if text:
            print(f"[Social Research] Imagem descrita por '{used}'.")
        return text.strip() if text else None
    except Exception as e:
        print(f"[Social Research] Falha ao descrever imagem: {e}")
        return None


# ---------------------------------------------------------
# AGENTE 1: Pesquisa e Direção Criativa
# ---------------------------------------------------------
def _heuristic_needs_search(user_prompt: str) -> bool:
    """Fallback do planejador: nomes próprios no meio da frase ou termos de evento/ano indicam pesquisa."""
    p = user_prompt.lower()
    if re.search(r"\b(torneio|campeonato|copa|trof[eé]u|evento|congresso|festival|lan[cç]amento|"
                 r"competi[cç][aã]o|olimp[ií]ada|etapa|edi[cç][aã]o|202\d)\b", p):
        return True
    words = re.findall(r"\b[A-ZÀ-Ú][a-zà-ú]{2,}\b", user_prompt)
    return len(words) >= 3


def is_complex_social_request(user_prompt: str, needs_search: bool) -> bool:
    """Decide se o pesquisador deve usar o modelo de raciocínio profundo (R1) em vez do V3."""
    p = user_prompt.lower()
    multi_platform = len(re.findall(r"\b(instagram|insta|linkedin|facebook|twitter|reddit|tiktok)\b", p)) > 1
    campaign = bool(re.search(r"\b(campanha|funil|calend[aá]rio|s[eé]rie|estrat[eé]gia|lead|leads|carrossel)\b", p))
    return multi_platform or campaign or len(user_prompt) > 600


async def plan_research(user_prompt: str, image_description: Optional[str] = None) -> Dict[str, Any]:
    """Planejador leve: decide se é necessária pesquisa web, domínio do assunto e consultas."""
    user_msg = f"Ano corrente: {get_now_local().year}\nPedido do usuário:\n{user_prompt}"
    if image_description:
        user_msg += f"\n\nDescrição da imagem anexada:\n{image_description}"
        
    try:
        from admin_config_service import get_prompt_text
        planner_prompt = get_prompt_text("SOCIAL_PLANNER_PROMPT") or SOCIAL_PLANNER_PROMPT
    except Exception:
        planner_prompt = SOCIAL_PLANNER_PROMPT

    try:
        from admin_config_service import apply_model_priority
        planner_models = apply_model_priority("SOCIAL_PLANNER", PLANNER_MODELS)
    except Exception:
        planner_models = PLANNER_MODELS

    text, _ = await _call_llm_chain(
        planner_models,
        [
            {"role": "system", "content": planner_prompt},
            {"role": "user", "content": user_msg}
        ],
        temperature=0.0, max_tokens=400, timeout=20.0
    )
    plan = _extract_json(text) if text else None
    if not isinstance(plan, dict):
        needs = _heuristic_needs_search(user_prompt)
        print(f"[Social Research] Planejador indisponível. Heurística needs_search={needs}")
        return {"needs_search": needs, "domain": "", "entities": [], "queries": [user_prompt[:150]] if needs else []}
    plan["needs_search"] = bool(plan.get("needs_search"))
    if plan["needs_search"] and not plan.get("queries"):
        plan["queries"] = [user_prompt[:150]]
    plan["queries"] = [q for q in (plan.get("queries") or []) if isinstance(q, str)][:2]
    return plan


async def run_web_research(queries: List[str]) -> Tuple[str, List[str]]:
    """Executa consultas SearXNG/Tavily em paralelo. Retorna (texto_consolidado, urls_fontes)."""
    if not queries:
        return "", []

    from config import SEARXNG_URL

    async def _one(q: str):
        try:
            if SEARXNG_URL:
                from agent_service import async_searx_search
                data = await async_searx_search(q, max_results=3)
            elif agent_service.tavily_tool:
                data = await agent_service.tavily_tool.ainvoke(q)
            else:
                data = None
            return q, data
        except Exception as e:
            print(f"[Social Research] Erro Search API para '{q}': {e}")
            return q, None

    results = await asyncio.gather(*[_one(q) for q in queries])
    blocks, sources = [], []
    for q, data in results:
        if not data:
            continue
        blocks.append(f"### Consulta: {q}")
        items = data if isinstance(data, list) else [data]
        for it in items[:4]:
            if isinstance(it, dict):
                url = it.get("url", "")
                content = str(it.get("content", ""))[:700]
                blocks.append(f"- Fonte: {url}\n  {content}")
                if url and url not in sources:
                    sources.append(url)
            else:
                blocks.append(f"- {str(it)[:700]}")
    return "\n".join(blocks), sources


async def run_research_agent(
    user_prompt: str,
    platform_key: str,
    image_description: Optional[str],
    image_available: bool,
    tools_called: List[Dict[str, Any]],
    exact_headline: Optional[str] = None,
    exact_subhead: Optional[str] = None
) -> Dict[str, Any]:
    """
    Agente 1: planeja, pesquisa na web (quando necessário), analisa a imagem anexa e produz o BRIEFING
    detalhado (fatos, vocabulário, elementos visuais, emojis, fontes, paleta, layout).
    """
    spec = PLATFORM_SPECS.get(platform_key, PLATFORM_SPECS["MULTIPLATAFORMA"])

    # 1.a Planejamento (decide se pesquisa)
    t0 = time.time()
    plan = await plan_research(user_prompt, image_description)
    tools_called.append({
        "action": "social_research_plan",
        "needs_search": plan.get("needs_search"),
        "domain": plan.get("domain"),
        "queries": plan.get("queries"),
        "duration_seconds": round(time.time() - t0, 2)
    })

    # 1.b Pesquisa web
    web_text, sources = "", []
    if plan.get("needs_search"):
        t0 = time.time()
        web_text, sources = await run_web_research(plan.get("queries", []))
        tools_called.append({
            "action": "social_web_research",
            "tool": "tavily_search_results",
            "status": "success" if web_text else "no_results",
            "sources": sources,
            "duration_seconds": round(time.time() - t0, 2)
        })

    # 1.c Briefing (V3 por padrão; R1 quando o pedido é complexo)
    complex_req = is_complex_social_request(user_prompt, bool(plan.get("needs_search")))
    models = RESEARCH_MODELS_COMPLEX if complex_req else RESEARCH_MODELS_DEFAULT
    cat_key = "SOCIAL_RESEARCH_COMPLEX" if complex_req else "SOCIAL_RESEARCH"
    try:
        from admin_config_service import apply_model_priority
        models = apply_model_priority(cat_key, models)
    except Exception:
        pass

    user_content = (
        f"DATA ATUAL: {get_now_local().strftime('%d/%m/%Y')}\n"
        f"PLATAFORMA: {spec['label']} | Formato: {spec['dimensions']} (aspect ratio {spec['aspect_ratio']})\n"
        f"TOM DA PLATAFORMA: {spec['tone']}\n"
        f"HASHTAGS: {spec['hashtags_count']}\n"
        f"DOMÍNIO IDENTIFICADO: {plan.get('domain') or 'não identificado'}\n"
        f"IMAGEM DO USUÁRIO DISPONÍVEL PARA REUSO: {'sim' if image_available else 'não'}\n\n"
        f"PEDIDO DO USUÁRIO:\n{user_prompt}\n\n"
        f"DESCRIÇÃO DA IMAGEM ANEXA:\n{image_description or 'Nenhuma imagem anexa.'}\n\n"
        f"DADOS DA PESQUISA WEB:\n{web_text or 'Nenhuma pesquisa realizada ou sem resultados.'}"
    )

    if exact_headline:
        user_content += f"\n\nTEXTOS/FRASES OBRIGATÓRIAS SOLICITADAS PELO USUÁRIO (NÃO ALTERE, NÃO INVENTE OUTRAS):\n- Título Obrigatório: {exact_headline}"
        if exact_subhead:
            user_content += f"\n- Subtítulo/Texto 2 Obrigatório: {exact_subhead}"

    t0 = time.time()
    try:
        from admin_config_service import get_prompt_text
        research_prompt = get_prompt_text("SOCIAL_RESEARCH_PROMPT") or SOCIAL_RESEARCH_PROMPT
    except Exception:
        research_prompt = SOCIAL_RESEARCH_PROMPT

    text, used_model = await _call_llm_chain(
        models,
        [
            {"role": "system", "content": research_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        max_tokens=3000,
        timeout=150.0 if complex_req else 70.0
    )
    briefing = _extract_json(text) if text else None

    if not isinstance(briefing, dict):
        print("[Social Research] Briefing indisponível. Usando briefing mínimo de contingência.")
        briefing = {
            "topic_summary": user_prompt[:200],
            "domain": plan.get("domain", ""),
            "verified_facts": [],
            "unverified_or_missing": ["Briefing automático indisponível: nenhum fato foi verificado."],
            "vocabulary_do": [],
            "vocabulary_dont": [],
            "tone": spec["tone"],
            "objective": "Engajamento",
            "key_messages": [],
            "must_include_text": "",
            "image_strategy": "reuse_attached" if image_available else "generate_new",
            "visual_elements": [], "complementary_images": [], "figures_icons": [], "emojis": [],
            "fonts": {}, "palette": [], "layout": "",
            "image_prompt_en": f"Editorial photography about {user_prompt[:80]}, clean composition, natural light",
            "hashtags_seed": []
        }
        used_model = used_model or "fallback"

    if exact_headline:
        briefing["must_include_text"] = exact_headline
        briefing["subhead_text"] = exact_subhead or ""
        if "verified_facts" not in briefing or not isinstance(briefing["verified_facts"], list):
            briefing["verified_facts"] = []
        if exact_headline not in briefing["verified_facts"]:
            briefing["verified_facts"].insert(0, f"Frase/Título obrigatório: {exact_headline}")
        if exact_subhead and exact_subhead not in briefing["verified_facts"]:
            briefing["verified_facts"].insert(1, f"Subtítulo obrigatório: {exact_subhead}")

    briefing["sources"] = sources
    briefing["web_research_used"] = bool(web_text)
    tools_called.append({
        "action": "social_research_briefing",
        "model": used_model,
        "reasoning_tier": "R1 (complexo)" if complex_req else "V3 (padrão)",
        "status": "success",
        "duration_seconds": round(time.time() - t0, 2)
    })
    return briefing


# ---------------------------------------------------------
# AGENTE 2: Criador do Post
# ---------------------------------------------------------
async def run_creator_agent(
    user_prompt: str,
    platform_key: str,
    briefing: Dict[str, Any],
    tools_called: List[Dict[str, Any]],
    exact_headline: Optional[str] = None,
    exact_subhead: Optional[str] = None
) -> Dict[str, Any]:
    """Agente 2: escreve a copy e o prompt visual executando fielmente o briefing."""
    spec = PLATFORM_SPECS.get(platform_key, PLATFORM_SPECS["MULTIPLATAFORMA"])
    briefing_for_creator = {k: v for k, v in briefing.items() if k not in ("sources", "web_research_used")}

    user_content = (
        f"PLATAFORMA: {spec['label']} | {spec['dimensions']} | Hashtags: {spec['hashtags_count']}\n\n"
        f"PEDIDO ORIGINAL DO USUÁRIO:\n{user_prompt}\n\n"
        f"BRIEFING DO AGENTE DE PESQUISA (siga à risca):\n{json.dumps(briefing_for_creator, ensure_ascii=False, indent=2)}\n\n"
    )

    if exact_headline:
        user_content += (
            f"ATENÇÃO - FRASES OBRIGATÓRIAS ESPECIFICADAS PELO USUÁRIO:\n"
            f"- TÍTULO/HOOK: '{exact_headline}'\n"
        )
        if exact_subhead:
            user_content += f"- SUBTÍTULO: '{exact_subhead}'\n"
        user_content += "Você DEVE usar essa frase no campo 'hook' e não substituí-la por títulos genéricos!\n\n"

    user_content += "Gere o JSON do post:"

    t0 = time.time()
    try:
        from admin_config_service import get_prompt_text
        creator_prompt = get_prompt_text("SOCIAL_CREATOR_PROMPT") or SOCIAL_CREATOR_PROMPT
    except Exception:
        creator_prompt = SOCIAL_CREATOR_PROMPT

    try:
        from admin_config_service import apply_model_priority
        creator_models = apply_model_priority("SOCIAL_CREATOR", CREATOR_MODELS)
    except Exception:
        creator_models = CREATOR_MODELS

    text, used_model = await _call_llm_chain(
        creator_models,
        [
            {"role": "system", "content": creator_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.5, max_tokens=2048, timeout=70.0
    )
    post = _extract_json(text) if text else None

    if not isinstance(post, dict):
        print("[Social Creator] Falha ao obter JSON do criador. Usando texto bruto/contingência.")
        key_msgs = briefing.get("key_messages") or []
        clean_fallback_body = text or "\n".join(key_msgs) or user_prompt
        # Remove eventuais chaves de JSON bruto vazadas
        clean_fallback_body = re.sub(r'\{\s*"[a-zA-Z_]+".*?\}', '', clean_fallback_body, flags=re.DOTALL).strip()
        if not clean_fallback_body:
            clean_fallback_body = "Com orgulho celebramos essa grande conquista e dedicação dos nossos atletas!"
        post = {
            "platform": spec["label"],
            "objective": briefing.get("objective", "Engajamento"),
            "hook": briefing.get("must_include_text") or (key_msgs[0] if key_msgs else "Destaque do dia"),
            "body_copy": clean_fallback_body[:800],
            "call_to_action": "Comente e compartilhe!",
            "hashtags": briefing.get("hashtags_seed", [])[:6],
            "image_prompt_en": briefing.get("image_prompt_en", ""),
            "visual_style": "Fotografia realista",
            "strategic_tip": "Interaja com os primeiros comentários nos primeiros 30 minutos."
        }
        used_model = used_model or "fallback"

    if exact_headline and isinstance(post, dict):
        post["hook"] = exact_headline

    tools_called.append({
        "action": "social_creator_post",
        "model": used_model,
        "status": "success",
        "duration_seconds": round(time.time() - t0, 2)
    })
    return post


# ---------------------------------------------------------
# Geração de imagem (quando não há imagem do usuário para reusar)
# ---------------------------------------------------------
async def generate_social_image(image_prompt_en: str, platform_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Gera a imagem temática da postagem no enquadramento e resolução ideais da rede social.
    Retorna (public_url, model_name, provider_name).
    """
    spec = PLATFORM_SPECS.get(platform_key, PLATFORM_SPECS["MULTIPLATAFORMA"])
    img_size = spec["image_size"]

    enhanced_prompt = (
        f"{image_prompt_en}, professional photography, clean composition, "
        f"award winning lighting, highly detailed, photorealistic, 8k resolution, no blurry elements"
    )

    candidate_models = [
        ("ImageGen_Pro_OpenAI", "OpenAI (GPT Image 2.5 Flare)"),
        ("ImageGen_Pro_Sunburst", "OpenAI (GPT Image 2.5 Sunburst)"),
        ("ImageGen_Pro_Gemini", "Google (Imagen 3 Pro)"),
        ("ImageGen_Pro_Flux", "Together AI (FLUX.2-dev)"),
        ("ImageGen_Pro", "xAI (Grok Imagine 2.0)"),
        ("ImageGen_Fast_SDXL", "Together AI (SDXL)"),
        ("ImageGen_Fast", "Google Gemini (Flash Lite)")
    ]

    try:
        from admin_config_service import apply_model_priority
        candidate_models = apply_model_priority("SOCIAL_IMAGE", candidate_models, name_index=0)
    except Exception as pri_err:
        print(f"[Social Image] Falha ao aplicar prioridade de modelos: {pri_err}")

    for model_name, provider_name in candidate_models:
        try:
            print(f"[Social Image] Gerando arte para {spec['label']} via {model_name} ({provider_name})...")
            async with httpx.AsyncClient(timeout=90.0) as client:
                payload = {"model": model_name, "prompt": enhanced_prompt, "n": 1}
                if "Flux" in model_name or "OpenAI" in model_name or "SDXL" in model_name:
                    payload["size"] = img_size

                res = await client.post(
                    f"{LITELLM_BASE_URL}/images/generations",
                    json=payload,
                    headers={"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
                )

                if res.status_code == 200:
                    data_items = res.json().get("data", [])
                    if data_items:
                        public_url = await save_image_locally(data_items[0], prefix=f"social_{platform_key.lower()[:5]}")
                        if public_url:
                            print(f"[Social Image] Sucesso com {model_name}: {public_url}")
                            return public_url, model_name, provider_name
                else:
                    print(f"[Social Image] Falha em {model_name} (HTTP {res.status_code}): {res.text[:150]}")
        except Exception as e:
            print(f"[Social Image] Erro ao gerar com {model_name}: {e}")

    return None, None, None


# ---------------------------------------------------------
# Apresentação
# ---------------------------------------------------------
def _fmt_list(items: Any, sep: str = ", ") -> str:
    if not items:
        return "—"
    if isinstance(items, list):
        return sep.join(str(i) for i in items if i)
    return str(items)


def _build_briefing_section(briefing: Dict[str, Any]) -> List[str]:
    """Seção Markdown com o briefing de design (fontes, layout, emojis, imagens complementares etc.)."""
    lines = ["#### 🧠 Briefing do Agente de Pesquisa & Direção de Arte:"]
    domain = briefing.get("domain")
    if domain:
        lines.append(f"- **Domínio:** {domain}")

    facts = briefing.get("verified_facts") or []
    if facts:
        lines.append("- **Fatos verificados:**")
        lines.extend(f"  - {f}" for f in facts[:6])
    missing = briefing.get("unverified_or_missing") or []
    if missing:
        lines.append("- **⚠️ Não confirmado (revise antes de publicar):**")
        lines.extend(f"  - {m}" for m in missing[:4])

    lines.append(f"- **Vocabulário adequado:** {_fmt_list(briefing.get('vocabulary_do'))}")
    lines.append(f"- **Termos evitados:** {_fmt_list(briefing.get('vocabulary_dont'))}")
    lines.append(f"- **Elementos visuais:** {_fmt_list(briefing.get('visual_elements'), '; ')}")
    lines.append(f"- **Imagens complementares sugeridas:** {_fmt_list(briefing.get('complementary_images'), '; ')}")
    lines.append(f"- **Figuras e ícones:** {_fmt_list(briefing.get('figures_icons'), '; ')}")
    lines.append(f"- **Emojis:** {_fmt_list(briefing.get('emojis'), ' ')}")

    fonts = briefing.get("fonts") or {}
    if isinstance(fonts, dict) and fonts:
        lines.append(
            f"- **Tipografia:** título `{fonts.get('headline', '—')}` · apoio `{fonts.get('body', '—')}`"
            + (f" — {fonts.get('rationale')}" if fonts.get("rationale") else "")
        )
    palette = briefing.get("palette") or []
    if palette:
        lines.append(f"- **Paleta:** {' '.join(f'`{c}`' for c in palette)}")
    if briefing.get("layout"):
        lines.append(f"- **Layout:** {briefing['layout']}")

    sources = briefing.get("sources") or []
    if sources:
        lines.append("- **Fontes consultadas:**")
        lines.extend(f"  - {s}" for s in sources[:5])
    return lines


async def execute_social_post_flow(
    user_prompt: str,
    last_user_msg: dict,
    messages: list,
    platform_override: Optional[str] = None,
    mode: str = "FAST"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fluxo multiagente de criação de postagens para redes sociais:
      Agente 1 (Pesquisa/Direção): planejamento, pesquisa web condicional, análise da imagem e briefing.
      Agente 2 (Criador/Modo): execução do post baseado no modo (FAST, SMART, DESIGN).
    """
    start_time = time.time()
    tools_called: List[Dict[str, Any]] = []

    # 0. Plataforma
    platform_key = platform_override or detect_social_platform(user_prompt)
    spec = PLATFORM_SPECS.get(platform_key, PLATFORM_SPECS["MULTIPLATAFORMA"])
    print(f"[Social Flow] Pipeline multiagente para: {spec['label']} (Aspect Ratio: {spec['aspect_ratio']})")

    # 1. Imagem de origem: anexo na mensagem atual tem prioridade; histórico só se o usuário pedir
    current_has_image = isinstance(last_user_msg.get("content"), list) and any(
        _extract_image_url_from_part(p) for p in last_user_msg["content"]
    )
    existing_img_url = find_existing_image_in_history(messages, last_user_msg)
    user_wants_existing = current_has_image or any(k in user_prompt.lower() for k in [
        "imagem", "foto", "story", "post", "arte", "montagem", "composição",
        "na foto", "nessa foto", "nesta foto", "dessa foto", "desta foto",
        "na imagem", "nessa imagem", "nesta imagem", "dessa imagem", "desta imagem",
        "refaça", "refazer", "recrie", "recriar", "edite", "editar", "melhore", "melhorar",
        "mude", "mudar", "altere", "alterar", "troque", "trocar", "ajuste", "ajustar",
        "esse", "este", "novo texto", "texto mais", "mais bonito", "mais bonita",
        "que enviei", "que anexei", "de novo", "novamente", "apenas", "só escreva"
    ])
    use_existing = bool(user_wants_existing and existing_img_url)

    all_raw_images = extract_all_images(messages, last_user_msg)
    source_img_public_urls = []
    if current_has_image:
        current_images = extract_all_images([], last_user_msg)
        for raw_u in current_images:
            pub_u = await ensure_public_image_url(raw_u)
            if pub_u and pub_u not in source_img_public_urls:
                source_img_public_urls.append(pub_u)
    elif use_existing and all_raw_images:
        for raw_u in all_raw_images:
            pub_u = await ensure_public_image_url(raw_u)
            if pub_u and pub_u not in source_img_public_urls:
                source_img_public_urls.append(pub_u)

    source_img_public_url = source_img_public_urls[0] if source_img_public_urls else None

    # 1.5. Guardrail de Intenção Genérica (Orientação Fast vs Smart)
    if len(user_prompt.split()) < 30 and not source_img_public_url:
        from langchain_openai import ChatOpenAI
        from config import LITELLM_BASE_URL, LITELLM_API_KEY
        
        gr_prompt = f"""Você é um auditor de comandos de marketing.
Avalie o seguinte pedido do usuário para criar um post de rede social:
"{user_prompt}"

Verifique se há informações suficientes (um TEMA, ASSUNTO, PRODUTO ou OBJETIVO).
Se o pedido for extremamente genérico (ex: "faça um post para mim", "crie um story pro insta", "faz um carrossel") e não informar SOBRE O QUÊ é o post, responda sufficient: false.
Se houver qualquer indício do assunto, responda sufficient: true.

Responda SOMENTE em JSON válido:
{{
  "sufficient": true | false
}}"""
        try:
            gr_llm = ChatOpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY, model="Classifier_Fast", temperature=0.0, max_tokens=100)
            gr_resp = await gr_llm.ainvoke(gr_prompt)
            match = re.search(r'\{.*\}', gr_resp.content.strip(), re.DOTALL)
            if match:
                gr_parsed = json.loads(match.group(0))
                if not gr_parsed.get("sufficient", True):
                    msg = (
                        "👋 **Notei que o seu pedido está um pouco genérico.**\n\n"
                        "Para que eu possa criar a arte e a copy ideais, por favor me dê mais detalhes sobre o **assunto, produto ou objetivo** do post.\n\n"
                        "💡 **Você pode guiar a criação usando nossos dois modos:**\n"
                        "- ⚡ **Modo Fast (Padrão):** Rápido e objetivo, ideal para posts informativos com uma foto central (que você pode anexar) e um texto direto.\n"
                        "- 🧠 **Modo Smart:** Geração multimodal avançada! Ele compõe artes detalhadas criadas do zero pela IA ou mescla as fotos que você enviar num design estilizado.\n\n"
                        "Me conte sobre o que será o post ou anexe uma imagem para começarmos!"
                    )
                    tools_called.append({"action": "social_generic_guardrail", "status": "blocked", "duration_seconds": 0})
                    return msg, tools_called
        except Exception as e:
            print(f"[Guardrail Social] Erro ao avaliar prompt genérico: {e}")

    # 2. AGENTE 1: pesquisa e briefing (inclui descrição da imagem anexa)
    image_description = None
    if source_img_public_url:
        t0 = time.time()
        image_description = await describe_source_image(source_img_public_url)
        tools_called.append({
            "action": "social_image_analysis",
            "model": "ComboVision",
            "status": "success" if image_description else "failed",
            "duration_seconds": round(time.time() - t0, 2)
        })

    # Extração prioritária determinística de frases solicitadas pelo usuário (aspas, tópicos, triggers 'inclua a frase...')
    exact_headline, exact_subhead = await extract_user_phrases_guardrail(user_prompt, messages)
    if exact_headline:
        tools_called.append({
            "action": "social_exact_phrases_guardrail",
            "headline": exact_headline,
            "subhead": exact_subhead,
            "status": "extracted"
        })
        print(f"[Social Flow] Frases exatas detectadas: Headline='{exact_headline}', Subhead='{exact_subhead}'")

    briefing = await run_research_agent(
        user_prompt=user_prompt,
        platform_key=platform_key,
        image_description=image_description,
        image_available=bool(source_img_public_url),
        tools_called=tools_called,
        exact_headline=exact_headline,
        exact_subhead=exact_subhead
    )

    if exact_headline:
        briefing["must_include_text"] = exact_headline
        briefing["subhead_text"] = exact_subhead or ""

    if mode == "SMART":
        from social_mode_smart import execute_smart_mode
        return await execute_smart_mode(
            user_prompt=user_prompt,
            platform_key=platform_key,
            briefing=briefing,
            source_img_public_urls=source_img_public_urls,
            spec=spec,
            tools_called=tools_called,
            exact_headline=exact_headline,
            exact_subhead=exact_subhead
        )
    elif mode == "DESIGN":
        from social_mode_design import execute_design_mode
        return await execute_design_mode(user_prompt, platform_key, briefing, source_img_public_url, spec, tools_called)

    # 3. AGENTE 2: criador (Modo FAST Default)
    post = await run_creator_agent(
        user_prompt=user_prompt,
        platform_key=platform_key,
        briefing=briefing,
        tools_called=tools_called,
        exact_headline=exact_headline,
        exact_subhead=exact_subhead
    )

    # 4. Imagem final
    public_img_url, img_model, img_provider, img_duration = None, None, None, 0.0
    img_prompt_en = post.get("image_prompt_en") or briefing.get("image_prompt_en") or f"Editorial photography about {user_prompt[:50]}"

    if source_img_public_url:
        public_img_url = source_img_public_url
        img_model = "Existing_Image"
        img_provider = "Imagem do usuário"
        tools_called.append({
            "action": "social_image_reuse",
            "image_url": public_img_url,
            "status": "success"
        })
    else:
        img_start = time.time()
        public_img_url, img_model, img_provider = await generate_social_image(img_prompt_en, platform_key)
        img_duration = round(time.time() - img_start, 2)
        tools_called.append({
            "action": "social_image_generation",
            "model": img_model,
            "provider": img_provider,
            "tier": "Pro" if "Pro" in str(img_model) else "Fast",
            "image_url": public_img_url,
            "status": "success" if public_img_url else "failed_or_skipped",
            "duration_seconds": img_duration
        })

    # 4.5. Fase 2: Composição Visual com Pillow (adiciona textos na imagem)
    if public_img_url:
        from social_image_composer import compose_image
        
        filename = public_img_url.split("/")[-1].split("?")[0]
        local_input_path = os.path.join(IMAGE_STORAGE_DIR, filename)
        composed_filename = f"composed_{filename}"
        local_output_path = os.path.join(IMAGE_STORAGE_DIR, composed_filename)
        
        # Prioridade máxima para as frases solicitadas pelo usuário entre aspas
        if exact_headline:
            headline = exact_headline
            subhead = exact_subhead or ""
        else:
            # Limpa emojis e extrai título principal do hook
            headline_raw = post.get("hook", "").split("\n")[0].strip()
            headline = re.sub(r'[^\w\s,\.\!\?áéíóúãõâêîôûçÀ-Ú]', '', headline_raw).strip()
            subhead_raw = post.get("call_to_action", "").split("\n")[0].strip()
            subhead = re.sub(r'[^\w\s,\.\!\?áéíóúãõâêîôûçÀ-Ú]', '', subhead_raw).strip()
        
        t0_comp = time.time()
        print(f"[Social Flow] Iniciando composição visual em {local_input_path} (Headline: '{headline}', Subhead: '{subhead}')...")
        success_comp = compose_image(local_input_path, local_output_path, headline, subhead, spec)
        
        if success_comp:
            sync_image_to_librechat(local_output_path)
            public_img_url = f"{SERVER_PUBLIC_URL}/{composed_filename}?t={int(time.time())}"
            img_model = f"Composed_{img_model}"
            img_provider = f"{img_provider} + Pillow"
            tools_called.append({
                "action": "social_image_composition",
                "status": "success",
                "duration_seconds": round(time.time() - t0_comp, 2)
            })
            print(f"[Social Flow] Composição visual concluída com sucesso: {public_img_url}")
        else:
            print("[Social Flow] Falha na composição visual. Usando imagem original.")

    # 5. Apresentação
    total_duration = round(time.time() - start_time, 2)

    hook = (post.get("hook") or "").strip()
    body = (post.get("body_copy") or "").strip()
    cta = (post.get("call_to_action") or "").strip()
    hashtags_list = post.get("hashtags", [])
    hashtags_str = " ".join(hashtags_list) if isinstance(hashtags_list, list) else str(hashtags_list)

    objective = post.get("objective") or briefing.get("objective") or "Engajamento"
    style = post.get("visual_style") or "Fotografia Profissional"
    tip = post.get("strategic_tip") or "Interaja nos primeiros 30 minutos."

    full_post_caption = f"{hook}\n\n{body}\n\n👉 {cta}\n\n.\n.\n{hashtags_str}".strip()

    out = []
    out.append(f"### 📱 Estratégia de Conteúdo: {spec['label']}")
    out.append(f"**Objetivo:** `{objective}` | **Formato:** `{spec['dimensions']}` (`{spec['aspect_ratio']}`)\n")

    if public_img_url:
        out.append(f"![Arte para {spec['label']}]({public_img_url})\n")
    else:
        out.append("> ℹ️ *Não foi possível obter a imagem automaticamente. Use o prompt visual abaixo para gerá-la externamente.*\n")

    out.append("#### 📝 Legenda Pronta para Publicação:")
    out.append("```text")
    out.append(full_post_caption)
    out.append("```\n")

    out.extend(_build_briefing_section(briefing))
    out.append("")

    out.append("#### 🎨 Direção de Arte & Métricas:")
    out.append(f"- **Estilo Visual:** {style}")
    if img_model == "Existing_Image":
        out.append("- **Arte Visual:** sua imagem foi utilizada como base da postagem.")
    else:
        out.append(f"- **Prompt Visual (EN):** _{img_prompt_en}_")
        if public_img_url:
            out.append(f"- **Motor Visual:** `{img_provider}` (`{img_model}`) em `{img_duration}s`")
    out.append(f"- **💡 Dica de Crescimento:** {tip}")
    out.append(f"\n_Pipeline multiagente concluído em {total_duration}s_")

    return "\n".join(out), tools_called
