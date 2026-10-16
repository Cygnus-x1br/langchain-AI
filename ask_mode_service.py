import os
import re
import json
import time
import asyncio
from typing import Tuple, Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY
)

ASK_MODE_PROMPT = """Você é o Assistente Sênior de Marketing e Conteúdo Digital do LibreChat.
O usuário solicitou a criação de uma publicação para redes sociais com a seguinte instrução:
"{user_prompt}"

O tema identificado é: "{user_topic}".

Sua missão é gerar uma mensagem rápida, acolhedora, moderna e elegante apresentando 4 opções de formato para o usuário escolher ANTES de produzir a peça final.

ESTRUTURA OBRIGATÓRIA DA SUA RESPOSTA:
1. Uma frase curta e engajadora reconhecendo o tema e confirmando que você vai estruturar a publicação com excelência.
2. O menu com exatamente as 4 opções abaixo, numeradas com os emojis correspondentes. Adapte suavemente as descrições ou exemplos entre parênteses para conectar com o tema do usuário:
   1️⃣ **Apenas Copy e Estratégia:** Texto persuasivo completo (headline de impacto, legenda, chamada para ação e hashtags nichadas), sem gerar imagens (ideal se você já possui suas fotos).
   2️⃣ **Arte Conceitual IA:** Criação de uma imagem artística ou fotorrealista exclusiva via IA generativa (Flux / DALL-E) acompanhada da copy completa.
   3️⃣ **Design Gráfico / Cartaz:** Composição visual moderna com tipografia marcante e destaque de frases/título, ideal para anúncios ou posts promocionais.
   4️⃣ **Modo Carrossel:** Roteiro educativo estruturado slide a slide (capa com gancho, conteúdo e CTA final) para alto engajamento no feed.
3. Uma linha final convidando o usuário a responder com o número da opção desejada (1, 2, 3 ou 4) ou detalhar qualquer preferência de cores/estilo (lembrando que ele também pode anexar fotos próprias se desejar).

DIRETRIZES:
- Mantenha a resposta concisa, limpa e altamente profissional.
- Use negrito para destacar os nomes das opções.
- NUNCA invente opções além das 4 especificadas acima."""


def extract_social_topic(prompt: str) -> str:
    """Extrai o tema central de um pedido de post para personalizar as perguntas do Ask-Mode."""
    p = prompt.strip()
    # Remove gatilhos de criação comuns
    patterns = [
        r"(?i)^(crie|fa[çc]a|elabore|gere|monte|escreva|produza)\s+(um[a]?\s+)?(post|postagem|publica[cç][aã]o|story|stories|reels|an[uú]ncio|conte[uú]do)\s+(para\s+(o\s+)?(instagram|linkedin|facebook|feed|insta|rede\s+social)\s+)?(sobre|de|com\s+o\s+tema|focado\s+em|falando\s+sobre)?:?\s*",
        r"(?i)^(post|postagem|publica[cç][aã]o)\s+(para\s+(o\s+)?(instagram|linkedin|facebook|feed)\s+)?(sobre|de)?:?\s*"
    ]
    for pat in patterns:
        p = re.sub(pat, "", p).strip()
    
    # Se sobrou um texto razoável
    clean_topic = p.strip('".:;,- ')
    return clean_topic if len(clean_topic) >= 3 else prompt


def is_ask_mode_eligible(user_prompt: str, messages: list, num_images: int) -> bool:
    """
    Determina se a mensagem de rede social deve acionar o Ask-Mode interativo.
    Retorna True se for um pedido novo e genérico/aberto onde o estilo da peça não foi especificado.
    """
    if num_images > 0:
        return False

    p_lower = user_prompt.lower()

    # 1. Se for reedição, ajuste ou refinamento de imagem/post anterior, executa direto
    refinement_keywords = [
        "refaça", "refazer", "recrie", "recriar", "edite", "editar",
        "melhore", "melhorar", "mude", "mudar", "altere", "alterar",
        "troque", "trocar", "ajuste", "ajustar", "reedite", "reeditar",
        "mais bonito", "mais bonita", "outro texto", "mude o texto",
        "mude a frase", "troque a frase", "coloque a frase"
    ]
    if any(k in p_lower for k in refinement_keywords):
        return False

    # 2. Se o usuário já especificou expressamente o modo ou formato desejado no próprio prompt:
    explicit_specs = [
        # Carrossel
        "carrossel", "carousel", "slides", "em slides", "sequencial",
        # Apenas copy / sem imagem
        "apenas copy", "apenas a copy", "apenas texto", "apenas o texto", "sem imagem",
        "sem foto", "só o texto", "só a copy", "só legenda", "apenas legenda", "texto puro",
        # Design / Cartaz / Tipografia
        "modo design", "estilo design", "cartaz", "poster", "tipografia", "design gráfico",
        # Smart multimodal explícito
        "modo smart", "smart mode", "composição", "componha", "fundo especial",
        # Geração visual específica / IA
        "imagem realista", "fotorrealista", "imagem ia", "foto de um", "foto de uma",
        "ilustração 3d", "desenhe", "gerar uma imagem", "gere uma foto"
    ]
    if any(k in p_lower for k in explicit_specs):
        return False

    # 3. Se for prompt excessivamente longo e detalhado (> 60 palavras), assume que já contém briefing fechado
    words = user_prompt.split()
    if len(words) > 60:
        return False

    return True


def is_ask_mode_reply(
    user_prompt: str,
    messages: list,
    session_state: dict
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """
    Verifica se a mensagem atual do usuário é uma resposta às opções do Ask-Mode.
    Retorna:
      (is_reply, selected_mode, refined_prompt, original_topic)
    """
    p_lower = user_prompt.strip().lower()
    
    # Verifica se a mensagem imediatamente anterior do assistente foi um menu do Ask-Mode
    last_assistant_text = ""
    if messages and len(messages) >= 2:
        for m in reversed(messages[:-1]):
            if m.get("role") == "assistant":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_assistant_text = content
                elif isinstance(content, list):
                    last_assistant_text = " ".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])
                break

    ask_mode_markers = [
        "1️⃣ **apenas copy",
        "2️⃣ **arte conceitual",
        "3️⃣ **design gráfico",
        "4️⃣ **modo carrossel",
        "como você prefere a peça",
        "escolha de estilo social",
        "como prefere a publicação"
    ]
    is_prev_ask_mode = any(m.lower() in last_assistant_text.lower() for m in ask_mode_markers)
    has_pending_flag = bool(session_state.get("pending_ask_mode"))

    if not (is_prev_ask_mode or has_pending_flag):
        return False, None, None, None

    # Resgata o prompt social original da sessão anterior
    original_prompt = (
        session_state.get("pending_social_prompt")
        or session_state.get("last_prompt")
        or ""
    )
    if not original_prompt and messages and len(messages) >= 3:
        # Busca a penúltima mensagem de usuário
        user_msgs = [m for m in messages[:-1] if m.get("role") == "user"]
        if user_msgs:
            c = user_msgs[-1].get("content", "")
            original_prompt = c if isinstance(c, str) else ""

    topic = session_state.get("pending_topic") or extract_social_topic(original_prompt)

    # Identificação da opção escolhida pelo usuário
    selected_mode = None
    mode_instruction = ""

    # Opção 1: Apenas Copy
    if re.search(r"^(1|opção 1|opcao 1|1️⃣)\b", p_lower) or any(k in p_lower for k in ["apenas copy", "apenas texto", "sem imagem", "só texto", "só a legenda"]):
        selected_mode = "COPY_ONLY"
        mode_instruction = "Gere exclusivamente a copy estratégica completa (headline, corpo, CTA e hashtags), SEM gerar imagens."

    # Opção 2: Arte Conceitual IA (Flux / DALL-E)
    elif re.search(r"^(2|opção 2|opcao 2|2️⃣)\b", p_lower) or any(k in p_lower for k in ["arte conceitual", "imagem ia", "foto ia", "fotorrealista", "com imagem", "com foto"]):
        selected_mode = "FAST"
        mode_instruction = "Gere a copy estratégica e uma imagem artística/fotorrealista conceitual via IA generativa para ilustrar a publicação."

    # Opção 3: Design Gráfico / Cartaz / Tipografia
    elif re.search(r"^(3|opção 3|opcao 3|3️⃣)\b", p_lower) or any(k in p_lower for k in ["design", "cartaz", "tipografia", "design gráfico", "poster", "estilo design"]):
        selected_mode = "DESIGN"
        mode_instruction = "Gere a copy estratégica e componha um design gráfico estilo cartaz moderno com tipografia marcante e destaque de frases."

    # Opção 4: Carrossel
    elif re.search(r"^(4|opção 4|opcao 4|4️⃣)\b", p_lower) or any(k in p_lower for k in ["carrossel", "carousel", "slides", "em slides"]):
        selected_mode = "CAROUSEL"
        mode_instruction = "Gere um post em formato carrossel educativo estruturado slide a slide com gancho forte na capa e CTA na conclusão."

    # Se o usuário digitou uma resposta curta como "2 com fundo azul"
    if selected_mode:
        extra_detail = user_prompt.strip()
        # Remove "1", "2", "3", "4", "opção X" do detalhe extra se for só o número
        extra_detail = re.sub(r"^(op[çc][aã]o\s*)?[1-4][️⃣\.:\-\s]*", "", extra_detail, flags=re.IGNORECASE).strip()
        
        combined_prompt = f"{original_prompt}. [MODO SELECIONADO PELO USUÁRIO NO ASK-MODE: {selected_mode}]. {mode_instruction}"
        if extra_detail:
            combined_prompt += f" Preferência adicional do usuário: {extra_detail}"
            
        return True, selected_mode, combined_prompt, topic

    return False, None, None, None


async def generate_social_ask_mode_message(user_prompt: str) -> str:
    """
    Invoca o modelo conversacional rápido (Combo_Free_Chat / Groq Llama 3.3 70B / Gemini Flash)
    para produzir a resposta interativa de Ask-Mode perfeitamente adaptada ao tema do usuário.
    """
    user_topic = extract_social_topic(user_prompt)
    prompt_text = ASK_MODE_PROMPT.format(
        user_prompt=user_prompt,
        user_topic=user_topic
    )

    candidate_models = ["Combo_Free_Chat", "Classifier_Fast", "ComboFree_Smart"]

    for mod in candidate_models:
        try:
            llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model=mod,
                temperature=0.3,
                max_tokens=450,
                timeout=5.0
            )
            response = await llm.ainvoke(prompt_text)
            content = response.content if hasattr(response, "content") else str(response)
            
            # Limpa possíveis blocos <think>
            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()

            if content and len(content.strip()) > 80:
                print(f"[Ask-Mode Social] Pergunta interativa gerada com sucesso via '{mod}' para o tema: '{user_topic}'.")
                return content.strip()
        except Exception as e:
            print(f"[Aviso Ask-Mode] Falha ao gerar com modelo '{mod}': {e}. Tentando próximo...")

    # Fallback Elegante Instantâneo (Zero Downtime)
    fallback_message = (
        f"Com certeza! Vou estruturar uma publicação de alto impacto sobre **{user_topic}**.\n\n"
        "Antes de gerar os arquivos visuais, **como você prefere a sua peça?**\n\n"
        "1️⃣ **Apenas Copy e Estratégia:** Texto persuasivo completo (gancho, legenda, chamada para ação e hashtags), sem gerar imagens (ideal se você já possui suas fotos).\n"
        "2️⃣ **Arte Conceitual IA:** Imagem fotorrealista exclusiva criada por IA generativa (Flux / DALL-E) acompanhada da copy completa.\n"
        "3️⃣ **Design Gráfico / Cartaz:** Composição visual moderna com tipografia elegante e destaque de frases/título, ideal para anúncios e feed.\n"
        "4️⃣ **Modo Carrossel:** Roteiro educativo estruturado slide a slide (capa, conteúdo e CTA final) para alta retenção.\n\n"
        "💬 *Basta responder com o número da opção (1, 2, 3 ou 4) ou me dizer seu estilo/cores de preferência (você também pode anexar fotos próprias se desejar)!*"
    )
    return fallback_message
