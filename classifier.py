import re
from typing import Tuple, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    OLLAMA_BASE_URL,
    message_has_image,
    history_has_images,
    is_context_dependent_prompt,
    get_conversation_state
)
from prompts import CLASSIFICATION_PROMPT_TEMPLATE

# Instancia o LLM de classificação primário e fallbacks dinâmicos com cache
_classifier_chains: Dict[str, Any] = {}

def get_classifier_chain(model_name: str):
    """Retorna chain de classificação para o modelo especificado (com cache)."""
    if model_name not in _classifier_chains:
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model=model_name,
            temperature=0.0,
            max_tokens=150,
            timeout=8.0
        )
        _classifier_chains[model_name] = CLASSIFICATION_PROMPT_TEMPLATE | llm | StrOutputParser()
    return _classifier_chains[model_name]

classifier_chain = get_classifier_chain("Classifier_Fast")
classifier_chain_fallback = get_classifier_chain("Combo_Free_Chat")


def quick_rule_classifier(prompt: str) -> Optional[str]:
    """Verificação ultrarrápida por regras com Regex (0ms)."""
    p = prompt.lower()
    
    # RAG / Documentos Internos / Memória Temporária de Livros/Textos
    rag_pattern = (
        r"(segundo\s+(meus|os|esse|este)\s+documentos?"
        r"|no\s+(manual|documento|arquivo|pdf|relat[oó]rio|livro|romance|texto(\s+(enviado|anexo|anterior|acima))?)"
        r"|(nesse|neste|do|deste|desse)\s+(texto|documento|arquivo|livro|cap[ií]tulo)"
        r"|de\s+acordo\s+com\s+(o\s+manual|o\s+arquivo|os\s+documentos|o\s+livro|o\s+texto)"
        r"|na\s+base\s+de\s+conhecimento|nos\s+meus\s+arquivos|na\s+mem[oó]ria"
        r"|personagens\s+do\s+livro|resumo\s+d[oe]\s+(livro|texto|documento|arquivo|cap[ií]tulo)"
        r"|analise\s+(esse|este|o)\s+(texto|documento|arquivo)"
        r"|sobre\s+o\s+texto|no\s+texto"
        r"|que\s+(te\s+)?(mandei|enviei|anexei|subi)"
        r"|attached\s+document\(s\)"
        r"|texto\s+(enviado|anexo|acima))"
    )
    if re.search(rag_pattern, p):
        return "DOCUMENT_RAG"

    # Clima / Tempo
    weather_pattern = r"(previs[aã]o\s+d[oe]\s+tempo|previs[aã]o\s+(para|de)|clima\s+em|temperatura\s+em|vai\s+chover|tempo\s+hoje|tempo\s+para\s+hoje|quantos\s+graus)"
    if re.search(weather_pattern, p):
        return "WEATHER_INFO"

    # 1. Indicadores Financeiros / Cotações / Séries Estatísticas / CNPJ / CEP / FIPE -> Obrigatoriamente REALTIME_SEARCH
    realtime_data_pattern = (
        r"\b(cota[cç][aã]o|pre[cç]o|valor)\s+(d[oe]|d[oa]|dos?|das?)?\s*(d[oó]lar|euro|bitcoin|btc|moeda|a[cç][oõ]es|petr4|vale3|bolsa|ibovespa)\b"
        r"|\b(taxa\s+selic|selic|ipca|infla[cç][aã]o|cdi|igp-?m|rendimento\s+da\s+poupan[cç]a)\b"
        r"|\b(d[oó]lar\s+(comercial|hoje|em\s+reais)|euro\s+hoje|bitcoin\s+hoje)\b"
        r"|\b(cnpj|consult(e|ar)?\s+(o\s+)?cnpj|razao\s+social|raz[aã]o\s+social)\b"
        r"|\b(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b"
        r"|\b(cep\s+\d{5}-?\d{3}|consult(e|ar)?\s+(o\s+)?cep|qual\s+(é|e)\s+o\s+endereço\s+do\s+cep)\b"
        r"|\b(tabela\s+fipe|pre[cç]o\s+fipe|valor\s+pela\s+fipe|c[oó]digo\s+fipe)\b"
    )
    if re.search(realtime_data_pattern, p):
        return "REALTIME_SEARCH"

    # 2. Busca em tempo real / Notícias / Eleições / Apuração / Esportes / Lugares e Restaurantes (Fast-Pass 0ms)
    realtime_pattern = (
        r"(not[ií]cias\s+(sobre|de|do|da|hoje)|[uú]ltimas\s+not[ií]cias|cota[cç][aã]o\s+d[oe]"
        r"|resultado\s+d(o\s+jogo|as?\s+elei[cç][oõ]es?|as?\s+apura[cç][oõ]es?)"
        r"|elei[cç][aã]o|elei[cç][oõ]es|apuração\s+d[aeo]|quem\s+ganhou\s+(a\s+elei[cç][aã]o|o\s+jogo)"
        r"|quem\s+foi\s+eleito|governador\s+eleito|senador\s+eleito|prefeito\s+eleito"
        r"|hoje\s+no\s+brasil|pre[cç]o\s+do\s+d[oó]lar|d[oó]lar\s+hoje"
        r"|\b(restaurantes?|pizzarias?|hamburguerias?|bares|onde\s+comer|onde\s+fica|o\s+que\s+fazer\s+em)\s+(em|no|na|de|do|da)\b"
        r"|\b(em\s+2026|de\s+2026|at[eé]\s+o\s+momento|atualmente|neste\s+ano|neste\s+m[eê]s|este\s+ano|ano\s+atual|ranking\s+atual|campeonato\s+de\s+2026|ranking\s+de\s+2026|melhores\s+nadadores|tabela\s+do\s+brasileir[aã]o|jogos\s+de\s+hoje)\b)"
    )
    if re.search(realtime_pattern, p):
        return "REALTIME_SEARCH"

    # 3. Exportador de Tabelas / Planilhas / CSV / Excel / Gráficos para dados sintéticos/conceituais (Fast-Pass 0ms)
    data_export_pattern = (
        r"\b(export(ar|e|ador)?\s+(para|em|pro|o|a)?\s*(excel|csv|planilha|tabela))"
        r"|\b(ger(ar|e|ador)?\s+(um[a]?\s+)?(planilha|csv|excel|\.xlsx|\.csv|gr[aá]fico))"
        r"|\b(baix(ar|e)?\s+(em|como|o|a)?\s*(excel|csv|planilha|\.xlsx))"
        r"|\b(monte|crie|fa[çc]a|construa)\s+(um[a]?\s+)?(planilha|tabela)\s+.*(gr[aá]fico|excel|csv)"
        r"|\b(plote|plotar|gere|gerar|crie|criar|fa[çc]a)\s+(um[a]?\s+)?gr[aá]fico\s+(de\s+barras?|de\s+linhas?|de\s+pizza|de\s+dispers[aã]o|estat[ií]stico|dos?\s+dados?|comparativo)"
        r"|\b(gr[aá]fico\s+e\s+(tabela|planilha|excel|csv))\b"
        r"|\b(tabela\s+e\s+(gr[aá]fico|excel|csv))\b"
        r"|\b(tabela\s+de\s+.*(gr[aá]fico|excel|csv))\b"
        r"|\b(dados\s+em\s+(excel|csv))\b"
    )
    if re.search(data_export_pattern, p):
        return "DATA_EXPORT_CHART"

    # Programação / Código / Scripts (Fast-Pass 0ms)
    coding_pattern = r"(escreva\s+um(a)?\s+(função|script|código|classe|programa|query|sql|regex)|crie\s+um(a)?\s+(script|função|código|api)|em\s+(python|javascript|typescript|bash|sql|html|css|php|java|c\+\+|rust|golang)|como\s+fazer\s+em\s+python|como\s+programar|depurar|debug|dockerfile|docker-compose|código\s+para)"
    if re.search(coding_pattern, p):
        return "CODING"

    # Raciocínio Pesado / Matemática Avançada / Xadrez / Física Teórica (Fast-Pass 0ms)
    heavy_reasoning_pattern = r"(equa[cç][aã]o\s+diferencial|c[aá]lculo\s+(avan[cç]ado|diferencial|integral)|derivada|integral|matriz(es)?|autovalor|autovetor|transformada\s+de\s+(laplace|fourier)|f[ií]sica\s+te[oó]rica|mec[aâ]nica\s+qu[aâ]ntica|relatividade\s+geral|partida\s+de\s+xadrez|jogada\s+de\s+xadrez|enigma\s+matem[aá]tico)"
    if re.search(heavy_reasoning_pattern, p):
        return "HEAVY_REASONING"

    # Raciocínio Lógico / Problemas Matemáticos (Fast-Pass 0ms)
    thinking_pattern = r"(quantos\s+(gatos|ratos|homens|dias|minutos|horas)|se\s+\d+\s+(gatos|pessoas|trabalhadores)|charada|enigma|racioc[ií]nio\s+l[oó]gico|problema\s+de\s+l[oó]gica|resolva\s+a\s+equa[cç][aã]o)"
    if re.search(thinking_pattern, p):
        return "LOCAL_THINKING"

    # Vídeos Curtos / Reels / TikTok / Shorts / Vídeo Vertical (Fast-Pass 0ms)
    short_video_pattern = (
        r"\b(crie|criar|gere|gerar|fa[çc]a|fazer|produza|produzir|elabore|monte)\s+(um[a]?\s+)?(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok|mini\s+v[ií]deo)\b"
        r"|\b(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok)\s+(sobre|de|do|da|com|para)\b"
        r"|\b(roteiro\s+e\s+v[ií]deo|gerador\s+de\s+v[ií]deo|criar\s+um\s+v[ií]deo|gere\s+um\s+v[ií]deo)\b"
    )
    if re.search(short_video_pattern, p):
        return "SHORT_VIDEO"

    # Redes Sociais / Posts / Anúncios / Leads / Campanhas (Fast-Pass 0ms)
    social_networks = r"(instagram|insta|linkedin|facebook|face|twitter|x\b|reddit|threads|tiktok|redes?\s+socia(is|l))"
    social_content = r"(post\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|lead\w*|legenda\w*|carrossel\w*|thread\w*|copy\w*|story|stories|feed)"
    social_preps = r"(para|pro|pra|no|na|de|do|da|em)(\s+(o|a|os|as|um|uma))?"

    social_pattern = (
        rf"{social_content}\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(criar|crie|fa[çc]a|gere|gerar|elabore|escreva|monte|produza)\s+(um[a]?\s+)?({social_content}\s+)?{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|publica\w*|an[uú]nc\w*)\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|an[uú]nc\w*|campanha\w*)\s+(de|para)\s+(capta[cç]\w+|leads?|marketing|vendas|engajamento)\b"
        rf"|\b(roteiro|ideia|estrat[eé]gia)\s+de\s+(post\w*|conte[uú]do)\s+{social_preps}\s+{social_networks}\b"
        rf"|{social_networks}\b.*(com\s+a\s+legenda|utilizando\s+a\s+imagem|use\s+a\s+imagem|hashtag)"
    )
    if re.search(social_pattern, p):
        return "SOCIAL_MEDIA_POST"

    # Geração de Imagem / Arte Visual (Fast-Pass 0ms)
    image_gen_pattern = (
        r"(gere|gerar|crie|criar|desenhe|desenhar|fa[çc]a|produza)\s+(uma?\s+)?(imagem|foto|desenho|ilustra[çc][aã]o|pintura|wallpaper|arte)\b"
        r"|(imagem|foto|desenho|ilustra[çc][aã]o|pintura)\s+d[oe]\s+"
        r"|^(gere|crie|desenhe)\s+.*(imagem|desenho|foto)"
        r"|\b(generate|create|draw)\s+(an?\s+)?(image|photo|picture|illustration|art)\b"
    )
    if re.search(image_gen_pattern, p):
        return "IMAGE_GENERATION"

    # Conversa casual / Saudações / Cortesia / Chat informal (Fast-Pass 0ms)
    casual_pattern = (
        r"^(oi|ol[aá]|bom\s+dia|boa\s+tarde|boa\s+noite|e\s+a[ií]|fala|opa|hello|hi|hey)\b"
        r"([\s,!.?]+(tudo\s+(bem|bom|certo|em\s+cima)(\s+com\s+(voc[eê]|vc|ti))?|como\s+(vai|voc[eê]\s+est[aá]|est[aá]|c[eê]\s+t[aá])|beleza|tranquilo|amigo|assistente|pessoal|a[ií])?)*[\s,!.?]*$"
        r"|^(tudo\s+(bem|bom|certo|em\s+cima)(\s+com\s+(voc[eê]|vc|ti))?|como\s+(vai|voc[eê]\s+est[aá]|est[aá]|c[eê]\s+t[aá])|como\s+vai\s+voc[eê]|como\s+est[aá]s|beleza\??|tranquilo\??|obrigad[oa]|muito\s+obrigad[oa]|valeu|falou|at[eé]\s+(logo|mais|amanh[aã])|tchau)[\s,!.?]*$"
    )
    if re.search(casual_pattern, p.strip(), re.IGNORECASE):
        return "CASUAL_CHAT"

    return None


async def classify_intent(
    user_prompt: str,
    messages: list,
    last_user_msg: dict,
    conversation_id: str = None
) -> Tuple[str, str, Optional[str]]:
    """
    Classifica a intenção da mensagem considerando:
    1. Regras estritas de títulos e anexos de imagem.
    2. Continuidade Contextual de Sessão (preserva a rota anterior se for pergunta de acompanhamento).
    3. Fast-Pass por Regex ultrarrápido (0ms).
    4. Classificador LLM Semântico com fallback resiliente para modelo local.
    """
    has_image_now = message_has_image(last_user_msg)
    has_image_in_history = history_has_images(messages)

    is_title_request = "Provide a concise, 5-word-or-less title" in user_prompt
    routing_method = "llm_classifier"
    raw_classifier_output = None

    edit_keywords_pattern = r"(altere|mude|troque|edite|remova|adicione|coloque|substitua|modifique|transforme|apague|retire|pinte|trocar|mudar|editar|remover|adicionar|retocar|remove|change|replace|edit|modify)\b"
    image_gen_keywords_pattern = (
        r"(gere|gerar|crie|criar|desenhe|desenhar|fa[çc]a|produza)\s+(uma?\s+)?(imagem|foto|desenho|ilustra[çc][aã]o|pintura|wallpaper|arte)\b"
        r"|(imagem|foto|desenho|ilustra[çc][aã]o|pintura)\s+d[oe]\s+"
        r"|^(gere|crie|desenhe)\s+.*(imagem|desenho|foto)"
        r"|\b(generate|create|draw)\s+(an?\s+)?(image|photo|picture|illustration|art)\b"
    )

    # 1. Requisição de Título interna do LibreChat
    if is_title_request:
        routing_method = "title_rule"
        category = "VISION" if has_image_in_history else "CASUAL_CHAT"
        print(f"[Orquestrador] Requisição de Título detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 1.1. Vídeos Curtos / Reels / TikTok / Shorts / Vídeo Vertical
    short_video_pattern = (
        r"\b(crie|criar|gere|gerar|fa[çc]a|fazer|produza|produzir|elabore|monte)\s+(um[a]?\s+)?(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok|mini\s+v[ií]deo)\b"
        r"|\b(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok)\s+(sobre|de|do|da|com|para)\b"
        r"|\b(roteiro\s+e\s+v[ií]deo|gerador\s+de\s+v[ií]deo|criar\s+um\s+v[ií]deo|gere\s+um\s+v[ií]deo)\b"
    )
    if re.search(short_video_pattern, user_prompt.lower()):
        routing_method = "short_video_rule"
        category = "SHORT_VIDEO"
        print(f"[Orquestrador] Vídeo Curto detectado -> Rota: {category}")
        return category, routing_method, None, None

    # 2. Post para Redes Sociais e Leads (precede edição e geração genérica de imagem)
    social_networks = r"(instagram|insta|linkedin|facebook|face|twitter|x\b|reddit|threads|tiktok|redes?\s+socia(is|l))"
    social_content = r"(post\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|lead\w*|legenda\w*|carrossel\w*|thread\w*|copy\w*|story|stories|feed)"
    social_preps = r"(para|pro|pra|no|na|de|do|da|em)(\s+(o|a|os|as|um|uma))?"

    social_keywords_pattern = (
        rf"{social_content}\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(criar|crie|fa[çc]a|gere|gerar|elabore|escreva|monte|produza)\s+(um[a]?\s+)?({social_content}\s+)?{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|publica\w*|an[uú]nc\w*)\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|an[uú]nc\w*|campanha\w*)\s+(de|para)\s+(capta[cç]\w+|leads?|marketing|vendas|engajamento)\b"
        rf"|\b(roteiro|ideia|estrat[eé]gia)\s+de\s+(post\w*|conte[uú]do)\s+{social_preps}\s+{social_networks}\b"
        rf"|{social_networks}\b.*(com\s+a\s+legenda|utilizando\s+a\s+imagem|use\s+a\s+imagem|hashtag)"
    )
    # Com imagem anexa, "post/publicação/legenda" já basta, mesmo sem citar a rede social
    social_with_image_pattern = r"\b(post\w*|publica[cç]\w*|legenda\w*|redes?\s+socia(is|l))\b"
    is_social_request = bool(re.search(social_keywords_pattern, user_prompt.lower())) or (
        (has_image_now or has_image_in_history) and re.search(social_with_image_pattern, user_prompt.lower())
    )
    if is_social_request:
        routing_method = "social_media_rule"
        category = "SOCIAL_MEDIA_POST"
        print(f"[Orquestrador] Post de Redes Sociais detectado -> Rota: {category}")
        return category, routing_method, None, None

    # 3. Edição de Imagem com anexo presente ou recente
    if (has_image_now or has_image_in_history) and re.search(edit_keywords_pattern, user_prompt.lower()):
        routing_method = "image_edit_attachment_rule"
        category = "IMAGE_EDIT"
        print(f"[Orquestrador] Edição de Imagem detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 4. Geração de Imagem Nova
    if re.search(image_gen_keywords_pattern, user_prompt.lower()):
        routing_method = "image_gen_rule"
        category = "IMAGE_GENERATION"
        print(f"[Orquestrador] Geração de Imagem detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 4. Análise de Imagem (Visão pura)
    if has_image_now:
        routing_method = "vision_attachment_rule"
        category = "VISION"
        return category, routing_method, None, None

    # 5. Avaliação do Estado e Continuidade da Conversa
    session_state = get_conversation_state(conversation_id, messages)
    last_category = session_state.get("last_category")
    last_model = session_state.get("last_model")
    is_followup = is_context_dependent_prompt(user_prompt, messages)

    # Se for pergunta de acompanhamento / continuação (ex: 'E para Senador?', 'E amanhã?', 'E em Santos?', 'Faça uma dissertação sobre o tema')
    if is_followup and last_category:
        # Verifica se o usuário não pediu explicitamente uma mudança de domínio via regra rígida
        explicit_rule = quick_rule_classifier(user_prompt)
        if not explicit_rule and last_category in [
            "FACTUAL_GENERAL",
            "REALTIME_SEARCH",
            "WEATHER_INFO",
            "DOCUMENT_RAG",
            "CODING",
            "HEAVY_REASONING",
            "LOCAL_THINKING",
            "SOCIAL_MEDIA_POST",
            "DATA_EXPORT_CHART"
        ]:
            # Se for pedido de planilha ou gráfico, respeita
            if any(k in user_prompt.lower() for k in ["gráfico", "grafico", "planilha", "excel", "csv"]):
                category = "DATA_EXPORT_CHART"
            else:
                category = last_category
            routing_method = "contextual_continuity_rule"
            print(f"[Orquestrador Contexto] Pergunta de acompanhamento detectada ('{user_prompt}'). Mantendo rota anterior: {category} (Modelo anterior: {last_model})")
            return category, routing_method, None, None

    # 6. Fast-Pass por Regras com Regex (0ms)
    rule_category = quick_rule_classifier(user_prompt)
    if rule_category:
        routing_method = "fast_pass_rule"
        category = rule_category
        print(f"[Orquestrador] Classificação por Regra Rápida -> Rota: {category}")
        return category, routing_method, None, None

    # 7. Classificador LLM Semântico com Suporte a Contexto Anterior, Guardrail e Query Rewriting
    routing_method = "llm_classifier"
    category = last_category if (is_followup and last_category) else "FACTUAL_GENERAL"
    rewritten_prompt = None

    # Extrai trecho da última resposta do assistente para contexto
    last_assistant_snippet = ""
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if isinstance(c, str):
                last_assistant_snippet = c[:300]
            elif isinstance(c, list):
                t = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                last_assistant_snippet = " ".join(t)[:300]
            break

    context_str = f"Última resposta do assistente: '{last_assistant_snippet}'" if last_assistant_snippet else "Início da conversa."
    prompt_inputs = {
        "context": context_str,
        "last_category": last_category or "Nenhuma (primeira mensagem)",
        "last_model": last_model or "Nenhum",
        "message": user_prompt
    }

    # Obtém modelos configurados no Dashboard (Padrão: Classifier_Fast -> Combo_Free_Chat)
    try:
        from admin_config_service import get_category_models
        configured_models = get_category_models("SEMANTIC_CLASSIFIER") or ["Classifier_Fast", "Combo_Free_Chat"]
    except Exception:
        configured_models = ["Classifier_Fast", "Combo_Free_Chat"]

    primary_model = configured_models[0] if len(configured_models) > 0 else "Classifier_Fast"
    fallback_model = configured_models[1] if len(configured_models) > 1 else "Combo_Free_Chat"

    raw_classification = None
    try:
        pri_chain = get_classifier_chain(primary_model)
        raw_classification = await pri_chain.ainvoke(prompt_inputs)
    except Exception as llm_err:
        print(f"[Aviso] Classificador primário ({primary_model}) falhou: {llm_err}. Tentando fallback ({fallback_model})...")
        try:
            fb_chain = get_classifier_chain(fallback_model)
            raw_classification = await fb_chain.ainvoke(prompt_inputs)
        except Exception as fb_err:
            print(f"[Aviso] Classificador secundário ({fallback_model}) também falhou: {fb_err}.")

    llm_followup_detected = False
    if raw_classification:
        raw_classifier_output = raw_classification
        clean_raw = re.sub(r"(?s)<think>.*?</think>", "", raw_classification)

        # 1. Parse de IS_FOLLOWUP
        if re.search(r"IS_FOLLOWUP:\s*TRUE", clean_raw, re.IGNORECASE):
            llm_followup_detected = True

        # 2. Parse de REWRITTEN_PROMPT
        rw_match = re.search(r"REWRITTEN_PROMPT:\s*(.+)", clean_raw, re.IGNORECASE)
        if rw_match:
            candidate_rw = rw_match.group(1).strip()
            # Valida se a reescrita é válida e não repetiu a chave CATEGORY
            if candidate_rw and "CATEGORY:" not in candidate_rw and len(candidate_rw) > 3:
                rewritten_prompt = candidate_rw

        # 3. Parse de CATEGORY
        cat_match = re.search(r"CATEGORY:\s*([A-Z_]+)", clean_raw)
        if cat_match:
            category = cat_match.group(1).strip()
        else:
            cat_matches = re.findall(
                r"\b(DOCUMENT_RAG|WEATHER_INFO|REALTIME_SEARCH|CASUAL_CHAT|CODING|LOCAL_THINKING|HEAVY_REASONING|FACTUAL_GENERAL|VISION|IMAGE_GENERATION|IMAGE_EDIT|SOCIAL_MEDIA_POST|DATA_EXPORT_CHART)\b",
                clean_raw.upper()
            )
            if cat_matches:
                category = cat_matches[-1]
            else:
                raw_upper = clean_raw.upper()
                if any(w in raw_upper for w in ["SOCIAL_MEDIA", "POSTAGEM", "POST", "INSTAGRAM", "LINKEDIN", "REDE SOCIAL"]):
                    category = "SOCIAL_MEDIA_POST"
                elif any(w in raw_upper for w in ["SAUDAÇÃO", "SAUDACAO", "CUMPRIMENTO", "CONVERSA INFORMAL", "CASUAL"]):
                    category = "CASUAL_CHAT"
                elif any(w in raw_upper for w in ["PROGRAMAÇÃO", "PROGRAMACAO", "CÓDIGO", "CODIGO", "SCRIPT"]):
                    category = "CODING"
                elif any(w in raw_upper for w in ["CLIMA", "TEMPO", "PREVISÃO", "METEOROLOGIA"]):
                    category = "WEATHER_INFO"
                elif any(w in raw_upper for w in ["NOTÍCIA", "NOTICIA", "ELEIÇÃO", "ELEICAO", "COTAÇÃO"]):
                    category = "REALTIME_SEARCH"

        print(f"[Debug Classificador LLM] Retorno bruto: '{raw_classification}' -> Rota final: {category} | LLM Follow-up: {llm_followup_detected} | Reescrita: {rewritten_prompt}")
    else:
        # Se todos os classificadores falharam
        if is_followup and last_category:
            category = last_category
            routing_method = "llm_error_context_fallback"
            print(f"[Aviso] Classificador LLM indisponível. Usando fallback contextual da categoria anterior: {category}")
        else:
            category = "FACTUAL_GENERAL"
            routing_method = "llm_error_default_fallback"
            print(f"[Aviso] Classificador LLM indisponível. Usando fallback padrão: FACTUAL_GENERAL.")

    # Guardrail adicional: Se o modelo identificou follow-up, mas a categoria anterior era FACTUAL_GENERAL e o usuário não pediu código/análise de arquivo
    if (is_followup or llm_followup_detected) and last_category == "FACTUAL_GENERAL" and category == "DOCUMENT_RAG":
        has_file_mention = any(k in user_prompt.lower() for k in ["pdf", "arquivo", "documento", "manual", "anexo", "texto enviado"])
        if not has_file_mention:
            print(f"[Guardrail Classificador] Bloqueando desvio para DOCUMENT_RAG em continuidade de FACTUAL_GENERAL.")
            category = "FACTUAL_GENERAL"

    # Proteção de Continuação Contextual: Se for instrução de acompanhamento sem menção a código, não enviar para CODING
    if category == "CODING" and (is_followup or llm_followup_detected):
        coding_explicit = re.search(r"(código|script|programa|python|sql|java|regex|função|function|dev|docker)\b", user_prompt.lower())
        if not coding_explicit:
            category = last_category if (last_category and last_category != "CODING") else "FACTUAL_GENERAL"
            print(f"[Orquestrador Contexto] Instrução '{user_prompt}' é continuação de conversa/dados, não programação. Rota ajustada para: {category}.")

    return category, routing_method, raw_classifier_output, rewritten_prompt
