"""
phrase_extractor_guardrail.py - Filtro e Guardrail Determinístico de Frases
==========================================================================
Identifica e extrai deterministicamente as frases solicitadas pelo usuário para
inserção em artes, títulos, legendas e composições de redes sociais.

Suporta:
1. Aspas duplas, simples, tipográficas e guillemets ("...", '...', “...”, «...»).
2. Listas em tópicos/bullets (- Frase 1, * Frase 2, 1. Frase 1, 2. Frase 2).
3. Padrões de gatilho no texto corrido ("inclua a frase...", "com o texto...", "título:...", etc.).
4. Separação por vírgula ou conjunção "e" / "subtítulo" / "rodapé".
5. Inspeção retroativa de histórico quando o usuário pede reedição ("as frases solicitadas").
6. Fallback estruturado ultra-rápido via LLM se o texto for ambíguo sem aspas.
"""

import re
import json
from typing import Tuple, List, Optional, Dict, Any


TRIGGER_KEYWORDS = [
    r"(?:inclua|incluir|adicione|adicionar|coloque|colocar|insira|inserir|ponha|bote|escreva|escrever|use|usando|utilize|utilizando|com)\s+(?:a\s+frase|as\s+frases|o\s+texto|os\s+textos|os\s+dizeres|o\s+t[ií]tulo|os\s+t[ií]tulos|a\s+seguinte\s+frase|as\s+seguintes\s+frases)",
    r"(?:altere|alterar|mude|mudar|troque|trocar|substitua|substituir)\s+(?:a\s+frase|as\s+frases|o\s+texto|os\s+textos|o\s+t[ií]tulo)\s+(?:para|por)",
    r"(?:inclua|coloque|insira|escreva)\s+(?:na\s+arte|na\s+foto|na\s+imagem|no\s+post)\s+(?:a\s+frase|o\s+texto|os\s+textos|os\s+dizeres)",
]

IGNORE_META_WORDS = [
    "simples", "smart", "fast", "post", "refaça", "imagem", "foto", "post acima",
    "post anterior", "postagem", "arte", "banner", "feed", "story", "stories",
    "provedor", "modelo", "diretriz", "aprendizado", "instrução", "prompt", "resposta"
]

INSTRUCTION_COMMAND_PATTERNS = [
    r"^(?:faça|fazer|monte|montar|crie|criar|gere|gerar|produza|produzir)\s+(?:um[a]?\s+)?(?:post|postagem|arte|montagem|foto|imagem|design|story|feed|carrossel|banner)",
    r"^(?:quero|gostaria\s+de)\s+(?:postar|um[a]?\s+post|um[a]?\s+arte|um[a]?\s+montagem|uma?\s+foto)",
    r"(?:para\s+postar|pro\s+feed|pro\s+insta|no\s+instagram|no\s+feed|no\s+stories|no\s+linkedin)",
    r"^(?:tire|tirar|remova|remover|apague|apagar)\s+(?:tudo|o\s+texto|as\s+letras|as\s+frases|o\s+que\s+estiver\s+escrito)",
    r"^(?:deixe|deixar)\s+(?:só|apenas|somente)\s+(?:a\s+foto|a\s+imagem|as\s+fotos)",
    r"^(?:mantenha|manter)\s+(?:a\s+mesma\s+arte|a\s+mesma\s+foto|a\s+foto)",
    r"^(?:refaça|refazer|recrie|recriar|edite|editar|ajuste|ajustar)\s+(?:o\s+post|a\s+edição|a\s+foto|a\s+imagem)",
    r"(?:dos?\s+meus\s+filhos|das?\s+minhas\s+fotos|minha\s+foto|minhas\s+fotos|meus\s+filhos|minha\s+fam[ií]lia)",
    r"(?:utilize|use|usando)\s+o\s+modelo",
]

NO_TEXT_PATTERNS = [
    r"(?:tire|tirar|remova|remover|apague|apagar)\s+tudo(?:\s+o\s+que\s+estiver\s+escrito)?",
    r"(?:deixe|deixar)\s+(?:só|apenas|somente)\s+(?:a\s+foto|a\s+imagem|as\s+fotos)",
    r"\b(?:sem\s+texto|sem\s+nada\s+escrito|sem\s+frase|sem\s+legenda|limpe\s+o\s+texto|sem\s+letras|deixe\s+sem\s+texto)\b"
]


def is_instruction_command(phrase: str) -> bool:
    """Verifica se uma frase é na verdade um comando/instrução de prompt em vez de um texto para ser impresso."""
    if not phrase:
        return True
    p = phrase.lower().strip()
    if p in IGNORE_META_WORDS or len(p) < 2:
        return True
    for pat in INSTRUCTION_COMMAND_PATTERNS:
        if re.search(pat, p):
            return True
    return False


def is_no_text_intent(text: str) -> bool:
    """Detecta se o usuário pediu expressamente para não incluir texto ou remover textos da imagem."""
    if not text:
        return False
    p = text.lower()
    return any(re.search(pat, p) for pat in NO_TEXT_PATTERNS)


def _clean_phrase(text: str) -> str:
    """Remove pontuações residuais de borda e espaços excessivos."""
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r'^[\s\-_,:;.\'\"“«\*`]+|[\s\-_,:;.\'\"”»\*`]+$', '', t)
    return t.strip()


def extract_phrases_from_text(text: str) -> List[str]:
    """Extrai frases candidatas de uma única mensagem de forma puramente determinística."""
    if not text:
        return []

    # Limpeza defensiva contra blocos sistêmicos injetados pelo orquestrador
    text = re.sub(r'\[DIRETRIZ DE APRENDIZADO.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[CONTEXTO & PREFERÊNCIAS.*', '', text, flags=re.DOTALL)
    text = re.sub(r'\[AVISO.*', '', text, flags=re.DOTALL).strip()
    if not text:
        return []

    # Se a intenção explícita for remover texto / deixar só a foto, não há frase para renderizar
    if is_no_text_intent(text):
        return []

    # 0. Checagem prioritária de substituição explícita ("substitua 'A' por 'B'", "troque 'A' por 'B'")
    sub_match = re.search(
        r'(?:substitua|troque|mude|altere)\s+(?:apenas\s+|só\s+)?(?:a\s+frase|o\s+texto)?\s*["“\'«]([^"”\'»\n]+)["”\'»]\s*(?:por|para)\s*["“\'«]([^"”\'»\n]+)["”\'»]',
        text,
        re.IGNORECASE
    )
    if sub_match:
        new_p = _clean_phrase(sub_match.group(2))
        if new_p and not is_instruction_command(new_p):
            return [new_p]

    # 0.1 Checagem de alteração de frase ("altere a frase para: X", "mude a frase para X")
    alt_match = re.search(
        r'(?:altere|alterar|mude|mudar|troque|trocar|substitua|substituir)\s+(?:a\s+frase|o\s+texto|o\s+t[ií]tulo)\s+(?:para|por)\s*[:=-]?\s*["“\'«]?([^"”\'»\n]+?)["”\'»]?(?:\s*$|\s+mas|\s+mantendo|\.)',
        text,
        re.IGNORECASE
    )
    if alt_match:
        cand = _clean_phrase(alt_match.group(1))
        if cand and not is_instruction_command(cand):
            return [cand]

    # 1. Busca por aspas (duplas, simples, tipográficas, guillemets)
    quote_matches = re.findall(r'["“\'«]([^"”\'»\n]{2,100})["”\'»]', text)
    clean_quotes = [_clean_phrase(q) for q in quote_matches if _clean_phrase(q)]
    valid_quotes = [q for q in clean_quotes if not is_instruction_command(q) and not any(m in q.lower() for m in ["provedor:", "modelo:"])]
    if valid_quotes:
        return valid_quotes

    # 2. Busca por tópicos / bullets explícitos
    # Ex:
    # - Troféu Kim Mollo 2026
    # - 3º lugar - 100m borbo
    bullet_matches = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*([^\n]{2,100})', text)
    clean_bullets = [_clean_phrase(b) for b in bullet_matches if _clean_phrase(b)]
    valid_bullets = [b for b in clean_bullets if b.lower() not in IGNORE_META_WORDS and len(b) >= 2 and not any(m in b.lower() for m in ["provedor:", "modelo:"])]
    if valid_bullets:
        return valid_bullets

    # 3. Busca por campos explícitos de Título e Subtítulo
    # Ex: "Título: Troféu Kim Mollo 2026 | Subtítulo: 3º lugar - 100m borbo"
    # ou "Texto 1: ... Texto 2: ..."
    title_sub_matches = []
    t_match = re.search(r'(?:t[ií]tulo|texto\s*1|frase\s*1)\s*[:=-]\s*([^\n,;|]+)', text, re.IGNORECASE)
    if t_match:
        title_sub_matches.append(_clean_phrase(t_match.group(1)))
    s_match = re.search(r'(?:subt[ií]tulo|texto\s*2|frase\s*2|coloca[cç][aã]o|rodap[eé])\s*[:=-]\s*([^\n,;|]+)', text, re.IGNORECASE)
    if s_match:
        title_sub_matches.append(_clean_phrase(s_match.group(1)))
    if title_sub_matches:
        return title_sub_matches

    # 4. Busca por padrões de gatilho no texto corrido ("inclua a frase...", "com o texto...")
    for trig in TRIGGER_KEYWORDS:
        m = re.search(trig + r'[:\s]+([^\n.]+)', text, re.IGNORECASE)
        if m:
            raw_segment = m.group(1).strip()
            # Remove palavras iniciais desnecessárias como "que seja", "dizendo"
            raw_segment = re.sub(r'^(?:que\s+diga|que\s+seja|dizendo|sobre)\s+', '', raw_segment, flags=re.IGNORECASE).strip()
            
            # Se tiver separação por ' e ' ou ' com subtítulo ' ou vírgula
            parts = re.split(r'\s+e\s+(?:o\s+subt[ií]tulo|a\s+frase|o\s+texto|o\s+rodap[eé]|coloca[cç][aã]o|subt[ií]tulo)?\s*|,\s*(?:e\s+)?|\s*;\s*', raw_segment)
            extracted = [_clean_phrase(p) for p in parts if _clean_phrase(p)]
            
            # Filtra palavras meta ou de referência ao histórico
            def _is_valid_phrase(phrase: str) -> bool:
                p_low = phrase.lower()
                if p_low in IGNORE_META_WORDS or len(p_low) < 2:
                    return False
                # Se for referência a algo já falado ("solicitadas", "anteriores", "acima")
                if any(ref in p_low for ref in [
                    "solicitad", "anterior", "acima", "abaixo", "que pedi", "que passei",
                    "que enviei", "mesma frase", "mesmas frases", "novamente"
                ]):
                    return False
                return True

            filtered = [p for p in extracted if _is_valid_phrase(p)]
            if filtered:
                return filtered

    return []


async def extract_phrases_with_llm_guardrail(text: str) -> List[str]:
    """Fallback por LLM ultrarrápido (Classifier_Fast) para prompts ambíguos sem aspas."""
    from langchain_openai import ChatOpenAI
    from config import LITELLM_BASE_URL, LITELLM_API_KEY

    prompt = f"""Você é um extrator semântico de textos exatos para artes gráficas.
Analise a mensagem do usuário e extraia ESTRITAMENTE as frases que o usuário pediu para incluir/escrever na imagem ou post.

MENSAGEM:
"{text}"

Se o usuário especificou frases ou textos para estarem na arte (mesmo sem aspas), retorne em JSON:
{{
  "has_specific_phrases": true,
  "phrases": ["Título Principal", "Subtítulo ou Segunda Frase"]
}}
Se o usuário NÃO especificou textos exatos a serem escritos:
{{
  "has_specific_phrases": false,
  "phrases": []
}}
Retorne APENAS o JSON."""

    try:
        llm = ChatOpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY, model="Classifier_Fast", temperature=0.0, max_tokens=150, timeout=10.0)
        resp = await llm.ainvoke(prompt)
        c = resp.content.strip()
        m = re.search(r'\{.*\}', c, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if data.get("has_specific_phrases") and data.get("phrases"):
                return [_clean_phrase(p) for p in data["phrases"] if _clean_phrase(p)]
    except Exception as e:
        print(f"[Phrase Extractor Guardrail] Erro no fallback LLM: {e}")
    return []


def extract_phrase_intent(text: str) -> Dict[str, Any]:
    """
    Analisa semanticamente e deterministicamente a intenção do usuário em relação a textos na arte:
    - remove_all_text: "Tire tudo que estiver escrito", "Deixe só a foto", "Sem texto"
    - replace_phrase: "substitua 'A' por 'B'"
    - set_phrase: "altere a frase para: 'B'"
    - remove_word: "remova a palavra 'A'"
    - explicit_phrases: aspas, tópicos ou 'inclua a frase X'
    """
    if not text:
        return {"action": "none", "headline": None, "subhead": None, "old_phrase": None, "remove_word": None, "remove_all_text": False, "has_explicit_text": False}

    # 1. Detecção de remoção total de texto
    if is_no_text_intent(text):
        return {
            "action": "remove_all_text",
            "headline": None,
            "subhead": None,
            "old_phrase": None,
            "remove_word": None,
            "remove_all_text": True,
            "has_explicit_text": False
        }

    # 2. Detecção de substituição de frase ("substitua 'A' por 'B'")
    sub_match = re.search(
        r'(?:substitua|troque|mude|altere)\s+(?:apenas\s+|só\s+)?(?:a\s+frase|o\s+texto)?\s*["“\'«]([^"”\'»\n]+)["”\'»]\s*(?:por|para)\s*["“\'«]([^"”\'»\n]+)["”\'»]',
        text,
        re.IGNORECASE
    )
    if sub_match:
        old_p = _clean_phrase(sub_match.group(1))
        new_p = _clean_phrase(sub_match.group(2))
        return {
            "action": "replace_phrase",
            "headline": new_p,
            "subhead": None,
            "old_phrase": old_p,
            "remove_word": None,
            "remove_all_text": False,
            "has_explicit_text": True
        }

    # 3. Detecção de remoção de palavra pontual ("remova apenas a palavra 'montagem'")
    rem_word_match = re.search(
        r'(?:remova|tire|apague)\s+(?:do\s+post\s+anterior\s+)?(?:apenas\s+|só\s+)?(?:a\s+palavra|o\s+termo)\s*["“\'«]?([^"”\'»\n.]+?)["”\'»]?(?:\s*$|\.|\s+utilize|\s+use)',
        text,
        re.IGNORECASE
    )
    if rem_word_match:
        w = _clean_phrase(rem_word_match.group(1))
        return {
            "action": "remove_word",
            "headline": None,
            "subhead": None,
            "old_phrase": None,
            "remove_word": w,
            "remove_all_text": False,
            "has_explicit_text": False
        }

    # 4. Detecção de alteração/definição de nova frase ("altere a frase para: X")
    alt_match = re.search(
        r'(?:altere|alterar|mude|mudar|troque|trocar|substitua|substituir)\s+(?:a\s+frase|o\s+texto|o\s+t[ií]tulo)\s+(?:para|por)\s*[:=-]?\s*["“\'«]?([^"”\'»\n]+?)["”\'»]?(?:\s*$|\s+mas|\s+mantendo|\.)',
        text,
        re.IGNORECASE
    )
    if alt_match:
        cand = _clean_phrase(alt_match.group(1))
        if cand and not is_instruction_command(cand):
            return {
                "action": "set_phrase",
                "headline": cand,
                "subhead": None,
                "old_phrase": None,
                "remove_word": None,
                "remove_all_text": False,
                "has_explicit_text": True
            }

    # 5. Extração padrão de frases do texto (aspas, tópicos, triggers)
    phrases = extract_phrases_from_text(text)
    if phrases:
        h = phrases[0] if not is_instruction_command(phrases[0]) else None
        s = phrases[1] if len(phrases) > 1 and not is_instruction_command(phrases[1]) else None
        if h:
            return {
                "action": "set_phrase",
                "headline": h,
                "subhead": s,
                "old_phrase": None,
                "remove_word": None,
                "remove_all_text": False,
                "has_explicit_text": True
            }

    return {
        "action": "none",
        "headline": None,
        "subhead": None,
        "old_phrase": None,
        "remove_word": None,
        "remove_all_text": False,
        "has_explicit_text": False
    }


async def extract_user_phrases_guardrail(user_prompt: str, messages: list) -> Tuple[Optional[str], Optional[str]]:
    """
    Função principal de alto nível:
    1. Avalia a intenção estruturada de frases no prompt atual.
    2. Se for reedição ("as frases solicitadas", "refaça o post"), busca no histórico.
    3. Se houver gatilho de 'inclua a frase' mas sem aspas claras, usa o guardrail LLM rápido.
    Retorna (headline, subhead).
    """
    intent = extract_phrase_intent(user_prompt)
    if intent["remove_all_text"]:
        return None, None

    if intent["headline"]:
        return intent["headline"], intent["subhead"]

    # 2. Busca retroativa em histórico caso o prompt atual faça referência
    is_referencing_history = any(k in user_prompt.lower() for k in [
        "frases solicitadas", "frase solicitada", "textos solicitados", "texto solicitado",
        "post acima", "post anterior", "como pedido", "como pedi", "mesmas frases", "mesma frase",
        "refaça", "recrie", "melhore", "ajuste"
    ])

    if is_referencing_history and messages:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                if isinstance(content, str):
                    if content.strip() == user_prompt.strip():
                        continue
                    h_found = extract_phrases_from_text(content)
                    if h_found:
                        h = h_found[0]
                        s = h_found[1] if len(h_found) > 1 else None
                        return h, s
                    # Se tinha gatilho no histórico mas não pegou por regex pura
                    has_trig = any(re.search(trig, content, re.IGNORECASE) for trig in TRIGGER_KEYWORDS)
                    if has_trig:
                        llm_found = await extract_phrases_with_llm_guardrail(content)
                        if llm_found:
                            h = llm_found[0]
                            s = llm_found[1] if len(llm_found) > 1 else None
                            return h, s

    # 3. Se há gatilhos explícitos de frases mas falhou em regex pura, aciona o guardrail LLM
    has_phrase_trigger = any(re.search(trig, user_prompt, re.IGNORECASE) for trig in TRIGGER_KEYWORDS) or any(
        k in user_prompt.lower() for k in ["inclua a frase", "inclua as frases", "com a frase", "com as frases", "com o texto", "coloque a frase", "coloque o texto"]
    )

    if has_phrase_trigger:
        llm_found = await extract_phrases_with_llm_guardrail(user_prompt)
        if llm_found:
            h = llm_found[0]
            s = llm_found[1] if len(llm_found) > 1 else None
            return h, s

    return None, None
