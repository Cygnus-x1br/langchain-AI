from typing import List, Tuple, Dict
from config import LITELLM_BASE_URL, LITELLM_API_KEY, OLLAMA_BASE_URL


def _prioritize(category_key, candidates):
    """Aplica a prioridade de modelos definida no Dashboard (fallback seguro para a cadeia original)."""
    try:
        from admin_config_service import apply_model_priority
        return apply_model_priority(category_key, candidates, name_index=1)
    except Exception:
        return candidates


_reasoning_provider_turn = 0

def select_reasoning_candidate_chain(user_prompt: str) -> List[Tuple[str, str]]:
    """
    Seleciona e rotaciona modelos para HEAVY_REASONING conforme complexidade e custo.
    Retorna lista de tuplas: (model_name_litellm, provider_label)
    
    Tiers de Custo & Capacidade:
    - Tier 1 (Econômico / CoT Puro): DeepSeek R1 ($0.70/$2.50), OpenAI o3-mini ($1.10/$4.40), Kimi K2 Thinking ($0.60/$2.50)
    - Tier 2 (Fronteira Avançada & Contexto Longo): Gemini 3.1 Pro ($2.00/$12.00), ChatGPT 6.1 Sol ($2.00/$10.00), Kimi K3 ($0.67/$14.00)
    - Tier 3 (Fronteira Máxima & Fallback Premium): Claude Sonnet 5.5 ($2.00-$3.00/$10.00-$15.00)
    """
    global _reasoning_provider_turn
    _reasoning_provider_turn += 1
    p_lower = user_prompt.lower()
    
    # 1. Indicadores de Matemática, Lógica Formal, Algoritmos e Xadrez (STEM / CoT)
    stem_keywords = [
        "cálculo", "calculo", "derivada", "integral", "matriz", "matrizes", "álgebra", "algebra",
        "teorema", "equação", "equacao", "física", "fisica", "xadrez", "chess", "enigma",
        "probabilidade", "estatística", "estatistica", "geometria", "algoritmo", "demonstre"
    ]
    is_stem = any(k in p_lower for k in stem_keywords)

    # 2. Indicadores de Longo Contexto, Análise Abrangente de Sistemas e Teses
    broad_keywords = [
        "análise de sistema", "analise de sistema", "arquitetura", "artigo", "tese",
        "dissertação", "dissertacao", "análise comparativa", "analise comparativa", "panorama",
        "abrangente", "revisão bibliográfica", "revisao bibliografica", "livro", "monografia"
    ]
    is_broad = any(k in p_lower for k in broad_keywords)

    deepseek_r1 = ("Reasoning_DeepSeek_R1", "DeepSeek R1 via OpenRouter ($0.70/$2.50)")
    o3_mini = ("Reasoning_O3_Mini", "OpenAI o3-mini via Direct API ($1.10/$4.40)")
    kimi_thinking = ("Reasoning_Kimi_Thinking", "Moonshot Kimi K2 Thinking via OpenRouter ($0.60/$2.50)")
    kimi_k3 = ("Reasoning_Kimi_K3", "Moonshot Kimi K3 via OpenRouter ($0.67/$14.00)")
    gemini_pro = ("Reasoning_Gemini_3_Pro", "Google Gemini 3.1 Pro via OpenRouter ($2.00/$12.00)")
    chatgpt_6 = ("Reasoning_ChatGPT_6_1", "OpenAI ChatGPT 6.1 Sol via OpenRouter ($2.00/$10.00)")
    claude_sonnet = ("Reasoning_Claude_Sonnet", "Anthropic Claude Sonnet 5.5 via OpenRouter ($2.00/$10.00)")

    if is_stem:
        # Foco em raciocínio analítico/matemático: alterna entre R1 e o3-mini na liderança
        if _reasoning_provider_turn % 2 == 1:
            return [deepseek_r1, o3_mini, kimi_thinking, gemini_pro, chatgpt_6, claude_sonnet]
        else:
            return [o3_mini, deepseek_r1, kimi_thinking, gemini_pro, chatgpt_6, claude_sonnet]

    elif is_broad:
        # Foco em síntese de alto volume e sistemas: alterna entre Gemini 3.1 Pro, ChatGPT 6.1 e Kimi K3
        turn = _reasoning_provider_turn % 3
        if turn == 1:
            return [gemini_pro, chatgpt_6, kimi_k3, o3_mini, deepseek_r1, claude_sonnet]
        elif turn == 2:
            return [chatgpt_6, gemini_pro, kimi_k3, o3_mini, deepseek_r1, claude_sonnet]
        else:
            return [kimi_k3, gemini_pro, chatgpt_6, o3_mini, deepseek_r1, claude_sonnet]

    else:
        # Raciocínio Geral / Heavy Reasoning Padrão: Rotação balanceada entre opções econômicas
        turn = _reasoning_provider_turn % 4
        if turn == 1:
            return [deepseek_r1, o3_mini, kimi_thinking, gemini_pro, chatgpt_6, claude_sonnet]
        elif turn == 2:
            return [o3_mini, kimi_thinking, deepseek_r1, gemini_pro, chatgpt_6, claude_sonnet]
        elif turn == 3:
            return [kimi_thinking, deepseek_r1, o3_mini, gemini_pro, chatgpt_6, claude_sonnet]
        else:
            return [gemini_pro, o3_mini, deepseek_r1, chatgpt_6, kimi_thinking, claude_sonnet]

def get_heavy_reasoning_candidates(user_prompt: str) -> List[Tuple[str, str, Dict[str, str], bool, str]]:
    """Retorna a cadeia de candidatos completa para requisições de raciocínio pesado."""
    reasoning_list = select_reasoning_candidate_chain(user_prompt)
    return [
        (
            f"{LITELLM_BASE_URL}/chat/completions",
            m_name,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LITELLM_API_KEY}"
            },
            False,
            m_label
        )
        for m_name, m_label in reasoning_list
    ]

def get_local_thinking_candidates() -> List[Tuple[str, str, Dict[str, str], bool, str]]:
    """Retorna candidatos para raciocínio local/rápido (LOCAL_THINKING)."""
    return _prioritize("LOCAL_THINKING", [
        (
            f"{LITELLM_BASE_URL}/chat/completions",
            "ComboFree_Smart",
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LITELLM_API_KEY}"
            },
            False,
            "LiteLLM / 9Router (ComboFree_Smart)"
        ),
        (
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "qwen3.5:9b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (qwen3.5:9b Fallback)"
        )
    ])
