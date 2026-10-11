from typing import List, Tuple, Dict
from config import LITELLM_BASE_URL, LITELLM_API_KEY, OLLAMA_BASE_URL


def _prioritize(category_key, candidates):
    """Aplica a prioridade de modelos definida no Dashboard (fallback seguro para a cadeia original)."""
    try:
        from admin_config_service import apply_model_priority
        return apply_model_priority(category_key, candidates, name_index=1)
    except Exception:
        return candidates


def get_factual_candidates() -> List[Tuple[str, str, Dict[str, str], bool, str]]:
    """
    Retorna a lista ordenada de candidatos (URL, modelo, headers, sanitize_ollama, label)
    para conhecimento geral, fatos, listas, receitas, ordenação de dados e curiosidades.
    
    Cadeia de Fallback:
    1. ComboFree_Smart (LiteLLM / 9Router: Modelos avançados equilibrados)
    2. llama3.2:3b via Ollama Local (Fallback offline)
    """
    return _prioritize("FACTUAL_GENERAL", [
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
            "llama3.2:3b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (llama3.2:3b Fallback)"
        )
    ])
