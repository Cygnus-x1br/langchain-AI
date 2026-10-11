from typing import List, Tuple, Dict
from config import LITELLM_BASE_URL, LITELLM_API_KEY, OLLAMA_BASE_URL


def _prioritize(category_key, candidates):
    """Aplica a prioridade de modelos definida no Dashboard (fallback seguro para a cadeia original)."""
    try:
        from admin_config_service import apply_model_priority
        return apply_model_priority(category_key, candidates, name_index=1)
    except Exception:
        return candidates


def get_chat_candidates() -> List[Tuple[str, str, Dict[str, str], bool, str]]:
    """
    Retorna a lista ordenada de candidatos (URL, modelo, headers, sanitize_ollama, label)
    para conversas casuais, saudações e interações sociais informais.
    
    Cadeia de Rotação / Fallback:
    1. Combo_Free_Chat (9Router: Groq OSS 120B/20B + Cloudflare Llama 3.3 70B Round Robin)
    2. llama3.2:3b via Ollama Local (Fallback ultrarrápido offline)
    """
    return _prioritize("CASUAL_CHAT", [
        (
            f"{LITELLM_BASE_URL}/chat/completions",
            "Combo_Free_Chat",
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LITELLM_API_KEY}"
            },
            False,
            "9Router (Combo_Free_Chat: Groq 120B/20B / Cloudflare Llama 3.3 70B)"
        ),
        (
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "llama3.2:3b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (llama3.2:3b Fallback)"
        )
    ])
