import os
import re
import copy
from typing import List, Tuple, Dict, Any, Optional
from config import LITELLM_BASE_URL, LITELLM_API_KEY, OLLAMA_BASE_URL
from prompts import CODING_FAST_PROMPT, CODING_EXEC_PROMPT, CODING_PLAN_PROMPT, CODING_SYSTEM_PROMPT

# Configurações para integração com OpenCode / Cline / Continue / IDEs
OPENCODE_ENDPOINT = os.getenv("OPENCODE_ENDPOINT", "").rstrip("/")
OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")

# Padrões para classificação de complexidade de código
COMPLEXITY_PRO_PATTERNS = [
    r"\b(arquitetura|microsservi[çc]os?|microservices|distributed\s+system|sistema\s+distribu[ií]do)\b",
    r"\b(refatora[cç][aã]o\s+(completa|geral|profunda|estrutural)|refatorar\s+(o\s+projeto|o\s+sistema|todo\s+o\s+c[oó]digo))\b",
    r"\b(concorr[eê]ncia|multithreading|asyncio\s+avan[cç]ado|deadlock|race\s+condition|thread\s+pool)\b",
    r"\b(otimiza[cç][aã]o\s+de\s+performance|complexidade\s+de\s+tempo|o\(n\s*\^?\s*2\)|o\(log\s*n\)|algoritmo\s+(complexo|avan[cç]ado|de\s+grafo|dijkstra|a\*|[aá]rvore\s+b|b-tree))\b",
    r"\b(parser\s+ast|compilador|interpretador|bytecode|engenharia\s+reversa|segfault|memory\s+leak|profiling\s+de\s+mem[oó]ria)\b",
    r"\b(projeto\s+completo|aplica[cç][aã]o\s+completa|full-stack|backend\s+e\s+frontend\s+completos)\b",
    r"\b(design\s+pattern|padr[aã]o\s+de\s+projeto|ddd|domain\s+driven\s+design|clean\s+architecture)\b"
]


def detect_coding_complexity(prompt: str, messages: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Avalia a complexidade técnica da demanda de código:
    - Retorna 'PLAN' para arquitetura, concorrência, refatoração estrutural, sistemas distribuídos.
    - Retorna 'EXEC' para implementação regular, scripts e módulos.
    """
    full_text = prompt or ""
    if messages:
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    full_text += " " + content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            full_text += " " + item.get("text", "")
                break

    p_lower = full_text.lower()
    for pattern in COMPLEXITY_PRO_PATTERNS:
        if re.search(pattern, p_lower):
            return "PLAN"

    return "EXEC"


def resolve_coding_tier(
    requested_model: Optional[str] = None,
    user_prompt: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    is_dedicated_endpoint: bool = False
) -> str:
    """
    Resolve o nível de Coding (FAST, EXEC ou PLAN):
    - INTERFACE DO LIBRECHAT (modelo 'agent-orchestrator' ou sem endpoint dedicado):
      Retorna SEMPRE 'FAST', garantindo respostas ultrarrápidas, gratuitas e sem consumo de tokens de raciocínio.
    - ENDPOINTS DEDICADOS (OpenCode, VSCode, Cline, Continue):
      - 'PLAN' (Avançado / Planejamento): se solicitado model com 'plan', 'pro', 'advanced', 'arch'.
      - 'EXEC' (Intermediário / Execução): se solicitado model com 'exec', 'mid', 'impl' ou 'opencode' / 'agent-coding'.
      - 'FAST': se solicitado model com 'fast'.
      - Se o modelo dedicado for genérico ('agent-coding', 'opencode'), detecta entre EXEC e PLAN pelo prompt.
    """
    req = (requested_model or "").lower().strip()

    # 1. Regra Estrita: Chamadas da interface do LibreChat (agent-orchestrator) usam sempre FAST
    if not is_dedicated_endpoint and (not req or req == "agent-orchestrator"):
        return "FAST"

    # 2. Resolução Explícita por Nome de Modelo para IDEs e Endpoints Especializados
    if any(k in req for k in ["plan", "pro", "advanced", "arch", "r1", "o3"]):
        return "PLAN"
    if any(k in req for k in ["exec", "mid", "impl"]):
        return "EXEC"
    if any(k in req for k in ["fast", "quick"]):
        return "FAST"

    # 3. Se for endpoint dedicado com alias geral ('agent-coding', 'opencode'):
    # avalia se o pedido é de arquitetura/planejamento ou execução
    if is_dedicated_endpoint:
        return detect_coding_complexity(user_prompt, messages)

    return "FAST"


def prepare_coding_messages(messages: List[Dict[str, Any]], tier: str = "FAST") -> List[Dict[str, Any]]:
    """
    Injeta o prompt de sistema especializado conforme o nível solicitado:
    - FAST: Código direto, sem enrolação.
    - EXEC: Código robusto, modular, tipado, com testes e tratamento defensivo.
    - PLAN: Arquitetura, planejamento estruturado, trade-offs e implementação profunda.
    """
    msgs = copy.deepcopy(messages)
    system_index = None

    for i, m in enumerate(msgs):
        if m.get("role") == "system":
            system_index = i
            break

    try:
        from admin_config_service import get_prompt_text
        if tier == "PLAN":
            target_prompt = get_prompt_text("CODING_PLAN_PROMPT") or CODING_PLAN_PROMPT
        elif tier == "EXEC":
            target_prompt = get_prompt_text("CODING_EXEC_PROMPT") or CODING_EXEC_PROMPT
        else:
            target_prompt = get_prompt_text("CODING_FAST_PROMPT") or CODING_FAST_PROMPT
    except Exception:
        target_prompt = CODING_FAST_PROMPT
        if tier == "PLAN":
            target_prompt = CODING_PLAN_PROMPT
        elif tier == "EXEC":
            target_prompt = CODING_EXEC_PROMPT

    mcp_notice = (
        "\n[MCP FILESYSTEM DISPONÍVEL]: O ambiente possui um diretório de trabalho seguro montado em '/app/workspace'. "
        "Se o usuário pedir para gerar, salvar ou organizar arquivos no workspace, forneça código completo e indique a localização no workspace."
    )

    if system_index is None:
        msgs.insert(0, {"role": "system", "content": target_prompt + mcp_notice})
    else:
        curr_sys = msgs[system_index].get("content", "")
        if "CÓDIGO COMPLETO" not in curr_sys and "DIRETRIZES" not in curr_sys:
            enhanced_sys = f"{curr_sys}\n\n[DIRETRIZES DE ENGENHARIA DE SOFTWARE - NÍVEL {tier}]\n{target_prompt}{mcp_notice}"
            msgs[system_index]["content"] = enhanced_sys

    return msgs


def get_coding_candidates(tier: str = "FAST") -> List[Tuple[str, str, Dict[str, str], bool, str]]:
    """
    Retorna a lista ordenada de candidatos (URL, modelo, headers, sanitize_ollama, label)
    conforme o nível de código selecionado.

    1. NÍVEL FAST (Interface LibreChat / Scripts Simples / Edições Rápidas):
       1º: Combo_Free_Coding (9Router: Qwen 2.5 Coder 32B Cloudflare + Groq OSS 120B)
       2º: Coding_DeepSeek_V3 (LiteLLM OpenRouter: DeepSeek V3 671B - ~$0.14/1M, ~1.5s)
       3º: Coding_Fast_Fallback (LiteLLM OpenAI: GPT-4o-mini)
       4º: qwen2.5-coder:7b (Ollama Local 100% offline)

    2. NÍVEL EXEC (Intermediário / Execução para OpenCode/IDEs):
       1º: Coding_DeepSeek_V3 (DeepSeek V3 671B - Líder de Execução direta sem reasoning overhead)
       2º: Combo_Free_Coding (Qwen 2.5 Coder 32B)
       3º: Coding_Fast_Fallback (GPT-4o-mini)
       4º: qwen2.5-coder:7b (Ollama Local)

    3. NÍVEL PLAN (Avançado / Planejamento Arquitetural com Raciocínio Profundo):
       1º: Reasoning_DeepSeek_R1 (LiteLLM OpenRouter: DeepSeek R1 - Raciocínio Profundo / CoT)
       2º: Reasoning_O3_Mini (LiteLLM OpenAI: o3-mini)
       3º: Coding_DeepSeek_V3 (DeepSeek V3)
       4º: Combo_Free_Coding (Qwen 2.5 Coder 32B)
       5º: qwen2.5-coder:7b (Ollama Local)
    """
    candidates = []

    # Prioridade opcional caso exista servidor OpenCode dedicado via env
    if OPENCODE_ENDPOINT:
        headers = {"Content-Type": "application/json"}
        if OPENCODE_API_KEY:
            headers["Authorization"] = f"Bearer {OPENCODE_API_KEY}"
        candidates.append((
            f"{OPENCODE_ENDPOINT}/chat/completions",
            "opencode-default",
            headers,
            False,
            "OpenCode Dedicated Server"
        ))

    litellm_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LITELLM_API_KEY}"
    }

    tier = (tier or "FAST").upper()

    if tier == "PLAN":
        # Nível Avançado / Planejamento & Arquitetura
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Reasoning_DeepSeek_R1",
            litellm_headers,
            False,
            "LiteLLM (DeepSeek R1 - Planejamento & Raciocínio Profundo)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Reasoning_O3_Mini",
            litellm_headers,
            False,
            "LiteLLM (OpenAI o3-mini - Raciocínio de Código Pro)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Coding_DeepSeek_V3",
            litellm_headers,
            False,
            "LiteLLM (DeepSeek V3 - Fallback de Execução)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Combo_Free_Coding",
            litellm_headers,
            False,
            "9Router (Combo_Free_Coding: Qwen 2.5 Coder 32B)"
        ))
        candidates.append((
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "qwen2.5-coder:7b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (qwen2.5-coder:7b Fallback Offline)"
        ))

    elif tier == "EXEC":
        # Nível Intermediário / Execução & Implementação
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Coding_DeepSeek_V3",
            litellm_headers,
            False,
            "LiteLLM (DeepSeek V3 - Execução & Implementação de Código)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Combo_Free_Coding",
            litellm_headers,
            False,
            "9Router (Combo_Free_Coding: Qwen 2.5 Coder 32B)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Coding_Fast_Fallback",
            litellm_headers,
            False,
            "LiteLLM (GPT-4o-mini - Contingência Nuvem)"
        ))
        candidates.append((
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "qwen2.5-coder:7b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (qwen2.5-coder:7b Fallback Offline)"
        ))

    else:
        # Nível FAST (Padrão para LibreChat e Scripts Rápidos)
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Combo_Free_Coding",
            litellm_headers,
            False,
            "9Router (Combo_Free_Coding: Qwen 2.5 Coder 32B / Groq OSS 120B)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Coding_DeepSeek_V3",
            litellm_headers,
            False,
            "LiteLLM (DeepSeek V3 - Nuvem Econômica Rápida)"
        ))
        candidates.append((
            f"{LITELLM_BASE_URL}/chat/completions",
            "Coding_Fast_Fallback",
            litellm_headers,
            False,
            "LiteLLM (GPT-4o-mini - Nuvem Econômica Contingência)"
        ))
        candidates.append((
            f"{OLLAMA_BASE_URL}/v1/chat/completions",
            "qwen2.5-coder:7b",
            {"Content-Type": "application/json"},
            True,
            "Ollama Local (qwen2.5-coder:7b Fallback Offline)"
        ))

    # Prioridade definida no Dashboard (o servidor OpenCode dedicado, se existir, permanece à frente)
    try:
        from admin_config_service import apply_model_priority
        candidates = apply_model_priority(f"CODING_{tier}", candidates, name_index=1)
    except Exception:
        pass

    return candidates


def is_code_response_complete(response_text: str) -> bool:
    """
    Verifica se a resposta de código gerada está completa ou foi truncada.
    Utilizado para evitar que o cache armazene respostas corrompidas ou cortadas.
    """
    if not response_text or len(response_text.strip()) < 80:
        return False

    text = response_text.strip()

    # Se há blocos de código Markdown abertos sem fechamento (número ímpar de ```)
    code_block_count = text.count("```")
    if code_block_count > 0 and code_block_count % 2 != 0:
        return False

    # Verifica se termina abruptamente com operadores ou instruções incompletas
    suspicious_endings = [
        "print(", "def ", "class ", "return (", "import ",
        "from ", "elif ", "else:", "try:", "except ", "with open(",
        " = ", " + ", " -> ", "...", "todo:", "TODO:"
    ]
    last_line = text.split("\n")[-1].strip()
    for s in suspicious_endings:
        if last_line.endswith(s) or last_line == s.strip():
            return False

    return True
