"""
MCP Brave Search Service - Busca Web Independente e Resiliente
===============================================================
Implementação da especificação Model Context Protocol (MCP) Brave Search:
Permite aos agentes e ao usuário realizar buscas na web através do índice
independente e descentralizado da Brave Search API.

Mecanismos de Resiliência e Segurança:
1. Conexão assíncrona HTTP via httpx com timeout e tratamento defensivo de erros.
2. Suporte a parâmetros de contagem (count), frescor (freshness) e idiomas.
3. Fallback gracioso: Se a chave BRAVE_API_KEY não estiver definida, instrui a configuração
   e pode redirecionar para o SearXNG local.
4. Exposição dupla: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
5. Integração com sintetizador de IA (ComboFree_Smart) para respostas completas com links de fonte.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple

import httpx
from langchain_core.tools import tool

# Configuração da API Brave Search
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "").strip()
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


# =========================================================================
# 1. FUNÇÕES DE CONSULTA DIRETA À API BRAVE SEARCH
# =========================================================================

async def async_brave_search(
    query: str,
    count: int = 5,
    freshness: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executa consulta à API Web da Brave Search.
    Retorna dicionário contendo lista de resultados e metadados.
    """
    key = (api_key or BRAVE_API_KEY).strip()
    if not key:
        return {
            "success": False,
            "error": "BRAVE_API_KEY não está configurada no ambiente (.env). Adicione sua chave para habilitar o MCP Brave Search.",
            "results": []
        }

    clean_q = (query or "").strip()
    if not clean_q:
        return {"success": False, "error": "A consulta de busca não pode ser vazia.", "results": []}

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": key,
        "User-Agent": "LangChain-Orchestrator-MCP-Brave/1.0"
    }

    params: Dict[str, Any] = {
        "q": clean_q,
        "count": min(max(1, int(count)), 20),
        "text_decorations": False,
        "spellcheck": True
    }

    if freshness and freshness.strip() in ["pd", "pw", "pm", "py"]:
        params["freshness"] = freshness.strip()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(BRAVE_SEARCH_URL, headers=headers, params=params)

            if resp.status_code == 200:
                data = resp.json()
                web_results = data.get("web", {}).get("results", [])
                formatted = []
                for item in web_results:
                    formatted.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("description", ""),
                        "extra_snippets": item.get("extra_snippets", []),
                        "age": item.get("age", "")
                    })
                return {
                    "success": True,
                    "query": clean_q,
                    "count": len(formatted),
                    "results": formatted
                }
            elif resp.status_code == 401 or resp.status_code == 403:
                return {
                    "success": False,
                    "error": f"Autenticação recusada pela Brave Search API (Status {resp.status_code}). Verifique a validade de BRAVE_API_KEY.",
                    "results": []
                }
            elif resp.status_code == 429:
                return {
                    "success": False,
                    "error": "Limite de requisições excedido na Brave Search API (HTTP 429 Rate Limit).",
                    "results": []
                }
            else:
                return {
                    "success": False,
                    "error": f"Erro retornado pela Brave Search API (HTTP {resp.status_code}): {resp.text[:300]}",
                    "results": []
                }
    except httpx.TimeoutException:
        return {"success": False, "error": "Tempo limite excedido (timeout 10s) ao consultar Brave Search API.", "results": []}
    except Exception as e:
        return {"success": False, "error": f"Falha na conexão com Brave Search API: {e}", "results": []}


def format_brave_results_markdown(data: Dict[str, Any]) -> str:
    """Formata os resultados da busca em Markdown legível e atraente."""
    if not data.get("success"):
        return f"⚠️ **Erro no MCP Brave Search:** {data.get('error', 'Falha desconhecida')}"

    results = data.get("results", [])
    if not results:
        return f"🔍 **Brave Search:** Nenhum resultado encontrado para a consulta `{data.get('query')}`."

    blocks = [f"### 🦁 MCP Brave Search: Resultados para `{data.get('query')}`\n"]
    for idx, r in enumerate(results, 1):
        age_badge = f" *({r.get('age')})*" if r.get('age') else ""
        desc = r.get("description", "").strip()
        blocks.append(f"**{idx}. [{r.get('title')}]({r.get('url')})**{age_badge}\n> {desc}\n")

    return "\n".join(blocks)


# =========================================================================
# 2. EXPOSIÇÃO COMO LANGCHAIN TOOLS (@tool)
# =========================================================================

@tool
def mcp_brave_web_search(query: str, count: int = 5, freshness: Optional[str] = None) -> str:
    """
    Realiza busca na web em tempo real utilizando a Brave Search API independente.
    Args:
        query: Termos de pesquisa na web.
        count: Quantidade de resultados desejados (padrão: 5, máximo: 20).
        freshness: Filtro de data opcional ('pd' para 24h, 'pw' para 7 dias, 'pm' para 1 mês, 'py' para 1 ano).
    """
    import asyncio
    try:
        # Se estiver em loop assíncrono ativo
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                res = pool.submit(asyncio.run, async_brave_search(query=query, count=count, freshness=freshness)).result()
        else:
            res = loop.run_until_complete(async_brave_search(query=query, count=count, freshness=freshness))
    except Exception:
        res = asyncio.run(async_brave_search(query=query, count=count, freshness=freshness))

    return format_brave_results_markdown(res)


def get_mcp_brave_langchain_tools() -> List[Any]:
    """Retorna ferramentas MCP Brave Search prontas para LangChain."""
    return [mcp_brave_web_search]


# =========================================================================
# 3. EXPOSIÇÃO COMO SCHEMAS OPENAI TOOL CALLING
# =========================================================================

def get_mcp_brave_openai_tools() -> List[Dict[str, Any]]:
    """Retorna os schemas de chamada de função no padrão OpenAI Tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp_brave_web_search",
                "description": "Busca informações atualizadas na web utilizando a API independente Brave Search.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Termos de busca na internet."
                        },
                        "count": {
                            "type": "integer",
                            "description": "Número de links para retornar (padrão 5, máximo 20)."
                        },
                        "freshness": {
                            "type": "string",
                            "enum": ["pd", "pw", "pm", "py"],
                            "description": "Filtro de recência: 'pd' (últimas 24h), 'pw' (última semana), 'pm' (último mês), 'py' (último ano)."
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]


# =========================================================================
# 4. FLUXO EXECUTOR AUTÔNOMO (execute_mcp_brave_search_flow)
# =========================================================================

async def execute_mcp_brave_search_flow(
    user_prompt: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executa a busca via Brave Search API e sintetiza a resposta final com IA e citações.
    Se a chave não estiver configurada, instrui com transparência ou recorre ao SearXNG.
    """
    t0 = time.time()
    tools_called = []

    # Extração limpa dos termos de busca
    clean_query = user_prompt
    prefix_patterns = [
        r"(?i)^(busque\s+no\s+brave|pesquise\s+no\s+brave|brave\s+search|mcp\s+brave|brave)\s*:\s*",
        r"(?i)^(busque\s+no\s+brave|pesquise\s+no\s+brave|brave\s+search|mcp\s+brave|brave)\s+"
    ]
    for pat in prefix_patterns:
        clean_query = re.sub(pat, "", clean_query).strip()

    search_data = await async_brave_search(query=clean_query, count=6)
    dur = round(time.time() - t0, 3)

    tools_called.append({
        "tool": "mcp_brave_web_search",
        "query": clean_query,
        "status": "success" if search_data.get("success") else "error",
        "duration_seconds": dur
    })

    # Caso a chave não esteja configurada ou tenha retornado erro
    if not search_data.get("success"):
        err_msg = search_data.get("error", "")
        # Tenta fallback gracioso para SearXNG se disponível
        from config import SEARXNG_URL
        if SEARXNG_URL and ("BRAVE_API_KEY" in err_msg or "Autenticação" in err_msg):
            print(f"[MCP Brave Fallback] Redirecionando para SearXNG local devido a: {err_msg}")
            from agent_service import async_searx_search
            fallback_text = await async_searx_search(clean_query, user_id=user_id)
            tools_called.append({
                "tool": "searxng_fallback",
                "query": clean_query,
                "status": "success" if fallback_text else "empty",
                "duration_seconds": round(time.time() - t0, 3)
            })
            if fallback_text:
                return (
                    f"💡 *[Nota: BRAVE_API_KEY não configurada. Resposta obtida via SearXNG local]*\n\n"
                    f"{fallback_text[:2500]}",
                    tools_called
                )

        return (
            f"⚠️ **MCP Brave Search Indisponível:**\n\n"
            f"> {err_msg}\n\n"
            f"Para ativar, insira sua chave gratuita em `BRAVE_API_KEY` no arquivo `.env` do servidor.",
            tools_called
        )

    results = search_data.get("results", [])
    if not results:
        return f"🔍 **Brave Search:** Nenhum resultado encontrado para `{clean_query}`.", tools_called

    # Monta o contexto para o sintetizador
    context_lines = []
    for idx, r in enumerate(results, 1):
        context_lines.append(
            f"[{idx}] TÍTULO: {r.get('title')}\n"
            f"URL: {r.get('url')}\n"
            f"RESUMO: {r.get('description')}\n"
        )
    raw_context = "\n".join(context_lines)

    # Síntese Inteligente via LLM
    from config import LITELLM_BASE_URL, LITELLM_API_KEY
    from langchain_openai import ChatOpenAI

    system_prompt = f"""Você é o Assistente Especialista MCP Brave Search.
Sua missão é responder à pergunta do usuário baseando-se RIGOROSAMENTE nas fontes e dados da Web retornados pela Brave Search API abaixo.

DIRETRIZES:
1. Responda de forma assertiva, clara e atualizada em português.
2. Cite explicitamente as fontes com links Markdown no texto [Título](URL).
3. Nunca invente dados que não constem nos resultados obtidos.

RESULTADOS DA PESQUISA BRAVE SEARCH:
{raw_context}
"""

    try:
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model="ComboFree_Smart",
            temperature=0.3,
            max_tokens=2500,
            timeout=30.0
        )
        resp = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        synthesis = resp.content if hasattr(resp, "content") else str(resp)
        tools_called.append({
            "tool": "mcp_brave_synthesizer",
            "model": "ComboFree_Smart",
            "status": "success",
            "duration_seconds": round(time.time() - t0, 3)
        })
        return synthesis, tools_called

    except Exception as llm_err:
        print(f"[Aviso Síntese MCP Brave]: {llm_err}")
        return format_brave_results_markdown(search_data), tools_called
