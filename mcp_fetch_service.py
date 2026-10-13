"""
MCP Fetch Service - Web Fetch Seguro e Extração de Conteúdo em Markdown
========================================================================
Implementação da especificação Model Context Protocol (MCP) Fetch:
Permite aos agentes e ao usuário acessar URLs públicas, extrair páginas da web,
converter HTML e documentos PDF em Markdown limpo e navegar em SPAs dinâmicas.

Mecanismos de Segurança (Zero-Trust SSRF Protection):
1. Validação estrita de esquema (apenas http e https).
2. Bloqueio automático de IP privado/loopback (127.0.0.1, 10.0.0.0/8, 172.16.0.0/12,
   192.168.0.0/16, 169.254.169.254, localhost) contra Server-Side Request Forgery (SSRF).
3. Limite de tamanho de conteúdo (max_length) para evitar estouro de memória (OOM).
4. Suporte a extração de HTML estático, SPAs (via Crawl4AI) e arquivos PDF (via pypdf).
5. Exposição dupla: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
"""

import os
import re
import io
import time
import socket
import ipaddress
import urllib.parse
from typing import Dict, Any, List, Optional, Tuple

import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# Tenta carregar markdownify para conversão rica de HTML -> Markdown
try:
    from markdownify import markdownify as md_convert
except ImportError:
    md_convert = None

# Tenta carregar pypdf para documentos PDF
try:
    import pypdf
except ImportError:
    pypdf = None

# Tenta carregar Crawl4AI para renderização de páginas dinâmicas (SPAs)
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None

# Configurações do serviço
DEFAULT_MAX_LENGTH = int(os.getenv("MCP_FETCH_MAX_LENGTH", "50000"))
ALLOW_INTERNAL_SSRF = os.getenv("ALLOW_INTERNAL_SSRF", "false").lower() in ("true", "1", "yes")
USER_AGENT = os.getenv(
    "MCP_FETCH_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 (MCP-Fetch/1.0)"
)


# =========================================================================
# 1. PROTEÇÃO CONTRA SSRF (SERVER-SIDE REQUEST FORGERY)
# =========================================================================

def is_safe_external_url(url: str) -> Tuple[bool, str]:
    """
    Valida a URL contra ataques SSRF.
    Impede que o agente requisite metadados de nuvem (169.254.169.254),
    loopback local (127.0.0.1) ou redes internas privadas sem permissão explícita.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Esquema de URL inválido: '{parsed.scheme}'. Apenas 'http' e 'https' são permitidos."

        hostname = parsed.hostname
        if not hostname:
            return False, "URL não contém um hostname válido."

        # Se SSRF interno for permitido por configuração explícita de ambiente
        if ALLOW_INTERNAL_SSRF:
            return True, "OK"

        # Resolve o IP associado ao hostname
        try:
            ip_str = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip_str)
        except Exception as dns_err:
            return False, f"Falha na resolução de DNS para '{hostname}': {dns_err}"

        # Verifica bloqueios de segurança
        if ip_obj.is_loopback:
            return False, f"Acesso bloqueado por segurança (Loopback/Localhost detectado: {ip_str})."
        if ip_obj.is_private:
            return False, f"Acesso bloqueado por segurança (Rede interna privada detectada: {ip_str})."
        if ip_obj.is_link_local:
            return False, f"Acesso bloqueado por segurança (Endereço Link-Local/Cloud Metadata detectado: {ip_str})."
        if ip_obj.is_reserved or ip_obj.is_multicast:
            return False, f"Acesso bloqueado por segurança (Endereço reservado detectado: {ip_str})."

        return True, "OK"
    except Exception as e:
        return False, f"Erro ao validar segurança da URL: {e}"


# =========================================================================
# 2. CONVERSÃO E LIMPEZA DE HTML PARA MARKDOWN
# =========================================================================

def html_to_clean_markdown(html_content: str) -> str:
    """
    Converte HTML em Markdown limpo e estruturado.
    Remove tags de navegação, scripts, banners, estilos e elementos visuais supérfluos.
    """
    if not html_content or not html_content.strip():
        return ""

    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove elementos não-textuais e ruídos
        for tag in soup([
            "script", "style", "nav", "footer", "header", "noscript",
            "svg", "form", "iframe", "button", "input", "aside",
            "dialog", "template", "select", "textarea"
        ]):
            tag.decompose()

        cleaned_html = str(soup)

        # Se markdownify estiver disponível, gera Markdown rico
        if md_convert is not None:
            md_text = md_convert(
                cleaned_html,
                heading_style="ATX",
                strip=["img", "a"] if False else [],
                autolinks=True,
                default_title=True
            )
        else:
            # Fallback direto via BeautifulSoup text
            md_text = soup.get_text(separator="\n\n")

        # Normaliza espaços em branco e quebras de linha excessivas
        md_text = re.sub(r"\n{3,}", "\n\n", md_text)
        lines = [l.strip() for l in md_text.splitlines()]
        return "\n".join(lines).strip()
    except Exception as e:
        return f"[Erro ao converter HTML para Markdown: {e}]"


def extract_metadata_from_soup(html_content: str) -> Dict[str, str]:
    """Extrai título e descrição da página a partir das meta tags HTML."""
    meta = {"title": "", "description": ""}
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            meta["title"] = title_tag.string.strip()

        desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if desc_tag and desc_tag.get("content"):
            meta["description"] = desc_tag["content"].strip()
    except Exception:
        pass
    return meta


# =========================================================================
# 3. CORE FETCH SERVICE
# =========================================================================

async def core_fetch(
    url: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    start_index: int = 0,
    raw: bool = False
) -> Dict[str, Any]:
    """
    Executa a requisição HTTP/HTTPS e retorna o conteúdo convertido em Markdown estruturado.
    Implementa fallback automático para SPAs dinâmicas via Crawl4AI e extração de PDFs.
    """
    t0 = time.time()
    url = url.strip()

    # Validação de segurança SSRF
    is_safe, reason = is_safe_external_url(url)
    if not is_safe:
        raise PermissionError(f"Acesso bloqueado por segurança (MCP Fetch Guardrail): {reason}")

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,text/plain;q=0.8,*/*;q=0.5",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Upgrade-Insecure-Requests": "1"
    }

    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            status_code = resp.status_code
            final_url = str(resp.url)
            content_type = resp.headers.get("content-type", "").lower()

            if status_code >= 400:
                return {
                    "url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "title": f"HTTP Error {status_code}",
                    "description": "",
                    "content": f"⚠️ O servidor remoto retornou o código HTTP {status_code}: {resp.text[:500]}",
                    "content_type": content_type,
                    "length": 0,
                    "duration_seconds": round(time.time() - t0, 3),
                    "method": "http_error"
                }

            # 1. Tratamento para Arquivo PDF
            if "application/pdf" in content_type or resp.content.startswith(b"%PDF"):
                pdf_text = ""
                if pypdf is not None:
                    try:
                        reader = pypdf.PdfReader(io.BytesIO(resp.content))
                        pages_out = []
                        for i, page in enumerate(reader.pages, 1):
                            txt = page.extract_text() or ""
                            if txt.strip():
                                pages_out.append(f"### Página {i}\n{txt.strip()}")
                        pdf_text = "\n\n".join(pages_out)
                    except Exception as pe:
                        pdf_text = f"[Erro ao extrair texto do documento PDF: {pe}]"
                else:
                    pdf_text = "[Biblioteca pypdf não disponível para extração de PDF]"

                final_text = pdf_text[start_index:start_index + max_length]
                return {
                    "url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "title": os.path.basename(urllib.parse.urlparse(url).path) or "Documento PDF",
                    "description": "Arquivo PDF extraído",
                    "content": final_text,
                    "content_type": "application/pdf",
                    "length": len(final_text),
                    "duration_seconds": round(time.time() - t0, 3),
                    "method": "pdf_extractor"
                }

            # 2. Tratamento para Conteúdo em Texto Puro / JSON / Markdown
            if any(t in content_type for t in ["text/plain", "application/json", "text/markdown"]):
                raw_text = resp.text
                final_text = raw_text[start_index:start_index + max_length]
                return {
                    "url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "title": os.path.basename(urllib.parse.urlparse(url).path) or "Texto",
                    "description": "",
                    "content": final_text,
                    "content_type": content_type,
                    "length": len(final_text),
                    "duration_seconds": round(time.time() - t0, 3),
                    "method": "plain_text"
                }

            # 3. Tratamento para Conteúdo HTML
            raw_html = resp.text
            meta = extract_metadata_from_soup(raw_html)

            # Se o usuário pediu conteúdo bruto (raw HTML)
            if raw:
                final_text = raw_html[start_index:start_index + max_length]
                return {
                    "url": url,
                    "final_url": final_url,
                    "status_code": status_code,
                    "title": meta.get("title", ""),
                    "description": meta.get("description", ""),
                    "content": final_text,
                    "content_type": "text/html",
                    "length": len(final_text),
                    "duration_seconds": round(time.time() - t0, 3),
                    "method": "raw_html"
                }

            # Converte HTML para Markdown limpo
            md_content = html_to_clean_markdown(raw_html)

            # 4. Fallback Dinâmico: se o conteúdo extraído for muito curto (< 350 chars),
            # indica provável Single Page Application (SPA em React/Vue) sem renderização no servidor.
            if len(md_content.strip()) < 350 and AsyncWebCrawler is not None:
                try:
                    print(f"[MCP Fetch] Conteúdo curto/SPA detectado ({len(md_content)} chars). Acionando renderização dinâmica Crawl4AI...")
                    async with AsyncWebCrawler(verbose=False) as crawler:
                        crawl_res = await crawler.arun(url=url, bypass_cache=False)
                        if crawl_res and crawl_res.success and crawl_res.markdown and len(crawl_res.markdown.strip()) > len(md_content):
                            md_content = crawl_res.markdown.strip()
                            if crawl_res.metadata and crawl_res.metadata.get("title"):
                                meta["title"] = crawl_res.metadata.get("title")
                except Exception as crawl_err:
                    print(f"[Aviso MCP Fetch Crawl4AI Fallback]: {crawl_err}")

            # Paginação de segurança
            final_content = md_content[start_index:start_index + max_length]

            return {
                "url": url,
                "final_url": final_url,
                "status_code": status_code,
                "title": meta.get("title", ""),
                "description": meta.get("description", ""),
                "content": final_content,
                "content_type": "text/markdown",
                "length": len(final_content),
                "duration_seconds": round(time.time() - t0, 3),
                "method": "html_markdown"
            }

    except Exception as req_err:
        return {
            "url": url,
            "final_url": url,
            "status_code": 0,
            "title": "Erro de Conexão",
            "description": "",
            "content": f"⚠️ Não foi possível carregar a URL `{url}`: {req_err}",
            "content_type": "text/plain",
            "length": 0,
            "duration_seconds": round(time.time() - t0, 3),
            "method": "exception"
        }


async def core_fetch_headers(url: str) -> Dict[str, Any]:
    """Inspeciona apenas os cabeçalhos HTTP (status, content-type, encoding, tamanho) sem carregar o body."""
    t0 = time.time()
    is_safe, reason = is_safe_external_url(url)
    if not is_safe:
        raise PermissionError(f"Acesso bloqueado por segurança (MCP Fetch Guardrail): {reason}")

    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
        resp = await client.head(url)
        return {
            "url": url,
            "final_url": str(resp.url),
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "duration_seconds": round(time.time() - t0, 3)
        }


# =========================================================================
# 4. FERRAMENTAS LANGCHAIN (@tool) & ESQUEMAS OPENAI (MCP SPEC)
# =========================================================================

@tool
def mcp_fetch(
    url: str,
    max_length: int = 50000,
    start_index: int = 0,
    raw: bool = False
) -> str:
    """
    Acessa uma página web ou recurso via URL, realiza scraping seguro e retorna o conteúdo convertido em Markdown estruturado.
    Parâmetros:
    - url: Endereço completo (http:// ou https://) da página a ser lida.
    - max_length: Limite máximo de caracteres a retornar (padrão: 50.000).
    - start_index: Posição inicial para paginação em textos longos.
    - raw: Se True, retorna o HTML original sem converter para Markdown.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    res = loop.run_until_complete(core_fetch(url, max_length, start_index, raw))
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_fetch_headers(url: str) -> str:
    """Inspeciona os cabeçalhos HTTP e metadados de uma URL sem baixar o corpo completo."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    res = loop.run_until_complete(core_fetch_headers(url))
    return json.dumps(res, ensure_ascii=False, indent=2)


def get_mcp_fetch_langchain_tools() -> List[Any]:
    """Retorna a lista de ferramentas LangChain prontas para agentes."""
    return [mcp_fetch, mcp_fetch_headers]


def get_mcp_fetch_openai_tools() -> List[Dict[str, Any]]:
    """Retorna os esquemas OpenAI Tool Calling da especificação MCP Fetch."""
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch",
                "description": "Recupera o conteúdo de uma página web ou documento via URL e o converte em Markdown estruturado limpo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "A URL completa (http ou https) da página a ser lida."
                        },
                        "max_length": {
                            "type": "integer",
                            "description": "Número máximo de caracteres a retornar (default: 50000)."
                        },
                        "start_index": {
                            "type": "integer",
                            "description": "Índice inicial para paginação em conteúdos muito extensos."
                        },
                        "raw": {
                            "type": "boolean",
                            "description": "Se verdadeiro, retorna o HTML bruto sem converter para Markdown."
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_headers",
                "description": "Obtém os cabeçalhos HTTP e informações de status da URL sem baixar o corpo completo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "A URL a ser inspecionada."
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    ]


# =========================================================================
# 5. PIPELINE PRINCIPAL DO AGENTE MCP FETCH
# =========================================================================

def extract_first_url(text: str) -> Optional[str]:
    """Localiza a primeira URL válida (http/https) presente no texto do usuário."""
    url_match = re.search(r"https?://[^\s<>\"'`\(\)]+", text)
    if url_match:
        url = url_match.group(0).rstrip(".,;:!?)'\"")
        return url
    return None


async def execute_mcp_fetch_flow(
    user_prompt: str,
    context_prompt: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fluxo orquestrado do Agente MCP Fetch:
    1. Extrai a URL alvo do prompt do usuário.
    2. Invoca o serviço seguro `core_fetch` (com SSRF protection e renderização em Markdown).
    3. Se o usuário solicitou uma análise, resumo, resposta a perguntas ou tradução,
       chama o LLM sintetizador para gerar uma resposta rica e contextualizada.
    4. Se o usuário apenas pediu o conteúdo da página, entrega o Markdown estruturado com metadados.
    """
    flow_start = time.time()
    tools_called: List[Dict[str, Any]] = []

    # Localiza a URL no prompt
    target_url = extract_first_url(user_prompt)
    if not target_url and context_prompt:
        target_url = extract_first_url(context_prompt)

    if not target_url:
        msg = (
            "🌐 **MCP Fetch:** Por favor, informe uma URL válida (ex: `https://exemplo.com.br`) "
            "para que eu possa acessar, extrair o conteúdo e analisar a página para você."
        )
        tools_called.append({"tool": "mcp_fetch", "status": "no_url_found", "duration_seconds": 0.0})
        return msg, tools_called

    # 1. Executa o fetch da URL
    t0 = time.time()
    try:
        fetch_result = await core_fetch(target_url)
    except PermissionError as perm_err:
        tools_called.append({
            "tool": "mcp_fetch",
            "arguments": {"url": target_url},
            "status": "blocked_ssrf",
            "duration_seconds": round(time.time() - t0, 3)
        })
        return f"🔒 **Acesso Bloqueado por Segurança (MCP Fetch Guardrail):**\n\n{str(perm_err)}", tools_called
    except Exception as fetch_err:
        tools_called.append({
            "tool": "mcp_fetch",
            "arguments": {"url": target_url},
            "status": f"error: {str(fetch_err)}",
            "duration_seconds": round(time.time() - t0, 3)
        })
        return f"⚠️ **Erro ao acessar `{target_url}`:** {str(fetch_err)}", tools_called

    dur = round(time.time() - t0, 3)

    tools_called.append({
        "tool": "mcp_fetch",
        "arguments": {"url": target_url},
        "status": "success" if fetch_result.get("status_code", 0) < 400 else "http_error",
        "method": fetch_result.get("method"),
        "status_code": fetch_result.get("status_code"),
        "duration_seconds": dur
    })

    page_content = fetch_result.get("content", "")
    page_title = fetch_result.get("title") or target_url
    status_code = fetch_result.get("status_code", 0)

    if status_code >= 400 or not page_content.strip():
        return (
            f"⚠️ **Falha ao acessar a URL:** O servidor retornou status `{status_code}`.\n\n"
            f"> {page_content[:400]}",
            tools_called
        )

    # 2. Avalia a intenção: o usuário fez uma pergunta/pedido específico sobre o conteúdo?
    p_lower = user_prompt.lower()
    needs_synthesis = any(k in p_lower for k in [
        "resuma", "resumo", "o que diz", "sobre o que", "quais os principais",
        "analise", "analisar", "explique", "extraia os tópicos", "extraia os topicos",
        "qual o valor", "qual o preço", "quem é", "quando", "traduza", "pontos-chave",
        "principais conclusões", "conclusoes", "compare", "responda", "segundo o link"
    ])

    # Se o usuário apenas pediu para "ler/mostrar/abrir/ver" o link
    if not needs_synthesis and len(user_prompt.split()) < 10:
        preview = page_content[:12000]
        response_md = f"""### 🌐 Conteúdo Extraído via MCP Fetch

- **Título:** {page_title}
- **URL:** [{target_url}]({target_url})
- **Status HTTP:** `{status_code}` | **Tamanho:** {fetch_result.get('length', 0):,} caracteres
- **Método de Extração:** `{fetch_result.get('method')}`

---

{preview}

{f"*... [Conteúdo truncado para exibição. Total extraído: {fetch_result.get('length', 0):,} caracteres]*" if fetch_result.get('length', 0) > 12000 else ""}
"""
        return response_md, tools_called

    # 3. Síntese Inteligente via LLM
    from config import LITELLM_BASE_URL, LITELLM_API_KEY
    from langchain_openai import ChatOpenAI

    system_prompt = f"""Você é o Agente MCP Fetch, especialista em análise e síntese precisa de páginas web.
Analise o conteúdo real extraído da página web abaixo e responda com precisão à pergunta do usuário.

REGRAS OBRIGATÓRIAS:
1. Baseie-se ESTRITAMENTE nos fatos e informações contidos no texto extraído.
2. Não invente detalhes, números ou datas que não estejam presentes na fonte.
3. Formate a resposta em Markdown estruturado, claro, com tópicos bem definidos.
4. Ao final, cite a fonte original com a URL analisada.

DADOS DA PÁGINA:
- TÍTULO: {page_title}
- URL: {target_url}

CONTEÚDO EXTRAÍDO DA PÁGINA:
{page_content[:30000]}
"""

    try:
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model="ComboFree_Smart",
            temperature=0.2,
            max_tokens=2500,
            timeout=35.0
        )
        llm_resp = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        synthesis = llm_resp.content if hasattr(llm_resp, "content") else str(llm_resp)
        tools_called.append({
            "tool": "mcp_fetch_synthesizer",
            "model": "ComboFree_Smart",
            "status": "success",
            "duration_seconds": round(time.time() - t0, 3)
        })
        return synthesis, tools_called

    except Exception as llm_err:
        print(f"[Aviso Síntese MCP Fetch]: {llm_err}")
        # Fallback para o texto extraído puro
        return (
            f"### 🌐 Conteúdo Extraído de [{page_title}]({target_url})\n\n"
            f"> *Nota: A análise de IA falhou ({llm_err}), exibindo o conteúdo direto extraído via MCP Fetch.*\n\n"
            f"{page_content[:8000]}",
            tools_called
        )
