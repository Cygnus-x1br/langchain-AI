import os
import asyncio
import re
from typing import Dict, Any
import httpx
from bs4 import BeautifulSoup

try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None

def extract_clean_html_text(html: str) -> str:
    """Extrai texto legível e estruturado de HTML caso o Crawl4AI falhe ou atinja timeout."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript", "header", "svg", "form"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        return f"[Erro parsing HTML]: {e}"

async def scrape_url_fast_fallback(url: str) -> str:
    """Fallback ultrarrápido via HTTPX e BeautifulSoup/pypdf."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,*/*;q=0.8"
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                c_type = resp.headers.get("content-type", "").lower()
                # Suporte automático a documentos PDF (ex: informativos, regulamentos, editais)
                if "application/pdf" in c_type or resp.content.startswith(b"%PDF"):
                    try:
                        import io
                        import pypdf
                        reader = pypdf.PdfReader(io.BytesIO(resp.content))
                        pdf_pages = []
                        for page_idx, page in enumerate(reader.pages, 1):
                            t = page.extract_text() or ""
                            if t.strip():
                                pdf_pages.append(f"--- Página {page_idx} ---\n{t.strip()}")
                        pdf_text = "\n\n".join(pdf_pages)
                        if len(pdf_text) > 15000:
                            pdf_text = pdf_text[:15000] + "\n...[Conteúdo PDF truncado por limite de tamanho]..."
                        return pdf_text or "[PDF vazio ou apenas com imagens]"
                    except Exception as pdf_err:
                        print(f"[Aviso PDF Parse {url}]: {pdf_err}")

                clean_text = extract_clean_html_text(resp.text)
                if len(clean_text) > 15000:
                    clean_text = clean_text[:15000] + "\n...[Conteúdo truncado por limite de tamanho]..."
                return clean_text
            else:
                return f"[Erro HTTP {resp.status_code}] Falha ao acessar {url}"
    except Exception as ex:
        return f"[Erro Scrape Fallback]: {ex}"

async def scrape_url_to_markdown(url: str) -> str:
    """
    Estratégia de extração em camadas:
    1. Tenta extração ultrarrápida via HTTPX + BeautifulSoup (< 1s para sites estáticos, GitHub, blogs, etc).
    2. Se retornar pouco conteúdo (< 250 caracteres, típico de SPAs puras em React/Vue) e Crawl4AI estiver disponível,
       aciona o Crawl4AI com headless browser com timeout estrito de 6s.
    """
    # 1. Tentativa rápida direta
    fast_text = await scrape_url_fast_fallback(url)
    if fast_text and not fast_text.startswith("[Erro") and len(fast_text.strip()) > 300:
        return fast_text

    # 2. Se for SPA ou página dinâmica e o Crawl4AI estiver disponível:
    if AsyncWebCrawler is not None:
        try:
            print(f"[Crawl4AI] Conteúdo dinâmico/SPA detectado. Executando browser para: {url}")
            async def _run_crawl():
                async with AsyncWebCrawler(verbose=False) as crawler:
                    return await crawler.arun(url=url, bypass_cache=False)

            result = await asyncio.wait_for(_run_crawl(), timeout=6.0)
            if result.success and result.markdown and len(result.markdown.strip()) > 200:
                md_content = result.markdown
                if len(md_content) > 15000:
                    md_content = md_content[:15000] + "\n...[Conteúdo truncado por limite de tamanho]..."
                return md_content
        except (asyncio.TimeoutError, Exception) as e:
            print(f"[Crawl4AI Timeout/Erro em {url}]: {e}")

    # Retorna o melhor resultado obtido na extração rápida
    return fast_text


