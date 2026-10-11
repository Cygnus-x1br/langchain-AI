import asyncio
import os
import json
import feedparser
import time
from scraper_service import scrape_url_to_markdown
from rag_service import ingest_user_text_memory

FEEDS_FILE = os.path.join(os.path.dirname(__file__), "rss_feeds.json")

def load_rss_feeds():
    """Carrega as URLs de feeds a partir do arquivo rss_feeds.json."""
    if os.path.exists(FEEDS_FILE):
        try:
            with open(FEEDS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                feeds = data.get("feeds", [])
                if feeds:
                    return feeds
        except Exception as e:
            print(f"[Aviso] Erro ao ler {FEEDS_FILE}: {e}")
    # Fallback padrão
    return [{"name": "G1 Tecnologia", "url": "https://g1.globo.com/rss/g1/tecnologia/", "category": "tecnologia"}]

def get_ttl_for_category(category: str) -> int:
    """Calcula o TTL ideal em dias com base na volatilidade e valor temporal do tema."""
    cat = (category or "").lower()
    
    # Alta volatilidade (notícias rápidas que perdem validade em poucos dias)
    if any(k in cat for k in ["politica", "esportes", "cultura", "entretenimento", "mercados"]):
        return 7   # 7 dias
        
    # Alta durabilidade e valor de referência (demoram para maturar ou requerem catálogo longo)
    # tecnologia, ia, ciencia, educacao, saude, startups
    if any(k in cat for k in ["tecnologia", "ia", "ciencia", "educacao", "saude", "startup"]):
        return 45  # 45 dias

    # Noticiário geral e economia ampla
    if any(k in cat for k in ["noticias", "geral", "economia", "internacional"]):
        return 15  # 15 dias
        
    return 15

async def process_feed(feed_info):
    if isinstance(feed_info, str):
        feed_url = feed_info
        feed_name = feed_url
        category = "geral"
    else:
        feed_url = feed_info.get("url", "")
        feed_name = feed_info.get("name", feed_url)
        category = feed_info.get("category", "geral")

    ttl_days = get_ttl_for_category(category)
    print(f"\n[Cron RAG] Processando feed: '{feed_name}' (Categoria: {category} | TTL: {ttl_days} dias)")
    feed = feedparser.parse(feed_url)
    
    # Processa os 3 artigos mais recentes de cada feed
    for entry in feed.entries[:3]:
        link = entry.link
        title = entry.title
        print(f"[Cron RAG] Coletando conteúdo: {title}")
        
        # 1. Extrai o Markdown da página usando Crawl4AI
        content_md = await scrape_url_to_markdown(link)
        
        if not content_md or "[Erro" in content_md:
            print(f"  -> Falha na extração. Pulando.")
            continue
            
        # 2. Gera os Embeddings via Ollama (bge-m3) e salva no Qdrant com TTL específico
        print(f"  -> Ingerindo no Banco Vetorial Qdrant (TTL: {ttl_days} dias)...")
        result = ingest_user_text_memory(
            text=f"Título: {title}\nCategoria: {category}\nLink: {link}\n\n{content_md}",
            user_id="system_news_bot",
            doc_title=title,
            ttl_days=ttl_days
        )
        print(f"  -> Status da Ingestão: {result.get('status')}")

async def main():
    print("==================================================")
    print(" Iniciando Cron Job de Ingestão Web RAG (Ollama)  ")
    print("==================================================")
    
    feeds = load_rss_feeds()
    print(f"[Cron RAG] Carregados {len(feeds)} feeds RSS para processamento.")
    for feed in feeds:
        try:
            await process_feed(feed)
        except Exception as f_err:
            print(f"[Erro no feed {feed}]: {f_err}")
        
    print("\n[Cron RAG] Ingestão concluída.")

if __name__ == "__main__":
    asyncio.run(main())
