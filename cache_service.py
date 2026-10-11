import os
import re
import time
import sqlite3
import hashlib
import datetime
from typing import Optional, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings

# Configurações de Ambiente
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CACHE_DB_PATH = os.getenv("CACHE_DB_PATH", "/app/data/cache.db")
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "7"))
SEMANTIC_CACHE_COLLECTION = os.getenv("SEMANTIC_CACHE_COLLECTION", "semantic_cache")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.88"))

# Embeddings e Cliente Qdrant
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBEDDING_MODEL
)

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


# ---------------------------------------------------------
# Camada 1: Cache Exato Ultrarrápido (SQLite)
# ---------------------------------------------------------
def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(CACHE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_DB_PATH, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_cache_db():
    """Inicializa as tabelas do SQLite para cache exato."""
    try:
        with get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exact_cache (
                    prompt_hash TEXT PRIMARY KEY,
                    normalized_prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    category TEXT,
                    target_model TEXT,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    hit_count INTEGER DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exact_expires ON exact_cache(expires_at)")
            conn.commit()
    except Exception as e:
        print(f"[Aviso Cache SQLite Init]: {e}")


# Inicializa DB na importação
init_cache_db()


def normalize_prompt(text: str) -> str:
    """Normaliza o texto para remover variações irrelevantes de pontuação e espaços."""
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"[?!.,;:\"']+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def get_exact_cache(prompt: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Consulta o cache exato no SQLite."""
    norm = normalize_prompt(prompt)
    if not norm or len(norm) < 4:
        return None

    prompt_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    now_ts = int(time.time())

    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            if category:
                cur.execute("""
                    SELECT response, category, target_model, created_at, expires_at, hit_count
                    FROM exact_cache
                    WHERE prompt_hash = ? AND expires_at >= ? AND category = ?
                """, (prompt_hash, now_ts, category))
            else:
                cur.execute("""
                    SELECT response, category, target_model, created_at, expires_at, hit_count
                    FROM exact_cache
                    WHERE prompt_hash = ? AND expires_at >= ?
                """, (prompt_hash, now_ts))
            row = cur.fetchone()
            if row:
                # Incrementa contador de hits
                conn.execute(
                    "UPDATE exact_cache SET hit_count = hit_count + 1 WHERE prompt_hash = ?",
                    (prompt_hash,)
                )
                conn.commit()
                return {
                    "hit": True,
                    "type": "exact_cache",
                    "similarity": 1.0,
                    "response": row["response"],
                    "category": row["category"],
                    "target_model": row["target_model"],
                    "created_at": row["created_at"],
                    "expires_at": row["expires_at"],
                    "hit_count": row["hit_count"] + 1
                }
    except Exception as e:
        print(f"[Aviso Consulta Cache Exato]: {e}")

    return None


def set_exact_cache(prompt: str, response: str, category: str, model: str, ttl_days: int = CACHE_TTL_DAYS):
    """Grava o par (pergunta, resposta) no cache exato do SQLite."""
    norm = normalize_prompt(prompt)
    if not norm or len(norm) < 4 or not response:
        return

    prompt_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    now_ts = int(time.time())
    expires_at = now_ts + (ttl_days * 86400)

    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO exact_cache (prompt_hash, normalized_prompt, response, category, target_model, created_at, expires_at, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(prompt_hash) DO UPDATE SET
                    response = excluded.response,
                    category = excluded.category,
                    target_model = excluded.target_model,
                    expires_at = excluded.expires_at,
                    hit_count = hit_count + 1
            """, (prompt_hash, norm, response, category, model, now_ts, expires_at))
            conn.commit()
    except Exception as e:
        print(f"[Aviso Gravação Cache Exato]: {e}")


# ---------------------------------------------------------
# Camada 2: Cache Semântico Inteligente (Qdrant)
# ---------------------------------------------------------
def ensure_semantic_cache_collection(client: QdrantClient):
    """Garante que a coleção de cache semântico exista no Qdrant com índice de expiração."""
    collections = [c.name for c in client.get_collections().collections]
    if SEMANTIC_CACHE_COLLECTION not in collections:
        client.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
        )
        try:
            client.create_payload_index(
                SEMANTIC_CACHE_COLLECTION,
                "metadata.expires_at",
                field_schema=models.PayloadSchemaType.INTEGER
            )
        except Exception:
            pass


def prune_expired_semantic_cache(client: QdrantClient):
    """Limpa itens expirados da coleção semântica."""
    try:
        now_ts = int(time.time())
        client.delete(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.expires_at", range=models.Range(lt=now_ts))]
                )
            )
        )
    except Exception:
        pass


async def get_semantic_cache(prompt: str, category: Optional[str] = None, threshold: float = SEMANTIC_CACHE_THRESHOLD) -> Optional[Dict[str, Any]]:
    """Busca pergunta semanticamente similar no Qdrant."""
    norm = normalize_prompt(prompt)
    if not norm or len(norm) < 6:
        return None

    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if SEMANTIC_CACHE_COLLECTION not in collections:
        return None

    prune_expired_semantic_cache(client)

    now_ts = int(time.time())
    must_conditions = [
        models.FieldCondition(key="metadata.expires_at", range=models.Range(gte=now_ts))
    ]
    if category:
        must_conditions.append(
            models.FieldCondition(key="metadata.category", match=models.MatchValue(value=category))
        )
    filter_non_expired = models.Filter(must=must_conditions)

    try:
        vs = QdrantVectorStore(
            client=client,
            collection_name=SEMANTIC_CACHE_COLLECTION,
            embedding=embeddings
        )
        results = await vs.asimilarity_search_with_score(prompt, k=1, filter=filter_non_expired)
        if results:
            doc, score = results[0]
            if score >= threshold:
                cached_resp = doc.metadata.get("response") or doc.page_content
                return {
                    "hit": True,
                    "type": "semantic_cache",
                    "similarity": round(float(score), 4),
                    "original_cached_prompt": doc.metadata.get("prompt") or doc.page_content,
                    "response": cached_resp,
                    "category": doc.metadata.get("category"),
                    "target_model": doc.metadata.get("target_model"),
                    "created_at": doc.metadata.get("created_at"),
                    "expires_at": doc.metadata.get("expires_at")
                }
    except Exception as e:
        print(f"[Aviso Busca Cache Semântico]: {e}")

    return None


def set_semantic_cache(prompt: str, response: str, category: str, model: str, ttl_days: int = CACHE_TTL_DAYS):
    """Armazena o vetor da pergunta e sua respectiva resposta no Qdrant."""
    norm = normalize_prompt(prompt)
    if not norm or len(norm) < 6 or not response:
        return

    client = get_qdrant_client()
    ensure_semantic_cache_collection(client)

    now_ts = int(time.time())
    expires_at = now_ts + (ttl_days * 86400)

    doc = Document(
        page_content=prompt,
        metadata={
            "response": response,
            "prompt": prompt,
            "category": category,
            "target_model": model,
            "created_at": now_ts,
            "expires_at": expires_at
        }
    )

    try:
        vs = QdrantVectorStore(
            client=client,
            collection_name=SEMANTIC_CACHE_COLLECTION,
            embedding=embeddings
        )
        vs.add_documents([doc])
        print(f"[Cache Semântico] Pergunta '{prompt[:40]}' indexada no Qdrant.")
    except Exception as e:
        print(f"[Aviso Gravação Cache Semântico]: {e}")


# ---------------------------------------------------------
# Interface Unificada de Cache (Lookup & Armazenamento)
# ---------------------------------------------------------
async def lookup_cache(prompt: str, category: Optional[str] = None, is_followup: bool = False) -> Optional[Dict[str, Any]]:
    """Consulta de cache em dois estágios: 
    1. Exact Match via SQLite (1ms)
    2. Semantic Match via Qdrant (15-25ms) (apenas para perguntas autocontidas)
    """
    # Categorias dinâmicas/privadas que NUNCA devem usar cache persistente
    dynamic_categories = ["WEATHER_INFO", "REALTIME_SEARCH", "DOCUMENT_RAG", "VISION", "IMAGE_GENERATION", "IMAGE_EDIT"]
    if category in dynamic_categories:
        return None

    # Textos muito curtos ou muito longos (ex: uploads de livros) não devem ser cacheados
    if len(prompt) < 4 or len(prompt) > 1500:
        return None

    # 1. Consulta Exata (SQLite)
    exact_hit = get_exact_cache(prompt, category=category)
    if exact_hit:
        return exact_hit

    # Para perguntas contextuais de continuação ("classifique...", "resuma", etc.),
    # NÃO utilizar aproximação semântica genérica para não misturar conversas distintas.
    if is_followup:
        return None

    # 2. Consulta Semântica (Qdrant) com filtro de categoria e limiar de precisão
    semantic_hit = await get_semantic_cache(prompt, category=category)
    if semantic_hit:
        return semantic_hit

    return None


def store_in_cache(prompt: str, response: str, category: str, model: str, ttl_days: int = CACHE_TTL_DAYS):
    """Armazena o resultado nas duas camadas de cache."""
    dynamic_categories = ["WEATHER_INFO", "REALTIME_SEARCH", "VISION", "DOCUMENT_RAG", "IMAGE_GENERATION", "IMAGE_EDIT"]
    if category in dynamic_categories:
        return

    # Não cachear respostas vazias, muito curtas ou muito longas
    if not response or len(prompt) < 4 or len(prompt) > 1500:
        return

    # Proteção: Nunca cachear respostas negativas, falhas ou "não encontrado"
    negative_indicators = [
        "ocorreu um erro", "não encontrei", "nao encontrei", "nenhum documento",
        "não há informações", "nao ha informacoes", "não foi possível localizar",
        "nao foi possivel localizar"
    ]
    resp_lower = response.lower()
    if any(ind in resp_lower for ind in negative_indicators):
        print(f"[Cache Ignorado] Resposta negativa ou vazia detectada, não armazenando.")
        return

    # 1. Grava no SQLite (Exato)
    set_exact_cache(prompt, response, category, model, ttl_days=ttl_days)

    # 2. Grava no Qdrant (Semântico)
    set_semantic_cache(prompt, response, category, model, ttl_days=ttl_days)


def get_cache_stats() -> Dict[str, Any]:
    """Retorna contadores de entradas cacheadas nas duas camadas."""
    exact_count = 0
    now_ts = int(time.time())
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM exact_cache WHERE expires_at >= ?", (now_ts,))
            exact_count = cur.fetchone()[0]
    except Exception:
        pass

    semantic_count = 0
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if SEMANTIC_CACHE_COLLECTION in collections:
            col = client.get_collection(SEMANTIC_CACHE_COLLECTION)
            semantic_count = getattr(col, "points_count", None) or getattr(col, "vectors_count", 0)
    except Exception:
        pass

    return {
        "status": "active",
        "exact_cache_sqlite": {
            "valid_entries": exact_count,
            "db_path": CACHE_DB_PATH
        },
        "semantic_cache_qdrant": {
            "collection": SEMANTIC_CACHE_COLLECTION,
            "points_count": semantic_count,
            "threshold": SEMANTIC_CACHE_THRESHOLD
        },
        "default_ttl_days": CACHE_TTL_DAYS
    }


def clear_cache_all() -> Dict[str, Any]:
    """Limpa ambas as camadas de cache."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM exact_cache")
            conn.commit()
    except Exception:
        pass

    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if SEMANTIC_CACHE_COLLECTION in collections:
            client.delete_collection(SEMANTIC_CACHE_COLLECTION)
            ensure_semantic_cache_collection(client)
    except Exception:
        pass

    return {"status": "success", "message": "Todos os registros de cache (SQLite e Qdrant) foram apagados."}
