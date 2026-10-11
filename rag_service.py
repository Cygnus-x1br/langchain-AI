import os
import re
import glob
import time
import hashlib
import datetime
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configurações de Ambiente
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "knowledge_base")
USER_TEMP_COLLECTION = os.getenv("USER_TEMP_COLLECTION", "user_temp_memory")
USER_MEMORY_TTL_DAYS = int(os.getenv("USER_MEMORY_TTL_DAYS", "30"))
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "/app/documents")

# Inicialização de Embeddings e Cliente Qdrant
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBEDDING_MODEL
)

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection_exists(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """Cria a coleção com métrica de Cosseno se ainda não existir."""
    collections = [col.name for col in client.get_collections().collections]
    if collection_name not in collections:
        print(f"[RAG] Criando coleção '{collection_name}' no Qdrant (vetor: {vector_size}, métrica: COSINE)...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        
        # Índices de payload específicos para memória do usuário
        if collection_name == USER_TEMP_COLLECTION:
            try:
                client.create_payload_index(collection_name, "metadata.user_id", field_schema=models.PayloadSchemaType.KEYWORD)
                client.create_payload_index(collection_name, "metadata.expires_at", field_schema=models.PayloadSchemaType.INTEGER)
                client.create_payload_index(collection_name, "metadata.doc_hash", field_schema=models.PayloadSchemaType.KEYWORD)
                print(f"[RAG] Índices de payload criados para '{USER_TEMP_COLLECTION}'.")
            except Exception as idx_err:
                print(f"[Aviso RAG Índice]: {idx_err}")


def sanitize_document_content(text: str) -> str:
    """Sanitiza o conteúdo textual de documentos para evitar Injeção de Código e Prompt Injection."""
    if not text:
        return ""

    # 1. Remoção de caracteres invisíveis e de controle (Zero-Width Characters)
    zero_width_chars = [
        "\u200B", "\u200C", "\u200D", "\u200E", "\u200F",
        "\uFEFF", "\u00AD", "\u202A", "\u202B", "\u202C", "\u202D", "\u202E"
    ]
    for ch in zero_width_chars:
        text = text.replace(ch, "")

    # 2. Remoção de blocos executáveis HTML/JS (Scripts, Iframes, Embeds)
    text = re.sub(r"(?is)<script.*?>.*?</script>", " [SCRIPT_REMOVIDO] ", text)
    text = re.sub(r"(?is)<iframe.*?>.*?</iframe>", " [IFRAME_REMOVIDO] ", text)
    text = re.sub(r"(?is)<object.*?>.*?</object>", " [OBJETO_REMOVIDO] ", text)
    text = re.sub(r"(?is)<embed.*?>.*?</embed>", " [EMBED_REMOVIDO] ", text)

    # 3. Neutralização de padrões clássicos de Injeção de Prompt (Prompt Injection Guardrail)
    injection_patterns = [
        r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
        r"(?i)\bsystem\s+override\b",
        r"(?i)\bdisregard\s+(all\s+)?(instructions|rules)\b",
        r"(?i)\byou\s+are\s+now\s+(in\s+developer\s+mode|dan)\b",
        r"(?i)\bprint\s+(system\s+prompt|all\s+documents|passwords)\b",
        r"(?i)\bshow\s+your\s+instructions\b",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text):
            text = re.sub(pattern, " [TENTATIVA_INJECAO_NEUTRALIZADA] ", text)

    # 4. Normalização de múltiplos espaços e quebras de linha
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file_content(file_path: str) -> List[Document]:
    """Carrega e sanitiza o conteúdo de um arquivo baseado em sua extensão."""
    ext = os.path.splitext(file_path)[1].lower()
    docs = []

    blocked_extensions = [".exe", ".bin", ".sh", ".bat", ".cmd", ".ps1", ".vbs", ".jar", ".app", ".msi"]
    if ext in blocked_extensions:
        print(f"[RAG Segurança] Arquivo com extensão potencialmente perigosa ignorado: {file_path}")
        return []

    try:
        if ext in [".txt", ".md", ".csv", ".json", ".yaml", ".yml"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                clean_content = sanitize_document_content(content)
                if clean_content:
                    docs.append(Document(
                        page_content=clean_content,
                        metadata={"source": os.path.basename(file_path), "path": file_path}
                    ))

        elif ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    clean_content = sanitize_document_content(text)
                    if clean_content:
                        docs.append(Document(
                            page_content=clean_content,
                            metadata={"source": os.path.basename(file_path), "page": page_num + 1}
                        ))

        elif ext == ".docx":
            import docx2txt
            text = docx2txt.process(file_path)
            if text:
                clean_content = sanitize_document_content(text)
                if clean_content:
                    docs.append(Document(
                        page_content=clean_content,
                        metadata={"source": os.path.basename(file_path)}
                    ))
    except Exception as e:
        print(f"[RAG Erro ao carregar arquivo {file_path}]: {e}")

    return docs


# ---------------------------------------------------------
# Gestão de Base Permanente (/app/documents)
# ---------------------------------------------------------
def ingest_documents_from_dir(docs_dir: str = DOCUMENTS_DIR) -> Dict[str, Any]:
    """Lê todos os arquivos do diretório, sanitiza, fatia em chunks e indexa no Qdrant."""
    client = get_qdrant_client()
    
    try:
        sample_embed = embeddings.embed_query("teste")
        vector_size = len(sample_embed)
    except Exception as e:
        print(f"[Aviso RAG Embeddings]: Não foi possível obter dimensões do modelo: {e}. Usando 768.")
        vector_size = 768

    ensure_collection_exists(client, RAG_COLLECTION_NAME, vector_size=vector_size)

    pattern = os.path.join(docs_dir, "*")
    files = [f for f in glob.glob(pattern) if os.path.isfile(f) and not f.endswith(".gitkeep")]

    if not files:
        return {"status": "no_files_found", "message": f"Nenhum arquivo encontrado em {docs_dir}", "files_indexed": 0, "chunks": 0}

    all_docs = []
    for file_path in files:
        docs = load_file_content(file_path)
        all_docs.extend(docs)

    if not all_docs:
        return {"status": "empty_documents", "message": "Arquivos encontrados porém vazios, bloqueados ou ilegíveis", "files_indexed": 0, "chunks": 0}

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=RAG_COLLECTION_NAME,
        embedding=embeddings
    )
    vector_store.add_documents(chunks)

    print(f"[RAG Sucesso] {len(files)} arquivos indexados com {len(chunks)} chunks gerados na coleção '{RAG_COLLECTION_NAME}'.")
    return {
        "status": "success",
        "files_indexed": [os.path.basename(f) for f in files],
        "total_chunks": len(chunks)
    }


# ---------------------------------------------------------
# Gestão de Memória Temporária do Usuário (TTL 30 Dias)
# ---------------------------------------------------------
def prune_expired_memories() -> int:
    """Remove automaticamente documentos expirados (expires_at < timestamp atual)."""
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        if USER_TEMP_COLLECTION not in collections:
            return 0

        now_ts = int(time.time())
        prune_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.expires_at",
                    range=models.Range(lt=now_ts)
                )
            ]
        )
        res = client.delete(
            collection_name=USER_TEMP_COLLECTION,
            points_selector=models.FilterSelector(filter=prune_filter)
        )
        return 1
    except Exception as e:
        print(f"[Aviso RAG Limpeza de Memória Expirada]: {e}")
        return 0


def ingest_user_text_memory(
    text: str,
    user_id: str,
    conversation_id: Optional[str] = None,
    doc_title: Optional[str] = None,
    ttl_days: int = USER_MEMORY_TTL_DAYS
) -> Dict[str, Any]:
    """Indexa um texto/documento longo fornecido pelo usuário com expiração programada (TTL)."""
    if not text or not user_id:
        return {"status": "error", "message": "Texto ou user_id inválido"}

    client = get_qdrant_client()
    ensure_collection_exists(client, USER_TEMP_COLLECTION, vector_size=768)

    # Identificação por hash para evitar duplicação do mesmo texto
    clean_text = sanitize_document_content(text)
    if not clean_text:
        return {"status": "error", "message": "Conteúdo vazio após sanitização"}

    doc_hash = hashlib.sha256(clean_text[:5000].encode("utf-8")).hexdigest()[:16]
    now_ts = int(time.time())
    expires_at = now_ts + (ttl_days * 86400)
    expires_str = datetime.datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S")

    # Verifica se já foi indexado para esse usuário
    try:
        check_filter = models.Filter(
            must=[
                models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="metadata.doc_hash", match=models.MatchValue(value=doc_hash)),
                models.FieldCondition(key="metadata.expires_at", range=models.Range(gte=now_ts))
            ]
        )
        existing, _ = client.scroll(
            collection_name=USER_TEMP_COLLECTION,
            scroll_filter=check_filter,
            limit=1
        )
        if existing:
            print(f"[RAG Memória] Documento '{doc_hash}' já indexado para o usuário '{user_id}'.")
            return {
                "status": "already_indexed",
                "doc_hash": doc_hash,
                "message": "Documento já presente na memória ativa do usuário.",
                "expires_at": expires_str
            }
    except Exception as check_err:
        print(f"[Aviso RAG Verificação Hash]: {check_err}")

    # Quebra o documento em chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    title = doc_title or f"Documento_{doc_hash[:6]}"
    raw_chunks = text_splitter.split_text(clean_text)

    docs = []
    for idx, chunk_text in enumerate(raw_chunks, 1):
        docs.append(Document(
            page_content=chunk_text,
            metadata={
                "user_id": user_id,
                "conversation_id": conversation_id or "default",
                "source": title,
                "chunk_index": idx,
                "doc_hash": doc_hash,
                "created_at": now_ts,
                "expires_at": expires_at
            }
        ))

    vs = QdrantVectorStore(
        client=client,
        collection_name=USER_TEMP_COLLECTION,
        embedding=embeddings
    )
    vs.add_documents(docs)

    print(f"[RAG Memória Sucesso] Documento '{title}' ({len(docs)} chunks) indexado para usuário '{user_id}'. Expira em {expires_str}.")
    return {
        "status": "success",
        "doc_title": title,
        "doc_hash": doc_hash,
        "chunks_indexed": len(docs),
        "expires_at": expires_str,
        "ttl_days": ttl_days
    }


# ---------------------------------------------------------
# Busca Unificada (Base Permanente + Memória Temporária)
# ---------------------------------------------------------
async def search_knowledge_base(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Busca trechos relevantes estritamente na base de conhecimento permanente."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if RAG_COLLECTION_NAME not in collections:
        return []

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=RAG_COLLECTION_NAME,
        embedding=embeddings
    )

    results = await vector_store.asimilarity_search_with_score(query, k=k)
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "base_permanente"),
            "page": doc.metadata.get("page", None),
            "memory_type": "permanente",
            "score": round(float(score), 4)
        })

    return formatted_results


async def search_unified_rag(query: str, user_id: str = "default_user", k: int = 4) -> List[Dict[str, Any]]:
    """Busca trechos relevantes tanto na base perene quanto na memória temporária ativa do usuário."""
    prune_expired_memories()

    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    all_results = []

    # 1. Base Permanente (Documentos corporativos/manuais)
    if RAG_COLLECTION_NAME in collections:
        try:
            kb_res = await search_knowledge_base(query, k=k)
            all_results.extend(kb_res)
        except Exception as e:
            print(f"[RAG Erro Busca KB]: {e}")

    # 2. Memória Temporária Pessoal do Usuário (Filtrada por user_id e expires_at)
    if USER_TEMP_COLLECTION in collections and user_id:
        try:
            vs_user = QdrantVectorStore(
                client=client,
                collection_name=USER_TEMP_COLLECTION,
                embedding=embeddings
            )
            now_ts = int(time.time())
            user_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.user_id", 
                        match=models.MatchAny(any=[user_id, "system_news_bot", "default_user"])
                    ),
                    models.FieldCondition(key="metadata.expires_at", range=models.Range(gte=now_ts))
                ]
            )
            user_res = await vs_user.asimilarity_search_with_score(query, k=k, filter=user_filter)
            for doc, score in user_res:
                expires_ts = doc.metadata.get("expires_at", 0)
                expires_str = datetime.datetime.fromtimestamp(expires_ts).strftime("%Y-%m-%d") if expires_ts else "30 dias"
                all_results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "memoria_temporaria"),
                    "page": doc.metadata.get("page", None),
                    "memory_type": f"temporaria (expira em {expires_str})",
                    "score": round(float(score), 4)
                })
        except Exception as e:
            print(f"[RAG Erro Busca Memória Usuário]: {e}")

    # 3. Memória Episódica Persistente de Longo Prazo do Usuário (Sem expiração / permanente)
    if "user_episodic_memory" in collections and user_id and user_id not in ["anon", "default_user"]:
        try:
            vs_ep = QdrantVectorStore(
                client=client,
                collection_name="user_episodic_memory",
                embedding=embeddings
            )
            ep_filter = models.Filter(
                must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
            )
            ep_res = await vs_ep.asimilarity_search_with_score(query, k=k, filter=ep_filter)
            for doc, score in ep_res:
                kind = doc.metadata.get("memory_kind", "fato")
                src = doc.metadata.get("source", "Memória Episódica")
                all_results.append({
                    "content": doc.page_content,
                    "source": f"{src} ({kind})",
                    "page": None,
                    "memory_type": "episodica_permanente",
                    "score": round(float(score), 4)
                })
        except Exception as e:
            print(f"[RAG Erro Busca Memória Episódica]: {e}")

    # Ordena todos os resultados unificados pela métrica de similaridade de cosseno
    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k]


def get_user_memories_status(user_id: Optional[str] = None) -> Dict[str, Any]:
    """Retorna contadores de documentos e vetores da base perene e da memória temporária."""
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    
    kb_points = 0
    if RAG_COLLECTION_NAME in collections:
        col = client.get_collection(RAG_COLLECTION_NAME)
        kb_points = getattr(col, "points_count", None) or getattr(col, "vectors_count", 0)

    user_temp_points = 0
    if USER_TEMP_COLLECTION in collections:
        col = client.get_collection(USER_TEMP_COLLECTION)
        user_temp_points = getattr(col, "points_count", None) or getattr(col, "vectors_count", 0)

    user_specific_points = 0
    if USER_TEMP_COLLECTION in collections and user_id:
        try:
            points, _ = client.scroll(
                collection_name=USER_TEMP_COLLECTION,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
                ),
                limit=100
            )
            user_specific_points = len(points)
        except Exception:
            pass

    return {
        "permanent_collection": {
            "name": RAG_COLLECTION_NAME,
            "total_points": kb_points
        },
        "temporary_collection": {
            "name": USER_TEMP_COLLECTION,
            "total_points": user_temp_points,
            "ttl_days_default": USER_MEMORY_TTL_DAYS,
            "user_active_points": user_specific_points if user_id else "todos"
        }
    }


def clear_user_memories(user_id: str) -> Dict[str, Any]:
    """Limpa todas as memórias temporárias pertencentes a um determinado usuário."""
    if not user_id:
        return {"status": "error", "message": "user_id obrigatório"}
    
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if USER_TEMP_COLLECTION not in collections:
        return {"status": "success", "deleted": 0}

    del_filter = models.Filter(
        must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
    )
    client.delete(collection_name=USER_TEMP_COLLECTION, points_selector=models.FilterSelector(filter=del_filter))
    return {"status": "success", "message": f"Memórias temporárias de '{user_id}' foram removidas com sucesso."}
