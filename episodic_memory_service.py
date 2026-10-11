import os
import re
import json
import time
import datetime
import hashlib
from typing import Dict, Any, List, Optional
from collections import Counter

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

# Configurações
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
USER_EPISODIC_COLLECTION = os.getenv("USER_EPISODIC_COLLECTION", "user_episodic_memory")
PROFILES_FILE = os.path.join(os.path.dirname(__file__), "data", "user_profiles.json")

# Embeddings
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=EMBEDDING_MODEL
)

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

def ensure_episodic_collection_exists(client: QdrantClient, vector_size: int = 768):
    """Cria a coleção de memória episódica permanente no Qdrant se ainda não existir."""
    collections = [col.name for col in client.get_collections().collections]
    if USER_EPISODIC_COLLECTION not in collections:
        print(f"[Memória Episódica] Criando coleção '{USER_EPISODIC_COLLECTION}' no Qdrant (vetor: {vector_size})...")
        client.create_collection(
            collection_name=USER_EPISODIC_COLLECTION,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        try:
            client.create_payload_index(USER_EPISODIC_COLLECTION, "metadata.user_id", field_schema=models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(USER_EPISODIC_COLLECTION, "metadata.memory_kind", field_schema=models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(USER_EPISODIC_COLLECTION, "metadata.category", field_schema=models.PayloadSchemaType.KEYWORD)
            client.create_payload_index(USER_EPISODIC_COLLECTION, "metadata.importance", field_schema=models.PayloadSchemaType.INTEGER)
            print(f"[Memória Episódica] Índices de payload criados com sucesso.")
        except Exception as idx_err:
            print(f"[Aviso Memória Episódica Índices]: {idx_err}")


# ---------------------------------------------------------
# 1. Perfil Analítico do Usuário (Preferências e Frequência)
# ---------------------------------------------------------
def load_all_user_profiles() -> Dict[str, Any]:
    """Carrega o dicionário consolidado de perfis de usuário do arquivo JSON."""
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Aviso Perfil Usuário]: Erro ao ler {PROFILES_FILE}: {e}")
    return {}

def save_all_user_profiles(profiles: Dict[str, Any]):
    """Persiste os perfis de usuário em data/user_profiles.json com segurança."""
    try:
        os.makedirs(os.path.dirname(PROFILES_FILE), exist_ok=True)
        temp_file = f"{PROFILES_FILE}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, PROFILES_FILE)
    except Exception as e:
        print(f"[Aviso Perfil Usuário]: Erro ao gravar {PROFILES_FILE}: {e}")

def update_user_behavior_profile(
    user_id: str,
    user_email: str = "",
    category: str = "UNKNOWN",
    prompt: str = "",
    target_model: str = "",
    tools_called: Optional[List[Dict[str, Any]]] = None
):
    """
    Atualiza estatísticas, arquétipos e temas de interesse do usuário a cada interação.
    Identifica padrões recorrentes (ex: marketing/instagram vs acadêmico/denso vs dev/código).
    """
    if not user_id or user_id in ["anon", "default_user", "system_news_bot"]:
        return

    profiles = load_all_user_profiles()
    p = profiles.get(user_id, {
        "user_id": user_id,
        "email": user_email,
        "total_interactions": 0,
        "categories_count": {},
        "top_keywords": {},
        "detected_archetype": "Geral",
        "preferences": {
            "style": "equilibrado",
            "frequent_topics": []
        },
        "first_seen": datetime.datetime.now().isoformat(),
        "last_seen": datetime.datetime.now().isoformat()
    })

    if user_email and not p.get("email"):
        p["email"] = user_email

    p["total_interactions"] = p.get("total_interactions", 0) + 1
    p["last_seen"] = datetime.datetime.now().isoformat()

    # Contagem de categorias
    cats = p.setdefault("categories_count", {})
    cats[category] = cats.get(category, 0) + 1

    # Extrai palavras-chave temáticas relevantes (filtrando stopwords comuns)
    stopwords = {
        "para", "com", "uma", "como", "mais", "sobre", "qual", "fazer", "pelo", "pela",
        "isso", "este", "esse", "onde", "quando", "quem", "porque", "qual", "quais",
        "criar", "gerar", "faca", "faça", "mostre", "diga", "quero", "voce", "você"
    }
    words = [
        w.lower().strip(".,;:?!()[]'\"")
        for w in prompt.split()
        if len(w) >= 4 and w.lower() not in stopwords
    ]
    kws = p.setdefault("top_keywords", {})
    for w in words[:6]:
        kws[w] = kws.get(w, 0) + 1

    # Atualiza Arquétipo com base nos padrões mais fortes
    social_cnt = cats.get("SOCIAL_MEDIA_POST", 0) + cats.get("IMAGE_GENERATION", 0) + cats.get("IMAGE_EDIT", 0)
    academic_cnt = cats.get("DOCUMENT_RAG", 0) + cats.get("FACTUAL_GENERAL", 0) + cats.get("HEAVY_REASONING", 0)
    dev_cnt = cats.get("CODING", 0)
    search_cnt = cats.get("REALTIME_SEARCH", 0)

    total = p["total_interactions"]
    if dev_cnt / max(1, total) >= 0.35:
        p["detected_archetype"] = "Desenvolvedor de Software / Engenheiro"
        p["preferences"]["style"] = "técnico, conciso e código executável"
    elif social_cnt / max(1, total) >= 0.35:
        p["detected_archetype"] = "Criador de Conteúdo / Social Media"
        p["preferences"]["style"] = "persuasivo, engajador com foco em hooks e copywriting"
    elif academic_cnt / max(1, total) >= 0.35:
        p["detected_archetype"] = "Pesquisador Acadêmico / Estudos Densos"
        p["preferences"]["style"] = "estruturado, analítico, fundamentado e aprofundado"
    elif search_cnt / max(1, total) >= 0.35:
        p["detected_archetype"] = "Explorador de Fatos & Atualidades"
        p["preferences"]["style"] = "informativo e baseado em fontes recentes"
    else:
        p["detected_archetype"] = "Multidisciplinar / Geral"
        p["preferences"]["style"] = "equilibrado"

    # Top tópicos frequentes
    sorted_kws = sorted(kws.items(), key=lambda x: x[1], reverse=True)[:8]
    p["preferences"]["frequent_topics"] = [k[0] for k in sorted_kws]

    profiles[user_id] = p
    save_all_user_profiles(profiles)


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Retorna o perfil comportamental de um usuário específico."""
    profiles = load_all_user_profiles()
    return profiles.get(user_id)


# ---------------------------------------------------------
# 2. Memória Episódica Persistente de Longo Prazo (Qdrant)
# ---------------------------------------------------------
def store_episodic_memory(
    user_id: str,
    content: str,
    memory_kind: str = "fact",
    category: str = "GENERAL",
    importance: int = 3,
    source_title: str = "Interação do Usuário"
) -> bool:
    """
    Armazena um fato biográfico, preferência ou resultado de alto valor
    na coleção permanente de memória episódica (SEM expiração / sem TTL).
    """
    if not user_id or not content or len(content.strip()) < 15:
        return False

    client = get_qdrant_client()
    ensure_episodic_collection_exists(client, vector_size=768)

    content_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()[:16]
    now_ts = int(time.time())

    # Verifica duplicidade
    try:
        check_filter = models.Filter(
            must=[
                models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="metadata.content_hash", match=models.MatchValue(value=content_hash))
            ]
        )
        existing, _ = client.scroll(
            collection_name=USER_EPISODIC_COLLECTION,
            scroll_filter=check_filter,
            limit=1
        )
        if existing:
            return True
    except Exception as e:
        print(f"[Aviso Deduplicação Episódica]: {e}")

    doc = Document(
        page_content=content.strip(),
        metadata={
            "user_id": user_id,
            "memory_kind": memory_kind,  # 'biographic', 'preference', 'frequent_query', 'dense_research'
            "category": category,
            "importance": importance,    # 1 a 5
            "source": source_title,
            "content_hash": content_hash,
            "created_at": now_ts,
            "created_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    )

    vs = QdrantVectorStore(
        client=client,
        collection_name=USER_EPISODIC_COLLECTION,
        embedding=embeddings
    )
    vs.add_documents([doc])
    print(f"[Memória Episódica Qdrant] Gravada nova memória permanente para '{user_id}' ({memory_kind} | grau {importance}): {content[:60]}...")
    return True


async def search_episodic_memories(
    query: str,
    user_id: str,
    k: int = 3,
    min_score: float = 0.55
) -> List[Dict[str, Any]]:
    """
    Recupera memórias episódicas e preferências do usuário correlacionadas semanticamente à pergunta atual.
    """
    if not user_id or user_id in ["anon", "default_user"]:
        return []

    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if USER_EPISODIC_COLLECTION not in collections:
        return []

    try:
        vs = QdrantVectorStore(
            client=client,
            collection_name=USER_EPISODIC_COLLECTION,
            embedding=embeddings
        )
        user_filter = models.Filter(
            must=[models.FieldCondition(key="metadata.user_id", match=models.MatchValue(value=user_id))]
        )
        results = await vs.asimilarity_search_with_score(query, k=k, filter=user_filter)
        memories = []
        for doc, score in results:
            if float(score) >= min_score:
                memories.append({
                    "content": doc.page_content,
                    "kind": doc.metadata.get("memory_kind", "fact"),
                    "source": doc.metadata.get("source", "Memória Episódica"),
                    "importance": doc.metadata.get("importance", 3),
                    "created_date": doc.metadata.get("created_date", ""),
                    "score": round(float(score), 4)
                })
        return memories
    except Exception as e:
        print(f"[Aviso Busca Episódica]: {e}")
        return []


# ---------------------------------------------------------
# 3. Extração Automática de Fatos Pessoais / Preferências
# ---------------------------------------------------------
def detect_and_extract_biographic_facts(prompt: str, assistant_response: str = "") -> List[Dict[str, Any]]:
    """
    Identifica declarações explícitas de identidade, cargo, preferências ou fatos de longo prazo
    (ex: 'meu nome é...', 'eu moro em Santo André', 'trabalho como médico', 'minha empresa chama...')
    sem sobrecarregar o fluxo principal com LLMs adicionais.
    """
    facts = []
    p = prompt.strip()
    p_lower = p.lower()

    # 1. Nome
    m_nome = re.search(r"\b(meu nome [ée]|me chamo|pode me chamar de)\s+([A-Z][a-zÀ-ÿ]+(?:\s+[A-Z][a-zÀ-ÿ]+)?)", p, re.IGNORECASE)
    if m_nome:
        facts.append({
            "content": f"O usuário se chama {m_nome.group(2)}.",
            "kind": "biographic",
            "importance": 5
        })

    # 2. Localização / Moradia
    m_loc = re.search(r"\b(eu moro em|moro em|vivo em|minha cidade [ée]|sou de)\s+([A-Za-zÀ-ÿ\s]+?)(?=[,.;\n]|$)", p, re.IGNORECASE)
    if m_loc:
        loc_str = m_loc.group(2).strip()
        if len(loc_str) >= 3 and len(loc_str) <= 30:
            facts.append({
                "content": f"O usuário mora/reside em {loc_str}.",
                "kind": "biographic",
                "importance": 4
            })

    # 3. Profissão / Ocupação
    m_prof = re.search(r"\b(trabalho como|sou|minha profiss[aã]o [ée])\s+(um[a]?\s+)?([A-Za-zÀ-ÿ\s]+?)(?=[,.;\n]|$)", p, re.IGNORECASE)
    if m_prof:
        prof_str = m_prof.group(3).strip()
        if len(prof_str) >= 4 and len(prof_str) <= 35 and not any(k in prof_str.lower() for k in ["homem", "mulher", "brasileiro", "humano", "aqui", "agora"]):
            facts.append({
                "content": f"A ocupação/profissão do usuário é {prof_str}.",
                "kind": "biographic",
                "importance": 4
            })

    # 4. Preferências explícitas
    m_pref = re.search(r"\b(eu prefiro|sempre use|nunca use|gosto muito de|minha prefer[eê]ncia [ée])\s+([^\n.,;]+)", p, re.IGNORECASE)
    if m_pref:
        pref_str = m_pref.group(2).strip()
        if len(pref_str) >= 5 and len(pref_str) <= 80:
            facts.append({
                "content": f"Preferência declarada do usuário: {pref_str}.",
                "kind": "preference",
                "importance": 4
            })

    return facts


# ---------------------------------------------------------
# 4. Abastecimento Automático de RAG Pessoal por Recorrência
# ---------------------------------------------------------
def auto_curate_frequent_queries(user_id: str, prompt: str, category: str, response_text: str):
    """
    Se o usuário fizer pesquisas recorrentes sobre temas específicos (ex: entretenimento,
    estudos escolares, cotações, etc), consolida o resultado na base episódica para
    aceleração de respostas futuras.
    """
    if not user_id or user_id in ["anon", "default_user"] or not response_text or len(response_text) < 150:
        return

    profile = get_user_profile(user_id)
    if not profile:
        return

    # Se a pergunta se enquadra nos tópicos frequentes do usuário
    frequent_topics = profile.get("preferences", {}).get("frequent_topics", [])
    prompt_words = [w.lower() for w in prompt.split()]
    matched_topic = next((t for t in frequent_topics if t in prompt_words), None)

    if matched_topic and category in ["REALTIME_SEARCH", "FACTUAL_GENERAL", "DOCUMENT_RAG", "HEAVY_REASONING"]:
        title = f"Pesquisa Recorrente: {prompt[:40]}"
        content = f"Tópico Frequente do Usuário: {matched_topic.upper()}\nConsulta: {prompt}\nSíntese Consolidada: {response_text[:1200]}"
        store_episodic_memory(
            user_id=user_id,
            content=content,
            memory_kind="dense_research" if category in ["FACTUAL_GENERAL", "DOCUMENT_RAG"] else "frequent_query",
            category=category,
            importance=4,
            source_title=title
        )
