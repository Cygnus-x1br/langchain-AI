import os
import re
import time
import hashlib
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from config import LIBRECHAT_MONGO_URI
from cache_service import (
    get_db_connection,
    normalize_prompt,
    get_qdrant_client,
    embeddings,
    SEMANTIC_CACHE_COLLECTION
)

FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION", "feedback_memory")
FEEDBACK_THRESHOLD = float(os.getenv("FEEDBACK_THRESHOLD", "0.82"))

def ensure_feedback_collection(client: Optional[QdrantClient] = None):
    """Garante que a coleção de memória de feedback existe no Qdrant."""
    qc = client or get_qdrant_client()
    try:
        cols = [c.name for c in qc.get_collections().collections]
        if FEEDBACK_COLLECTION not in cols:
            qc.create_collection(
                collection_name=FEEDBACK_COLLECTION,
                vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
            )
            print(f"[Feedback Service] Coleção '{FEEDBACK_COLLECTION}' criada no Qdrant.")
    except Exception as e:
        print(f"[Feedback Service] Erro ao verificar/criar coleção Qdrant: {e}")


def invalidate_cache_for_prompt(user_prompt: str):
    """
    Invalida imediatamente o cache exato (SQLite) e o cache semântico (Qdrant)
    para evitar que uma resposta avaliada negativamente continue sendo servida.
    """
    if not user_prompt:
        return

    # 1. Invalida SQLite
    norm = normalize_prompt(user_prompt)
    if norm:
        prompt_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("DELETE FROM exact_cache WHERE prompt_hash = ?", (prompt_hash,))
                if cur.rowcount > 0:
                    print(f"[Feedback Service] Cache exato (SQLite) invalidado para: '{user_prompt[:50]}'")
                conn.commit()
        except Exception as e:
            print(f"[Feedback Service] Erro ao invalidar cache SQLite: {e}")

    # 2. Invalida Qdrant Semantic Cache
    try:
        qc = get_qdrant_client()
        cols = [c.name for c in qc.get_collections().collections]
        if SEMANTIC_CACHE_COLLECTION in cols:
            # Busca itens com alta similaridade (>0.92) e remove
            vs = QdrantVectorStore(client=qc, collection_name=SEMANTIC_CACHE_COLLECTION, embedding=embeddings)
            results = vs.similarity_search_with_score(user_prompt, k=3)
            ids_to_del = []
            for doc, score in results:
                if score >= 0.90:
                    point_id = doc.metadata.get("_id") or doc.metadata.get("id")
                    if point_id:
                        ids_to_del.append(point_id)
            if ids_to_del:
                qc.delete(
                    collection_name=SEMANTIC_CACHE_COLLECTION,
                    points_selector=models.PointIdsList(points=ids_to_del)
                )
                print(f"[Feedback Service] Removidos {len(ids_to_del)} pontos do cache semântico para a pergunta invalidada.")
    except Exception as e:
        print(f"[Feedback Service] Erro ao invalidar cache semântico: {e}")


def sync_feedbacks_from_mongodb() -> Dict[str, Any]:
    """
    Conecta ao MongoDB do LibreChat, extrai todas as avaliações (thumbsUp e thumbsDown),
    invalida caches ruins e alimenta o Qdrant com os feedbacks estruturados.
    """
    ensure_feedback_collection()
    qc = get_qdrant_client()

    client = None
    try:
        client = MongoClient(LIBRECHAT_MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client.get_default_database()
        messages_col = db["messages"]

        # Busca todas as mensagens com feedback
        cursor = messages_col.find({"feedback": {"$exists": True}})
        feedbacks_processed = 0
        thumbs_down_count = 0
        thumbs_up_count = 0

        for msg in cursor:
            fb = msg.get("feedback") or {}
            rating = fb.get("rating")  # "thumbsUp" ou "thumbsDown"
            tag = fb.get("tag", "")
            msg_id = str(msg.get("messageId") or msg.get("_id"))
            parent_id = msg.get("parentMessageId")
            user_id = str(msg.get("user") or "")

            # Conteúdo da resposta
            asst_text = ""
            if msg.get("text"):
                asst_text = msg.get("text")
            elif isinstance(msg.get("content"), list):
                asst_text = " ".join([c.get("text", "") for c in msg.get("content") if isinstance(c, dict)])

            # Localiza a mensagem do usuário (pergunta)
            user_prompt = ""
            if parent_id:
                parent_msg = messages_col.find_one({"messageId": parent_id})
                if parent_msg:
                    if not user_id and parent_msg.get("user"):
                        user_id = str(parent_msg.get("user"))
                    user_prompt = parent_msg.get("text", "")
                    if not user_prompt and isinstance(parent_msg.get("content"), list):
                        user_prompt = " ".join([c.get("text", "") for c in parent_msg.get("content") if isinstance(c, dict)])

            if not user_prompt:
                continue

            feedbacks_processed += 1

            if rating == "thumbsDown":
                thumbs_down_count += 1
                # 1. Invalida do cache
                invalidate_cache_for_prompt(user_prompt)

                # 2. Purga referências corrompidas na memória do usuário (Léxico aprendido e Memória Episódica)
                if user_id:
                    try:
                        from user_lexicon_service import remove_user_learned_patterns_for_prompt
                        from episodic_memory_service import remove_episodic_memories_for_prompt
                        purged_lex = remove_user_learned_patterns_for_prompt(user_id, user_prompt)
                        purged_epi = remove_episodic_memories_for_prompt(user_id, user_prompt)
                        if purged_lex > 0 or purged_epi > 0:
                            print(f"[Feedback Purge] Usuário '{user_id}': expurgados {purged_lex} padrões léxicos e {purged_epi} memórias episódicas.")
                    except Exception as purge_err:
                        print(f"[Aviso Feedback Purge Memória]: {purge_err}")

                # 3. Registra na memória de erros (Anti-Exemplo)
                point_id = hashlib.md5(f"down_{msg_id}".encode()).hexdigest()
                vector = embeddings.embed_query(user_prompt)
                qc.upsert(
                    collection_name=FEEDBACK_COLLECTION,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "message_id": msg_id,
                                "user_id": user_id,
                                "rating": "thumbsDown",
                                "tag": tag,
                                "prompt": user_prompt,
                                "bad_response_snippet": asst_text[:600],
                                "timestamp": int(time.time()),
                                "instruction": "Esta resposta anterior recebeu avaliação negativa (rejeitada pelo usuário). Não repita os erros ou alucinações nela presentes."
                            }
                        )
                    ]
                )

            elif rating == "thumbsUp":
                thumbs_up_count += 1
                # Registra na memória de exemplos positivos (Padrão Ouro)
                point_id = hashlib.md5(f"up_{msg_id}".encode()).hexdigest()
                vector = embeddings.embed_query(user_prompt)
                qc.upsert(
                    collection_name=FEEDBACK_COLLECTION,
                    points=[
                        models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload={
                                "message_id": msg_id,
                                "rating": "thumbsUp",
                                "tag": tag,
                                "prompt": user_prompt,
                                "good_response_snippet": asst_text[:600],
                                "timestamp": int(time.time()),
                                "instruction": "Esta resposta anterior recebeu avaliação positiva (aprovada com louvor pelo usuário). Use como referência de qualidade e precisão."
                            }
                        )
                    ]
                )

        return {
            "status": "success",
            "feedbacks_processed": feedbacks_processed,
            "thumbs_down": thumbs_down_count,
            "thumbs_up": thumbs_up_count
        }

    except Exception as e:
        print(f"[Feedback Service] Erro ao sincronizar feedbacks do MongoDB: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if client:
            client.close()


async def check_negative_feedback_guardrail(user_prompt: str) -> Optional[str]:
    """
    Verifica se a pergunta do usuário possui similaridade semântica com algum caso
    anterior que recebeu 'thumbsDown' (avaliação negativa).
    Retorna uma instrução de aviso para ser injetada no prompt do sistema se houver match.
    """
    if not user_prompt or len(user_prompt.strip()) < 6:
        return None

    try:
        qc = get_qdrant_client()
        cols = [c.name for c in qc.get_collections().collections]
        if FEEDBACK_COLLECTION not in cols:
            return None

        vector = embeddings.embed_query(user_prompt)
        hits = qc.query_points(
            collection_name=FEEDBACK_COLLECTION,
            query=vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="rating", match=models.MatchValue(value="thumbsDown"))]
            ),
            limit=1
        ).points

        if hits and hits[0].score >= FEEDBACK_THRESHOLD:
            payload = hits[0].payload or {}
            bad_snippet = payload.get("bad_response_snippet", "")
            return (
                "\n\n[DIRETRIZ DE APRENDIZADO CONTÍNUO E EVITAÇÃO DE ERRO]\n"
                "Para perguntas similares a esta, uma resposta anterior recebeu avaliação negativa dos usuários "
                f"(Score de similaridade: {round(hits[0].score, 2)}).\n"
                f"Trecho da resposta reprovada: \"{bad_snippet[:300]}...\"\n"
                "INSTRUÇÃO: Não reproduza os mesmos erros, premissas incorretas ou omissões deste trecho anterior. "
                "Forneça uma resposta revisada, precisa e factual."
            )
    except Exception as e:
        print(f"[Feedback Service] Erro ao consultar guardrail de feedback negativo: {e}")

    return None


def process_immediate_negative_feedback(
    user_id: str,
    prompt: str,
    bad_response_snippet: str = "",
    tag: str = ""
) -> Dict[str, Any]:
    """
    Executa a purga imediata de memória e cache após um polegar para baixo (thumbsDown)
    sem depender exclusivamente do ciclo de sincronização em lote.
    """
    invalidate_cache_for_prompt(prompt)
    purged_lex = 0
    purged_epi = 0

    if user_id:
        try:
            from user_lexicon_service import remove_user_learned_patterns_for_prompt
            from episodic_memory_service import remove_episodic_memories_for_prompt
            purged_lex = remove_user_learned_patterns_for_prompt(user_id, prompt)
            purged_epi = remove_episodic_memories_for_prompt(user_id, prompt)
        except Exception as e:
            print(f"[Aviso Immediate Feedback Purge]: {e}")

    try:
        ensure_feedback_collection()
        qc = get_qdrant_client()
        point_id = hashlib.md5(f"down_imm_{user_id}_{prompt[:40]}".encode()).hexdigest()
        vector = embeddings.embed_query(prompt)
        qc.upsert(
            collection_name=FEEDBACK_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "user_id": user_id,
                        "rating": "thumbsDown",
                        "tag": tag,
                        "prompt": prompt,
                        "bad_response_snippet": bad_response_snippet[:600],
                        "timestamp": int(time.time()),
                        "instruction": "Esta resposta anterior recebeu avaliação negativa (rejeitada pelo usuário). Não repita os erros ou alucinações nela presentes."
                    }
                )
            ]
        )
    except Exception as q_err:
        print(f"[Aviso Gravação Anti-Exemplo Qdrant]: {q_err}")

    return {
        "status": "success",
        "purged_lexicons": purged_lex,
        "purged_episodic_memories": purged_epi,
        "cache_invalidated": True
    }

