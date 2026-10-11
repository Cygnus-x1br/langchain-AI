import os
import re
import json
import time
import asyncio
from typing import Optional
import httpx
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_service import transcribe_audio_cascade, synthesize_speech_openai
from dashboard_service import (
    get_available_log_dates,
    load_logs_for_date,
    calculate_dashboard_metrics,
    read_markdown_document
)

# Configurações globais, logs e utilitários
from config import (
    get_now_local,
    write_log,
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    OLLAMA_BASE_URL,
    IMAGE_STORAGE_DIR,
    ORCHESTRATOR_DEBUG,
    message_has_image,
    history_has_images,
    sanitize_messages_for_ollama,
    ensure_alternating_roles,
    is_context_dependent_prompt,
    get_conversation_state,
    update_conversation_state,
    is_debug_allowed_for_user
)

# RAG Service (Qdrant)
from rag_service import (
    ingest_documents_from_dir,
    ingest_user_text_memory,
    get_user_memories_status,
    clear_user_memories,
    prune_expired_memories
)

# Admin & Runtime Config Service
from admin_config_service import (
    load_runtime_config,
    save_runtime_config,
    reset_runtime_config,
    verify_admin_password,
    PROMPTS_METADATA,
    DEFAULT_GUARDRAILS,
    DEFAULT_CATEGORY_MODELS,
    MODEL_CATALOG
)

# Cache Híbrido (SQLite + Qdrant)
from cache_service import lookup_cache, store_in_cache, get_cache_stats, clear_cache_all

# Feedback Loop & Memória Negativa (LibreChat Thumbs Up/Down)
from feedback_service import (
    sync_feedbacks_from_mongodb,
    check_negative_feedback_guardrail,
    invalidate_cache_for_prompt
)

# Serviços Modulares de Tarefas
from classifier import classify_intent
from agent_service import execute_agent_flow, execute_rag_flow, weather_tool, tavily_tool
from image_service import execute_image_flow
from social_service import execute_social_post_flow, extract_all_images
from data_export_service import execute_data_export_flow
from video_service import execute_short_video_flow
from coding_service import (
    get_coding_candidates,
    resolve_coding_tier,
    prepare_coding_messages,
    is_code_response_complete
)
from chat_service import get_chat_candidates
from factual_service import get_factual_candidates
from reasoning_service import get_heavy_reasoning_candidates, get_local_thinking_candidates

# Memória Episódica Persistente & Perfis de Usuário
from episodic_memory_service import (
    update_user_behavior_profile,
    get_user_profile,
    load_all_user_profiles,
    store_episodic_memory,
    search_episodic_memories,
    detect_and_extract_biographic_facts,
    auto_curate_frequent_queries
)

app = FastAPI(title="LangChain Modular Smart Router, Agent & RAG")

# Montagem de imagens geradas para acesso estático
app.mount("/images", StaticFiles(directory=IMAGE_STORAGE_DIR), name="images")


# ---------------------------------------------------------
# Rotas de Gestão do Cache Híbrido (SQLite + Qdrant)
# ---------------------------------------------------------
@app.get("/v1/cache/status")
async def cache_status():
    """Retorna estatísticas do cache híbrido (SQLite Exato + Qdrant Semântico)."""
    try:
        return get_cache_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/cache/clear")
async def cache_clear():
    """Limpa todas as entradas de cache (SQLite e Qdrant)."""
    try:
        return clear_cache_all()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------
# Rotas de Gestão do RAG
# ---------------------------------------------------------
@app.get("/v1/rag/status")
async def rag_status():
    """Retorna o status do banco Qdrant para as bases permanente e temporária."""
    try:
        stats = get_user_memories_status()
        return {"qdrant": "connected", "details": stats}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/rag/user-memory/status")
async def user_memory_status(request: Request, user_id: str = None):
    """Retorna o status da memória temporária para um usuário específico."""
    try:
        uid = user_id or request.headers.get("x-user-id") or "default_user"
        stats = get_user_memories_status(user_id=uid)
        return {"status": "success", "user_id": uid, "details": stats}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/rag/user-memory/ingest")
async def user_memory_ingest(request: Request):
    """Indexa diretamente um texto/documento longo na memória temporária do usuário com TTL."""
    try:
        data = await request.json()
        text = data.get("text", "")
        uid = data.get("user_id") or request.headers.get("x-user-id") or "default_user"
        cid = data.get("conversation_id") or request.headers.get("x-conversation-id")
        title = data.get("title") or "Texto do Usuário"
        ttl = int(data.get("ttl_days", 30))
        
        result = ingest_user_text_memory(
            text=text,
            user_id=uid,
            conversation_id=cid,
            doc_title=title,
            ttl_days=ttl
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/v1/rag/user-memory/clear")
async def user_memory_clear(request: Request, user_id: str = None):
    """Limpa todas as memórias temporárias pertencentes a um determinado usuário."""
    try:
        uid = user_id or request.headers.get("x-user-id") or "default_user"
        res = clear_user_memories(user_id=uid)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/rag/ingest")
async def rag_ingest():
    """Dispara a leitura e indexação vetorial de todos os arquivos da pasta /app/documents."""
    try:
        result = ingest_documents_from_dir()
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------
# Rotas do Dashboard de Monitoramento, Logs e Documentação
# ---------------------------------------------------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view():
    """Retorna a interface visual completa do Dashboard de Métricas, Logs e Documentação."""
    dash_path = "/app/dashboard.html"
    if not os.path.exists(dash_path):
        dash_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dash_path):
        with open(dash_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard não encontrado</h1>", status_code=404)


@app.get("/v1/dashboard/dates")
async def dashboard_dates():
    """Retorna a lista de datas disponíveis nos arquivos de log."""
    dates = get_available_log_dates()
    return {"dates": dates}


@app.get("/v1/dashboard/data")
async def dashboard_data(date: Optional[str] = None):
    """Retorna os logs tabulados e métricas calculadas para a data especificada."""
    logs = load_logs_for_date(target_date=date)
    metrics = calculate_dashboard_metrics(logs)
    return {
        "date": date or (get_available_log_dates()[0] if get_available_log_dates() else None),
        "metrics": metrics,
        "logs": logs
    }


@app.get("/v1/dashboard/docs")
async def dashboard_docs(name: str = "architecture"):
    """Retorna o conteúdo Markdown dos arquivos de documentação do sistema."""
    content = read_markdown_document(doc_name=name)
    return {"name": name, "content": content}


# ---------------------------------------------------------
# Rotas Administrativas e de Gestão de Guardrails / Prompts
# ---------------------------------------------------------
@app.get("/v1/admin/config")
async def admin_get_config():
    """Retorna os prompts ativos, guardrails numéricos e metadados descritivos."""
    try:
        config = load_runtime_config()
        return {
            "status": "success",
            "prompts": config.get("prompts", {}),
            "guardrails": config.get("guardrails", {}),
            "metadata": PROMPTS_METADATA,
            "default_guardrails": DEFAULT_GUARDRAILS,
            "updated_at": config.get("updated_at")
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


def _check_admin_auth(request: Request, body_data: Optional[dict] = None) -> bool:
    """Valida se a senha administrativa enviada (via header ou payload) está correta."""
    auth_header = request.headers.get("X-Admin-Password") or request.headers.get("x-admin-password")
    if auth_header and verify_admin_password(auth_header):
        return True
    if body_data and isinstance(body_data, dict):
        pwd = body_data.get("admin_password") or body_data.get("password")
        if pwd and verify_admin_password(pwd):
            return True
    return False


@app.post("/v1/admin/auth/verify")
async def admin_verify_auth(request: Request):
    """Valida credenciais administrativas para o Dashboard."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    if _check_admin_auth(request, body):
        return {"status": "success", "authenticated": True, "message": "Senha administrativa válida."}
    return JSONResponse(status_code=401, content={"status": "error", "authenticated": False, "message": "Senha administrativa incorreta ou não fornecida."})


@app.post("/v1/admin/config")
async def admin_save_config(request: Request):
    """Atualiza de forma segura prompts e guardrails em tempo de execução sem reiniciar o container."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        saved = save_runtime_config(body)
        return {
            "status": "success",
            "message": "Configurações e guardrails atualizados com sucesso!",
            "data": saved
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/config/reset")
async def admin_reset_config(request: Request):
    """Restaura todos os prompts e guardrails para os padrões originais de fábrica."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        scope = body.get("scope") if body.get("scope") in ("all", "prompts", "models") else "prompts"
        reset_data = reset_runtime_config(scope=scope)
        return {
            "status": "success",
            "message": "Configurações restauradas com sucesso para os padrões de fábrica!",
            "data": reset_data
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ---------------------------------------------------------
# Gestão de Modelos Padrão por Categoria & Prioridade de Fallback
# ---------------------------------------------------------
@app.get("/v1/admin/models")
async def admin_get_models():
    """Retorna as categorias, modelos permitidos, ordem de fábrica e ordem efetiva atual."""
    try:
        config = load_runtime_config()
        overrides = config.get("category_models", {})
        categories = {}
        for key, spec in DEFAULT_CATEGORY_MODELS.items():
            options = spec.get("options") or spec["models"]
            categories[key] = {
                **spec,
                "factory_models": spec["models"],
                "options": options,
                "active_models": overrides.get(key) or spec["models"],
                "customized": key in overrides
            }
        return {
            "status": "success",
            "categories": categories,
            "catalog": MODEL_CATALOG,
            "updated_at": config.get("updated_at")
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/models")
async def admin_save_models(request: Request):
    """Salva a prioridade de modelos por categoria. Payload: {category_models: {KEY: [modelos] | null}}."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    payload = body.get("category_models")
    if not isinstance(payload, dict):
        return JSONResponse(status_code=400, content={"status": "error", "message": "Campo 'category_models' obrigatório."})

    try:
        saved = save_runtime_config({"category_models": payload})
        return {
            "status": "success",
            "message": "Prioridades de modelos atualizadas com sucesso!",
            "category_models": saved.get("category_models", {})
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/models/reset")
async def admin_reset_models(request: Request):
    """Restaura todas as categorias para a cadeia de modelos de fábrica (preserva prompts e guardrails)."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        reset_runtime_config(scope="models")
        return {"status": "success", "message": "Modelos restaurados para os padrões de fábrica!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/cache/clear")
async def admin_cache_clear(request: Request):
    """Limpa todas as entradas de cache (SQLite exato e Qdrant semântico)."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        res = clear_cache_all()
        return {
            "status": "success",
            "message": "Cache limpo com sucesso!",
            "details": res
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/rag/reindex")
async def admin_rag_reindex(request: Request):
    """Dispara a releitura e reindexação vetorial de todos os arquivos da pasta /app/documents."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        result = ingest_documents_from_dir()
        return {
            "status": "success",
            "message": "Reindexação vetorial do RAG concluída com sucesso!",
            "details": result
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/rag/prune")
async def admin_rag_prune(request: Request):
    """Limpa memórias temporárias expiradas por TTL no Qdrant."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        pruned_count = prune_expired_memories()
        return {
            "status": "success",
            "message": f"Varredura de expiração concluída. Status: {pruned_count}",
            "pruned": pruned_count
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/v1/admin/feedback/sync")
async def admin_feedback_sync(request: Request):
    """Sincroniza imediatamente os feedbacks de polegar positivo/negativo do LibreChat via MongoDB."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if not _check_admin_auth(request, body):
        return JSONResponse(status_code=401, content={"status": "error", "message": "Acesso negado: Senha administrativa incorreta ou não fornecida."})

    try:
        res = await sync_feedbacks_from_mongodb()
        return {
            "status": "success",
            "message": "Sincronização de feedbacks concluída!",
            "details": res
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ---------------------------------------------------------
# Rotas de Gestão de Perfis Comportamentais & Memória Episódica
# ---------------------------------------------------------
@app.get("/v1/user/profile")
async def user_profile_endpoint(request: Request, user_id: Optional[str] = None):
    """Retorna o perfil comportamental, arquétipo detectado e temas frequentes do usuário."""
    uid = user_id or request.headers.get("X-User-Id") or request.headers.get("x-user-id")
    if not uid:
        all_profiles = load_all_user_profiles()
        return {"profiles_count": len(all_profiles), "profiles": all_profiles}
    
    profile = get_user_profile(uid)
    if not profile:
        return JSONResponse(status_code=404, content={"message": f"Nenhum perfil encontrado para '{uid}'"})
    return profile


@app.get("/v1/user/memories")
async def user_memories_endpoint(request: Request, user_id: Optional[str] = None, q: Optional[str] = None):
    """Consulta memórias episódicas de longo prazo do usuário (fatos biográficos, preferências, pesquisas recorrentes)."""
    uid = user_id or request.headers.get("X-User-Id") or request.headers.get("x-user-id")
    if not uid:
        return JSONResponse(status_code=400, content={"message": "user_id obrigatório"})
    
    query = q or "preferências, dados do usuário, perfil e pesquisas anteriores"
    mems = await search_episodic_memories(query=query, user_id=uid, k=6, min_score=0.45)
    return {
        "user_id": uid,
        "query": query,
        "memories_count": len(mems),
        "memories": mems
    }



# ---------------------------------------------------------
# Rota Direta de Geração de Vídeos Curtos (Reels/Shorts/TikTok)
# ---------------------------------------------------------
@app.post("/v1/video/short")
async def generate_short_video_endpoint(request: Request):
    """Endpoint dedicado para roteirização e renderização direta de vídeos curtos verticais."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    prompt = body.get("prompt") or body.get("topic") or body.get("theme") or ""
    if not prompt:
        return JSONResponse(status_code=400, content={"status": "error", "message": "O campo 'prompt' ou 'topic' é obrigatório."})
    
    output_text, tools_called = await execute_short_video_flow(user_prompt=prompt)
    return {
        "status": "success",
        "result": output_text,
        "tools_called": tools_called
    }


# ---------------------------------------------------------
# Rotas de Gestão do Feedback Loop (LibreChat Thumbs Up/Down)
# ---------------------------------------------------------
@app.post("/v1/feedback/sync")
async def feedback_sync():
    """
    Sincroniza imediatamente todas as avaliações do MongoDB do LibreChat:
    invalida respostas ruins do cache e alimenta a memória de anti-exemplos no Qdrant.
    """
    try:
        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(None, sync_feedbacks_from_mongodb)
        return res
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.on_event("startup")
async def schedule_feedback_sync_task():
    """Tarefa periódica de fundo para sincronizar feedbacks do LibreChat a cada 2 minutos."""
    async def periodic_sync():
        await asyncio.sleep(10) # Aguarda inicialização
        while True:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, sync_feedbacks_from_mongodb)
            except Exception as e:
                print(f"[Aviso Feedback Periodic Sync]: {e}")
            await asyncio.sleep(120)

    asyncio.create_task(periodic_sync())


# ---------------------------------------------------------
# Motor de Execução e Fallback de Inferência (Stream & JSON)
# ---------------------------------------------------------
async def execute_candidates_pipeline(
    route_candidates: list,
    payload: dict,
    messages: list,
    stream: bool,
    category: str,
    cache_lookup_key: str,
    log_entry: dict,
    req_start_time: float,
    conversation_id: str = None,
    user_prompt: str = "",
    user_email: str = "",
    user_id: str = ""
):
    """Executa a cadeia de candidatos com fallback automático, suporte a cache e atualização de sessão."""
    initial_url, initial_model, _, _, initial_label = route_candidates[0]
    log_entry["target_destination"] = initial_url
    log_entry["target_model"] = initial_model
    print(f"[Orquestrador] Encaminhando ({category}) para: {initial_url} | Modelo Inicial: {initial_model} ({initial_label})")

    timeout_config = httpx.Timeout(300.0, connect=30.0)

    if stream:
        async def stream_generator():
            try:
                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    success = False
                    accumulated_chunks = []

                    for tgt_url, mod_name, hdrs, sanitize_ollama, prov_label in route_candidates:
                        if sanitize_ollama:
                            curr_msgs = sanitize_messages_for_ollama(messages, keep_images=False)
                        else:
                            curr_msgs = ensure_alternating_roles(messages)

                        curr_payload = {**payload, "model": mod_name, "messages": curr_msgs}
                        # Garante orçamento suficiente de tokens de saída para evitar cortes prematuros da Cloudflare / LiteLLM
                        req_max_tokens = curr_payload.get("max_tokens")
                        if not req_max_tokens or req_max_tokens < 4096:
                            curr_payload["max_tokens"] = 8192 if category in ["CODING", "HEAVY_REASONING"] else 4096

                        print(f"[Orquestrador {category}] Tentando inferência com '{mod_name}' via {prov_label} (max_tokens: {curr_payload['max_tokens']})...")

                        try:
                            async with client.stream("POST", tgt_url, json=curr_payload, headers=hdrs) as resp:
                                if resp.status_code == 200:
                                    log_entry["target_destination"] = tgt_url
                                    log_entry["target_model"] = mod_name
                                    
                                    if is_debug_allowed_for_user(user_email=user_email, user_id=user_id):
                                        debug_content = format_debug_block(log_entry)
                                        if debug_content:
                                            # Emite como reasoning_content para o LibreChat renderizar nativamente no bloco de Pensamentos
                                            chunk_resp = {
                                                "id": "chatcmpl-debug",
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": "Orchestrator-Debug",
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "role": "assistant",
                                                        "reasoning_content": debug_content
                                                    },
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")

                                    async for chunk in resp.aiter_bytes():
                                        yield chunk
                                        accumulated_chunks.append(chunk)
                                    success = True
                                    break
                                else:
                                    err = await resp.aread()
                                    print(f"[Aviso Inferência {resp.status_code}] Modelo '{mod_name}' ({prov_label}) retornou erro: {err.decode('utf-8', errors='ignore')[:150]}. Tentando próximo...")
                        except Exception as stream_call_err:
                            print(f"[Aviso Exceção Chamada] Modelo '{mod_name}' ({prov_label}) falhou: {stream_call_err}. Tentando próximo...")

                    if not success:
                        log_entry["status"] = "all_backends_failed"
                        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                        write_log(log_entry)
                        msg = json.dumps({"choices": [{"delta": {"content": "\n\n[Erro: Não foi possível obter resposta dos backends de inferência. Por favor, tente novamente.]"}}]})
                        yield f"data: {msg}\n\n".encode("utf-8")
                        yield b"data: [DONE]\n\n"
                        return

                # Gravação em Cache após streaming com sucesso
                accumulated_text = ""
                try:
                    full_raw = b"".join(accumulated_chunks).decode("utf-8", errors="ignore")
                    delta_texts = []
                    for line in full_raw.split("\n"):
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            cjson = json.loads(line[6:])
                            delta = cjson.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if delta:
                                delta_texts.append(str(delta))
                    accumulated_text = "".join(delta_texts)
                    if accumulated_text:
                        if category == "CODING" and not is_code_response_complete(accumulated_text):
                            print(f"[Aviso Cache Coding] Resposta de código parece incompleta (tamanho: {len(accumulated_text)} chars). Gravação em cache ignorada.")
                        else:
                            store_in_cache(cache_lookup_key, accumulated_text, category, log_entry.get("target_model"))
                except Exception as cache_err:
                    print(f"[Aviso Gravação Cache Stream]: {cache_err}")

                # Atualiza memória de sessão conversacional
                if conversation_id:
                    update_conversation_state(
                        conversation_id=conversation_id,
                        category=category,
                        target_model=log_entry.get("target_model", ""),
                        prompt=user_prompt,
                        assistant_snippet=accumulated_text[:300]
                    )

                if accumulated_text:
                    auto_curate_frequent_queries(
                        user_id=user_id,
                        prompt=user_prompt,
                        category=category,
                        response_text=accumulated_text
                    )

                log_entry["status"] = "success"
                log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                write_log(log_entry)

            except httpx.ConnectError as conn_err:
                print(f"[Erro de Conexão] Falha ao conectar a {log_entry.get('target_destination')}: {conn_err}")
                log_entry["status"] = "connection_error"
                log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                write_log(log_entry)
                msg = json.dumps({"choices": [{"delta": {"content": "\n\n[Erro de Conexão: Não foi possível alcançar o servidor de inferência local/nuvem.]"}}]})
                yield f"data: {msg}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except httpx.ReadTimeout:
                print(f"[Erro Timeout] Backend excedeu tempo limite de 300s.")
                log_entry["status"] = "timeout_error"
                log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                write_log(log_entry)
                msg = json.dumps({"choices": [{"delta": {"content": "\n\n[Tempo limite atingido. Por favor, repita a pergunta.]"}}]})
                yield f"data: {msg}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as ex:
                print(f"[Exceção Stream]: {ex}")
                log_entry["status"] = f"stream_error: {str(ex)}"
                log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                write_log(log_entry)
                msg = json.dumps({"choices": [{"delta": {"content": f"\n\n[Erro no Orquestrador: {str(ex)}]"}}]})
                yield f"data: {msg}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    else:
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = None
                resp_json = {}
                for tgt_url, mod_name, hdrs, sanitize_ollama, prov_label in route_candidates:
                    if sanitize_ollama:
                        curr_msgs = sanitize_messages_for_ollama(messages, keep_images=False)
                    else:
                        curr_msgs = ensure_alternating_roles(messages)

                    curr_payload = {**payload, "model": mod_name, "messages": curr_msgs}
                    # Garante orçamento suficiente de tokens de saída para evitar cortes prematuros da Cloudflare / LiteLLM
                    req_max_tokens = curr_payload.get("max_tokens")
                    if not req_max_tokens or req_max_tokens < 4096:
                        curr_payload["max_tokens"] = 8192 if category in ["CODING", "HEAVY_REASONING"] else 4096

                    print(f"[Orquestrador {category}] Tentando inferência com '{mod_name}' via {prov_label} (max_tokens: {curr_payload['max_tokens']})...")

                    try:
                        resp = await client.post(tgt_url, json=curr_payload, headers=hdrs)
                        if resp.status_code == 200:
                            log_entry["target_destination"] = tgt_url
                            log_entry["target_model"] = mod_name
                            resp_json = resp.json()
                            break
                        else:
                            err_txt = resp.text[:150]
                            print(f"[Aviso Inferência {resp.status_code}] Modelo '{mod_name}' ({prov_label}) erro: {err_txt}. Tentando próximo...")
                    except Exception as c_err:
                        print(f"[Aviso Exceção Chamada] Modelo '{mod_name}' ({prov_label}) falhou: {c_err}. Tentando próximo...")

                if resp is None or resp.status_code != 200:
                    status_c = resp.status_code if resp else 500
                    return JSONResponse(status_code=status_c, content={"error": {"message": "Todos os backends de inferência falharam."}})

                ans_text = ""
                try:
                    ans_text = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                    if is_debug_allowed_for_user(user_email=user_email, user_id=user_id):
                        debug_content = format_debug_block(log_entry)
                        if debug_content:
                            try:
                                msg_dict = resp_json["choices"][0]["message"]
                                existing_reasoning = msg_dict.get("reasoning_content") or ""
                                msg_dict["reasoning_content"] = debug_content + existing_reasoning
                            except (KeyError, IndexError):
                                pass

                    if ans_text:
                        if category == "CODING" and not is_code_response_complete(ans_text):
                            print(f"[Aviso Cache Coding] Resposta de código parece incompleta (tamanho: {len(ans_text)} chars). Gravação em cache ignorada.")
                        else:
                            store_in_cache(cache_lookup_key, ans_text, category, log_entry.get("target_model"))
                except Exception as c_err:
                    print(f"[Aviso Gravação Cache JSON]: {c_err}")

                # Atualiza memória de sessão conversacional
                if conversation_id:
                    update_conversation_state(
                        conversation_id=conversation_id,
                        category=category,
                        target_model=log_entry.get("target_model", ""),
                        prompt=user_prompt,
                        assistant_snippet=ans_text[:300]
                    )

                log_entry["status"] = "success"
                log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
                write_log(log_entry)
                return JSONResponse(status_code=200, content=resp_json)
        except httpx.ConnectError:
            log_entry["status"] = "connection_error"
            log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
            write_log(log_entry)
            return JSONResponse(status_code=503, content={"error": {"message": "Não foi possível conectar ao backend."}})
        except httpx.ReadTimeout:
            log_entry["status"] = "timeout_error"
            log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
            write_log(log_entry)
            return JSONResponse(status_code=504, content={"error": {"message": "O backend de inferência excedeu o tempo limite."}})
        except Exception as ex:
            log_entry["status"] = f"exception: {str(ex)}"
            log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
            write_log(log_entry)
            return JSONResponse(status_code=500, content={"error": {"message": str(ex)}})


# ---------------------------------------------------------
# Endpoints Dedicados para CODING / OpenCode / Cline / VSCode
# ---------------------------------------------------------
@app.post("/v1/coding/completions")
@app.post("/v1/opencode/completions")
@app.post("/v1/coding/fast")
@app.post("/v1/opencode/fast")
@app.post("/v1/coding/exec")
@app.post("/v1/opencode/exec")
@app.post("/v1/coding/plan")
@app.post("/v1/opencode/plan")
async def coding_completions(request: Request):
    """
    Endpoint dedicado para ferramentas de desenvolvimento de software (OpenCode, Cline, Continue, VSCode).
    Suporta 3 níveis de engenharia de software:
    - FAST: Scripts rápidos e edições diretas
    - EXEC: Implementação intermediária e refatoração precisa
    - PLAN: Arquitetura avançada e planejamento com raciocínio profundo
    """
    req_start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    path = request.url.path
    payload = await request.json()
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)

    user_id = request.headers.get("x-user-id") or payload.get("user") or f"opencode_{client_ip.replace('.', '_')}"
    conversation_id = request.headers.get("x-conversation-id") or payload.get("conversation_id") or "opencode_session"

    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                user_prompt = " ".join(texts)
            break

    # Se a sub-rota especifica o nível (ex: /fast, /exec, /plan)
    forced_model = payload.get("model", "")
    if path.endswith("/plan"):
        forced_model = "agent-coding-plan"
    elif path.endswith("/exec"):
        forced_model = "agent-coding-exec"
    elif path.endswith("/fast"):
        forced_model = "agent-coding-fast"

    tier = resolve_coding_tier(
        requested_model=forced_model,
        user_prompt=user_prompt,
        messages=messages,
        is_dedicated_endpoint=True
    )

    log_entry = {
        "timestamp": get_now_local().isoformat(),
        "client_ip": client_ip,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "prompt": user_prompt[:300] + ("..." if len(user_prompt) > 300 else ""),
        "category": "CODING",
        "routing_method": f"dedicated_coding_endpoint_{tier.lower()}",
        "coding_tier": tier,
        "stream": stream
    }

    route_candidates = get_coding_candidates(tier=tier)
    enhanced_messages = prepare_coding_messages(messages, tier=tier)

    return await execute_candidates_pipeline(
        route_candidates=route_candidates,
        payload=payload,
        messages=enhanced_messages,
        stream=stream,
        category="CODING",
        cache_lookup_key=user_prompt,
        log_entry=log_entry,
        req_start_time=req_start_time,
        conversation_id=conversation_id,
        user_prompt=user_prompt
    )


# ---------------------------------------------------------
# Endpoint de Descoberta de Modelos (Compatível com OpenAI / IDEs)
# ---------------------------------------------------------
@app.get("/v1/models")
async def list_models():
    """
    Lista de modelos compatível com a especificação da OpenAI.
    Permite que IDEs (OpenCode, Cline, Continue, Cursor) e clientes LibreChat
    descubram os modelos e agentes disponíveis.
    """
    now_ts = int(time.time())
    models = [
        {"id": "agent-orchestrator", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Orquestrador Geral Unificado (LibreChat - Fast Coding)"},
        {"id": "agent-social", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Agente Especialista em Redes Sociais, Leads e Imagens (Instagram, LinkedIn, X, Facebook, Reddit)"},
        {"id": "agent-social-smart", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Agente Social Media SMART (Composição Multimodal OpenAI GPT Image com múltiplas fotos)"},
        {"id": "agent-coding-fast", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Coding Fast: Scripts Rápidos, Edições e Automação (Qwen 32B / DeepSeek V3 / Ollama)"},
        {"id": "agent-coding-exec", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Coding Intermediário / Execução: Implementação de Módulos e Refatoração (DeepSeek V3 671B / Qwen 32B)"},
        {"id": "agent-coding-plan", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Coding Avançado / Planejamento: Arquitetura e Raciocínio Profundo (DeepSeek R1 / o3-mini)"},
        {"id": "opencode-fast", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "OpenCode Fast: Edições Rápidas e Automação"},
        {"id": "opencode-exec", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "OpenCode Exec: Implementação e Refatoração de Código"},
        {"id": "opencode-plan", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "OpenCode Plan: Planejamento Arquitetural & CoT"},
        {"id": "agent-coding", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Alias para Coding Intermediário / Execução"},
        {"id": "agent-coding-pro", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Alias para Coding Avançado / Planejamento"},
        {"id": "opencode", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Alias para OpenCode Execução"},
        {"id": "Combo_Free_Coding", "object": "model", "created": now_ts, "owned_by": "9router", "description": "Modelo 9Router Qwen 2.5 Coder 32B Cloudflare + Groq OSS 120B"},
        {"id": "Coding_DeepSeek_V3", "object": "model", "created": now_ts, "owned_by": "litellm", "description": "Modelo DeepSeek V3 671B MoE Nuvem Rápida"},
        {"id": "Reasoning_DeepSeek_R1", "object": "model", "created": now_ts, "owned_by": "litellm", "description": "Modelo DeepSeek R1 Raciocínio Profundo"},
        {"id": "qwen2.5-coder:7b", "object": "model", "created": now_ts, "owned_by": "ollama", "description": "Modelo Local Ollama Resiliente Offline"},
        {"id": "whisper-1", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Transcrição de Áudio STT em Cascata (Groq Turbo -> Gemini Flash -> OpenAI Whisper)"},
        {"id": "Audio_Transcribe", "object": "model", "created": now_ts, "owned_by": "antigravity-orchestrator", "description": "Alias para Cascata de Reconhecimento de Fala STT"},
    ]
    return {"object": "list", "data": models}


# ---------------------------------------------------------
# Endpoint de Reconhecimento de Fala (STT / Transcrições de Áudio)
# Compatível 100% com OpenAI POST /v1/audio/transcriptions (LibreChat Voice)
# ---------------------------------------------------------
@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0)
):
    """
    Endpoint compatível com OpenAI para transcrição de áudio gravado no LibreChat ou outras ferramentas.
    Aplica cascata com fallback:
      1º Groq Whisper (Zero custo & ultra-rápido)
      2º Gemini Flash Transcribe (Zero custo / Google)
      3º OpenAI Whisper-1 (Fallback oficial resiliente)
    """
    req_start_time = time.time()
    try:
        file_bytes = await file.read()
        filename = file.filename or "audio.wav"

        result = await transcribe_audio_cascade(
            file_bytes=file_bytes,
            filename=filename,
            requested_model=model,
            language=language or "pt",
            prompt=prompt,
            temperature=temperature or 0.0
        )

        text = result.get("text", "")
        model_used = result.get("model_used", model)
        provider = result.get("provider", "cascade")
        total_duration = round(time.time() - req_start_time, 3)

        write_log({
            "timestamp": get_now_local().isoformat(),
            "event": "audio_transcription",
            "filename": filename,
            "bytes_received": len(file_bytes),
            "model_requested": model,
            "model_used": model_used,
            "provider": provider,
            "transcribed_text": text,
            "duration_seconds": total_duration
        })

        if response_format == "text":
            return text
        
        return JSONResponse(status_code=200, content={"text": text})

    except Exception as e:
        print(f"[Erro /v1/audio/transcriptions]: {e}")
        return JSONResponse(status_code=500, content={"error": str(e), "text": ""})


# ---------------------------------------------------------
# Endpoint de Síntese de Voz (TTS - Text to Speech)
# Compatível 100% com OpenAI POST /v1/audio/speech (LibreChat Read Aloud)
# ---------------------------------------------------------
@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """
    Endpoint compatível com OpenAI para síntese de voz ultra-realista.
    Consumido nativamente pelo LibreChat no botão 'Ouvir / Read Aloud'.
    """
    req_start_time = time.time()
    try:
        payload = await request.json()
        input_text = payload.get("input", "")
        model = payload.get("model", "tts-1")
        voice = payload.get("voice", "nova")
        response_format = payload.get("response_format", "mp3")
        speed = payload.get("speed", 1.0)

        if not input_text:
            return JSONResponse(status_code=400, content={"error": "Campo 'input' é obrigatório."})

        # Limita para 4096 caracteres (padrão de segurança de TTS)
        text_to_speak = input_text[:4096]

        success, audio_bytes, content_type_or_err = await synthesize_speech_openai(
            text=text_to_speak,
            voice=voice,
            model=model,
            response_format=response_format,
            speed=speed
        )

        total_duration = round(time.time() - req_start_time, 3)

        if success:
            write_log({
                "timestamp": get_now_local().isoformat(),
                "event": "audio_speech_tts",
                "voice": voice,
                "model": model,
                "input_chars": len(text_to_speak),
                "audio_bytes": len(audio_bytes),
                "duration_seconds": total_duration,
                "status": "success"
            })
            return Response(
                content=audio_bytes,
                media_type=content_type_or_err,
                headers={
                    "Content-Disposition": f"inline; filename=speech.{response_format}",
                    "Cache-Control": "public, max-age=86400"
                }
            )
        else:
            write_log({
                "timestamp": get_now_local().isoformat(),
                "event": "audio_speech_tts",
                "voice": voice,
                "model": model,
                "error": content_type_or_err,
                "duration_seconds": total_duration,
                "status": "error"
            })
            return JSONResponse(status_code=500, content={"error": content_type_or_err})

    except Exception as e:
        print(f"[Erro /v1/audio/speech]: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---------------------------------------------------------
# Endpoint de Autocompletação / FIM (Fill-In-The-Middle)
# ---------------------------------------------------------
@app.post("/v1/completions")
async def completions(request: Request):
    """
    Endpoint para Autocompletação de Código / FIM (Fill-in-the-middle) usado por IDEs.
    Tenta primeiro o modelo local qwen2.5-coder:7b (que possui suporte FIM nativo via Ollama).
    Caso indisponível, adapta o prompt para completion via Combo_Free_Coding.
    """
    req_start_time = time.time()
    payload = await request.json()
    prompt = payload.get("prompt", "")
    suffix = payload.get("suffix", "")
    stream = payload.get("stream", False)
    max_tokens = payload.get("max_tokens") or 128
    temperature = payload.get("temperature", 0.2)

    # Constrói prompt FIM se houver suffix
    full_prompt = prompt
    if suffix and "<|fim_suffix|>" not in prompt:
        full_prompt = f"<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>"

    # 1. Tentar Ollama Local nativo com FIM
    ollama_payload = {
        "model": "qwen2.5-coder:7b",
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/v1/completions", json=ollama_payload)
            if resp.status_code == 200:
                return JSONResponse(status_code=200, content=resp.json())
    except Exception as e:
        print(f"[Aviso FIM Ollama]: {e}")

    # 2. Fallback via Chat Completion adaptado (Combo_Free_Coding)
    adapted_messages = [
        {"role": "system", "content": "Você é um motor de autocompletação de código. Retorne ESTRITAMENTE o código que preenche a posição indicada, sem explicações em linguagem natural, sem introdução e sem delimitadores de markdown."},
        {"role": "user", "content": f"Complete o código:\nPrefixo:\n{prompt}\nSufixo:\n{suffix}\nPreenchimento:" if suffix else f"Complete o código a seguir:\n{prompt}"}
    ]
    chat_payload = {
        "model": "Combo_Free_Coding",
        "messages": adapted_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{LITELLM_BASE_URL}/chat/completions",
                json=chat_payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {LITELLM_API_KEY}"}
            )
            if resp.status_code == 200:
                c_json = resp.json()
                text_out = c_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                return JSONResponse(status_code=200, content={
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": "agent-coding-fim",
                    "choices": [{"text": text_out, "index": 0, "finish_reason": "stop"}]
                })
    except Exception as e:
        print(f"[Aviso FIM Chat Adapt]: {e}")

    return JSONResponse(status_code=500, content={"error": {"message": "Falha na autocompletação de código."}})


# ---------------------------------------------------------
# Endpoints Dedicados de Redes Sociais e Geração de Leads
# ---------------------------------------------------------
@app.post("/v1/social/create-post")
async def create_social_post(request: Request):
    """
    Endpoint dedicado para automações (n8n, Make, scripts, bots) criarem postagens
    completas de redes sociais (copywriting + direção de arte + imagem contextualizada).
    """
    req_start_time = time.time()
    try:
        data = await request.json()
        prompt = data.get("prompt") or data.get("text") or data.get("topic")
        if not prompt:
            return JSONResponse(status_code=400, content={"error": "Campo 'prompt' é obrigatório."})

        platform = data.get("platform")
        user_msg = {"role": "user", "content": prompt}
        output_text, tools_called = await execute_social_post_flow(
            user_prompt=prompt,
            last_user_msg=user_msg,
            messages=[user_msg],
            platform_override=platform.upper() if platform else None
        )
        return {
            "status": "success",
            "platform": platform or "auto_detected",
            "markdown": output_text,
            "tools_called": tools_called,
            "duration_seconds": round(time.time() - req_start_time, 3)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/v1/social/completions")
async def social_completions(request: Request):
    """
    Alias OpenAI-compatible direcionado para o Agente de Redes Sociais e Leads.
    """
    payload = await request.json()
    payload["model"] = "agent-social"
    # Redireciona internamente para o pipeline principal
    return await chat_completions(request)


def format_debug_block(log_entry: dict) -> str:
    """Gera um bloco formatado para o painel de Pensamentos (Reasoning) do LibreChat."""
    details = (
        f"🛠️ [Orquestrador Debug - Rota: {log_entry.get('category', 'Desconhecida')}]\n"
        f"• Decisão: {log_entry.get('routing_method', 'N/A')}\n"
        f"• Alvo: {log_entry.get('target_model', 'N/A')}\n"
    )
    
    raw_class = log_entry.get('raw_classifier_output')
    if raw_class:
        details += f"• Classificador: {str(raw_class).strip()}\n"
        
    tools = log_entry.get('tools_called', [])
    if tools:
        tool_names = ", ".join([f"{t.get('tool') or t.get('action') or 'Tool'} ({t.get('duration_seconds') or t.get('duration_sec', 0)}s)" for t in tools])
        details += f"• Ferramentas: {tool_names}\n"

        # Destaque de auditoria factual se presente
        audit_entries = [t for t in tools if "fact_audit" in str(t.get("action", ""))]
        for ae in audit_entries:
            st = ae.get("status", "N/A")
            hal = "ALUCINAÇÃO CORRIGIDA" if ae.get("hallucination_detected") else "VERIFICADO"
            score = ae.get("confidence_score", 1.0)
            details += f"• Auditor Anti-Alucinação: [{st} / {hal}] Score: {score} ({ae.get('notes', '')})\n"
        
    details += "─" * 40 + "\n\n"
    return details


# ---------------------------------------------------------
# Endpoint Principal OpenAI-Compatible
# ---------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    req_start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    payload = await request.json()
    messages = payload.get("messages", [])
    stream = payload.get("stream", False)

    # Identidade do Usuário e Sessão
    user_id = (
        request.headers.get("x-user-id")
        or payload.get("user")
        or request.headers.get("x-user-email")
        or f"user_{client_ip.replace('.', '_')}"
    )
    user_email = request.headers.get("x-user-email") or ""
    conversation_id = (
        request.headers.get("x-conversation-id")
        or payload.get("conversation_id")
        or "default_session"
    )

    # 1. Localizar última mensagem enviada pelo USUÁRIO
    last_user_msg = {}
    user_prompt = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg
            content = msg.get("content", "")
            if isinstance(content, str):
                user_prompt = content
            elif isinstance(content, list):
                texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
                user_prompt = " ".join(texts)
            break

    # Checagens visuais
    has_image_now = message_has_image(last_user_msg)
    has_image_in_history = history_has_images(messages)

    # Auto-indexação na Memória Temporária caso envie texto/documento longo
    auto_ingest_log = None
    if len(user_prompt) >= 1200:
        first_line = user_prompt.strip().split("\n")[0][:60]
        clean_title = re.sub(r"[#*_`]", "", first_line).strip() or "Texto Longo Fornecido"
        print(f"[Orquestrador Memória] Documento longo detectado ({len(user_prompt)} chars). Salvando na memória de '{user_id}'...")
        auto_ingest_log = ingest_user_text_memory(
            text=user_prompt,
            user_id=user_id,
            conversation_id=conversation_id,
            doc_title=clean_title,
            ttl_days=30
        )

    # 2. Recupera estado prévio da conversa para continuidade contextual
    session_state = get_conversation_state(conversation_id, messages)
    last_context_prompt = session_state.get("last_prompt", "")
    is_followup = is_context_dependent_prompt(user_prompt, messages)

    # 3. Classificação de Intenção via Módulo classifier (com memória de sessão) ou Alias Direto de IDE/Agente
    req_model = payload.get("model", "")
    is_dedicated_coding_alias = any(
        k in req_model.lower() for k in [
            "agent-coding", "opencode", "coding-fast", "coding-exec", "coding-plan"
        ]
    )
    is_dedicated_social_alias = any(
        k in req_model.lower() for k in [
            "agent-social", "social-media", "agent-marketing", "social-post"
        ]
    )
    is_dedicated_export_alias = any(
        k in req_model.lower() for k in [
            "agent-data", "data-export", "export-excel", "chart-generator", "bi-chart"
        ]
    )
    rewritten_prompt = None
    if is_dedicated_coding_alias:
        category = "CODING"
        routing_method = "direct_model_alias"
        raw_classifier_output = f"Direct alias: {req_model}"
    elif is_dedicated_social_alias:
        category = "SOCIAL_MEDIA_POST"
        routing_method = "direct_model_alias"
        raw_classifier_output = f"Direct alias: {req_model}"
    elif is_dedicated_export_alias:
        category = "DATA_EXPORT_CHART"
        routing_method = "direct_model_alias"
        raw_classifier_output = f"Direct alias: {req_model}"
    else:
        category, routing_method, raw_classifier_output, rewritten_prompt = await classify_intent(
            user_prompt=user_prompt,
            messages=messages,
            last_user_msg=last_user_msg,
            conversation_id=conversation_id
        )

    # Se o classificador detectou que era frase incompleta e reescreveu o prompt com o tema anterior
    if rewritten_prompt and rewritten_prompt.strip() and rewritten_prompt.strip() != user_prompt.strip():
        print(f"[Orquestrador Query Rewriter] Prompt incompleto reescrito com tema anterior: '{user_prompt}' -> '{rewritten_prompt}'")
        is_followup = True
        # Atualiza a mensagem atual no payload de mensagens para que os modelos downstream recebam o contexto pleno
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
            orig_c = messages[-1].get("content")
            if isinstance(orig_c, str):
                messages[-1]["content"] = rewritten_prompt
            elif isinstance(orig_c, list):
                for part in messages[-1]["content"]:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = rewritten_prompt
                        break

    print(f"[Orquestrador] Pergunta: '{user_prompt[:50]}' -> Rota: {category} (Usuário: {user_id}) | Contextual: {is_followup} | Método: {routing_method}")

    # Registro de Log
    log_entry = {
        "timestamp": get_now_local().isoformat(),
        "client_ip": client_ip,
        "user_id": user_id,
        "user_email": user_email,
        "conversation_id": conversation_id,
        "prompt": user_prompt[:300] + ("..." if len(user_prompt) > 300 else ""),
        "has_image": has_image_now or has_image_in_history,
        "is_followup": is_followup,
        "routing_method": routing_method,
        "raw_classifier_output": raw_classifier_output,
        "category": category,
        "stream": stream,
        "auto_ingest_memory": auto_ingest_log
    }

    # ---------------------------------------------------------
    # Camada de Cache Híbrido (SQLite Exato + Qdrant Semântico)
    # ---------------------------------------------------------
    cache_lookup_key = user_prompt
    if is_followup:
        last_asst_text = next((m.get("content", "") for m in reversed(messages[:-1]) if m.get("role") == "assistant"), "")
        asst_snip = (last_asst_text[:150] if isinstance(last_asst_text, str) else str(last_asst_text)[:150]).strip()
        cache_lookup_key = f"[Ctx: {asst_snip}] {user_prompt}"

    cached_hit = None
    if category not in ["IMAGE_GENERATION", "IMAGE_EDIT", "SOCIAL_MEDIA_POST", "DATA_EXPORT_CHART", "SHORT_VIDEO"]:
        cached_hit = await lookup_cache(cache_lookup_key, category=category, is_followup=is_followup)

    if cached_hit:
        cache_type = cached_hit.get("type", "cache")
        similarity = cached_hit.get("similarity", 1.0)
        cached_response = cached_hit["response"]
        cached_model = cached_hit.get("target_model", "Cache")
        print(f"[Orquestrador CACHE HIT] Tipo: {cache_type} (Similaridade: {similarity}) | Respondendo instantaneamente...")

        log_entry["target_destination"] = cache_type
        log_entry["target_model"] = f"{cached_model} ({cache_type})"
        log_entry["cache_hit"] = True
        log_entry["cache_type"] = cache_type
        log_entry["similarity"] = similarity
        log_entry["status"] = "success"
        log_entry["response_preview"] = cached_response[:250]
        log_entry["duration_seconds"] = round(time.time() - req_start_time, 4)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model=cached_model,
            prompt=user_prompt,
            assistant_snippet=cached_response[:300]
        )

        user_debug_allowed = is_debug_allowed_for_user(user_email=user_email, user_id=user_id)
        debug_reasoning = format_debug_block(log_entry) if user_debug_allowed else ""

        if stream:
            async def cache_stream():
                if debug_reasoning:
                    debug_chunk = {
                        "id": f"chatcmpl-{cache_type}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": cached_model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "reasoning_content": debug_reasoning},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(debug_chunk)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": f"chatcmpl-{cache_type}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cached_model,
                    "choices": [{"index": 0, "delta": {"content": cached_response}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": f"chatcmpl-{cache_type}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": cached_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(cache_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": cached_response}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": f"chatcmpl-{cache_type}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": cached_model,
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # Auto-Scraping de URLs e Domínios (Web Crawler / Crawl4AI)
    # ---------------------------------------------------------
    url_pattern = r'(?:https?://[^\s<>"\')]+|(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|io|ai|ia|dev|app|br|tech|me|co|info|edu|gov)(?:/[^\s<>"\')]*|\b))'
    raw_urls_in_prompt = re.findall(url_pattern, user_prompt, re.IGNORECASE)
    urls_in_prompt = []
    for raw_u in raw_urls_in_prompt:
        clean_u = raw_u.rstrip(".,;:?!)]")
        if not clean_u.startswith(("http://", "https://")):
            clean_u = "https://" + clean_u
        if clean_u not in urls_in_prompt:
            urls_in_prompt.append(clean_u)

    if urls_in_prompt and category not in ["IMAGE_GENERATION", "IMAGE_EDIT"]:
        try:
            from scraper_service import scrape_url_to_markdown
            from rag_service import ingest_user_text_memory
            print(f"[Orquestrador] Detectadas {len(urls_in_prompt)} URLs/Domínios ({urls_in_prompt}). Iniciando scraping...")
            scraped_blocks = []
            for u in urls_in_prompt[:3]: # Limita a 3 URLs para não explodir tempo
                md = await scrape_url_to_markdown(u)
                if md and not md.startswith("[Erro"):
                    scraped_blocks.append(f"--- Conteúdo de {u} ---\n{md}\n")
                    # Salva na memória do usuário em background para perguntas de acompanhamento
                    asyncio.create_task(asyncio.to_thread(
                        ingest_user_text_memory,
                        text=f"Conteúdo de URL extraída pelo usuário: {u}\n\n{md[:6000]}",
                        user_id=user_id,
                        conversation_id=conversation_id,
                        doc_title=f"URL: {u}",
                        ttl_days=1
                    ))
            
            if scraped_blocks:
                extra_ctx = "\n\n[CONTEXTO EXTRAÍDO DOS LINKS FORNECIDOS]\n" + "\n".join(scraped_blocks)
                user_prompt += extra_ctx
                if isinstance(messages[-1].get("content"), str):
                    messages[-1]["content"] += extra_ctx
                elif isinstance(messages[-1].get("content"), list):
                    messages[-1]["content"].append({"type": "text", "text": extra_ctx})
        except Exception as e:
            print(f"[Aviso Auto-Scraping]: Erro ao tentar processar URLs: {e}")

    # ---------------------------------------------------------
    # Memória de Feedback Negativo / Anti-Exemplos (Thumbs Down)
    # ---------------------------------------------------------
    feedback_guardrail = await check_negative_feedback_guardrail(user_prompt)
    if feedback_guardrail:
        print(f"[Orquestrador Guardrail] Pergunta similar a erro anterior detectada! Injetando diretriz corretiva.")
        log_entry["feedback_guardrail_applied"] = True
        user_prompt += feedback_guardrail
        if isinstance(messages[-1].get("content"), str):
            messages[-1]["content"] += feedback_guardrail
        elif isinstance(messages[-1].get("content"), list):
            messages[-1]["content"].append({"type": "text", "text": feedback_guardrail})

    # ---------------------------------------------------------
    # Memória Episódica Persistente & Perfil Analítico do Usuário
    # ---------------------------------------------------------
    episodic_context = ""
    if user_id and user_id not in ["anon", "default_user", "system_news_bot"]:
        # 1. Atualiza o perfil comportamental em segundo plano
        update_user_behavior_profile(
            user_id=user_id,
            user_email=user_email,
            category=category,
            prompt=user_prompt,
            target_model=payload.get("model", "")
        )

        # 2. Extrai e armazena fatos biográficos / preferências explícitas (ex: 'meu nome é', 'eu moro em')
        bio_facts = detect_and_extract_biographic_facts(user_prompt)
        for bf in bio_facts:
            store_episodic_memory(
                user_id=user_id,
                content=bf["content"],
                memory_kind=bf["kind"],
                importance=bf["importance"],
                source_title="Declaração Direta do Usuário"
            )

        # 3. Consulta memórias episódicas persistentes pertinentes à consulta
        user_prof = get_user_profile(user_id)
        retrieved_memories = await search_episodic_memories(
            query=user_prompt,
            user_id=user_id,
            k=3,
            min_score=0.55
        )

        context_parts = []
        if user_prof:
            arch = user_prof.get("detected_archetype", "Geral")
            style = user_prof.get("preferences", {}).get("style", "")
            context_parts.append(f"- Perfil Comportamental Detectado: {arch} (Estilo ideal de resposta: {style})")

        if retrieved_memories:
            for rm in retrieved_memories:
                context_parts.append(f"- Memória Permanente ({rm.get('kind')}): {rm.get('content')}")

        if context_parts:
            episodic_context = (
                "\n\n[CONTEXTO & PREFERÊNCIAS PERMANENTES DO USUÁRIO]:\n" +
                "\n".join(context_parts) +
                "\n(Utilize estas preferências e memórias de longo prazo para enriquecer sua resposta de forma personalizada e coerente)\n"
            )
            print(f"[Orquestrador Episódico] Injetadas {len(context_parts)} diretrizes/memórias de longo prazo para '{user_id}'.")
            log_entry["episodic_memories_injected"] = len(retrieved_memories)
            user_prompt += episodic_context
            if isinstance(messages[-1].get("content"), str):
                messages[-1]["content"] += episodic_context
            elif isinstance(messages[-1].get("content"), list):
                messages[-1]["content"].append({"type": "text", "text": episodic_context})


    # ---------------------------------------------------------
    # ROTA ESPECIAL 0: Geração e Edição de Imagens Multi-Provedor
    # ---------------------------------------------------------
    if category in ["IMAGE_GENERATION", "IMAGE_EDIT"]:
        print(f"[Orquestrador] Executando Fluxo de Imagem Multi-Provedor ({category})...")
        log_entry["target_destination"] = "image_flow_executor"
        log_entry["target_model"] = "MultiProvider_ImageGen"

        output_text, tools_called_log = await execute_image_flow(category, user_prompt, last_user_msg, messages)
        log_entry["tools_called"] = tools_called_log
        log_entry["status"] = "success"
        log_entry["response_preview"] = output_text[:250]
        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="MultiProvider_ImageGen",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def image_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": f"chatcmpl-img-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "ImageOrchestrator",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": f"chatcmpl-img-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ImageOrchestrator",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": f"chatcmpl-img-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ImageOrchestrator",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(image_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": f"chatcmpl-img-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "ImageOrchestrator",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # ROTA ESPECIAL 0.5: Postagens Estratégicas para Redes Sociais & Leads
    # ---------------------------------------------------------
    if category == "SOCIAL_MEDIA_POST":
        print(f"[Orquestrador] Executando Fluxo de Redes Sociais & Leads ({category})...")
        
        # Heurística de Decisão de Modo: FAST vs SMART
        # Contagem de imagens presentes na mensagem atual ou histórico
        all_imgs = extract_all_images(messages, last_user_msg)
        num_images = len(all_imgs)
        
        prompt_lower = user_prompt.lower()
        # Gatilhos explícitos que exigem síntese multimodal avançada (OpenAI GPT Image / SMART)
        smart_keywords = [
            "smart", "modo smart", "fundo", "criar fundo", "mudar fundo", "trocar fundo",
            "fonte especial", "fontes especiais", "tipografia", "design avançado",
            "componha", "composição", "junte as fotos", "junte as imagens", "combinar fotos",
            "cartaz", "poster", "banner complexo", "efeito visual", "estilização", "estilizada"
        ]
        has_smart_instruction = any(k in prompt_lower for k in smart_keywords)

        # Gatilhos de reedição / refinamento contextual após geração inicial
        # Ex: "refaça esse post com o texto mais bonito", "melhore a imagem", "edite a imagem", etc.
        refinement_keywords = [
            "refaça", "refazer", "recrie", "recriar", "edite", "editar",
            "melhore", "melhorar", "mude", "mudar", "altere", "alterar",
            "troque", "trocar", "ajuste", "ajustar", "reedite", "reeditar",
            "mais bonito", "mais bonita", "outro texto", "mude o texto"
        ]
        is_refinement = any(k in prompt_lower for k in refinement_keywords)
        prev_was_social = session_state.get("last_category") == "SOCIAL_MEDIA_POST" or is_followup
        is_post_reedition = prev_was_social and (is_refinement or num_images >= 1)
        
        req_model_lower = str(req_model).lower() if req_model else ""
        # Regra de negócio:
        # - Se modelo requisitado tiver "smart": SMART
        # - Se houver mais de 1 foto (ex: 2 ou mais fotos para compor): SMART
        # - Se o usuário pedir regras de design avançadas (fundo, fontes, composição): SMART
        # - Se for reedição/refinamento de um post anterior: SMART automaticamente (preserva contexto e usa IA visual)
        # - Caso contrário (sem regras específicas, geração inicial ou 1 foto simples): FAST (sem custo de OpenAI)
        if "smart" in req_model_lower or num_images > 1 or has_smart_instruction or (prev_was_social and is_refinement):
            selected_mode = "SMART"
        else:
            selected_mode = "FAST"
            
        print(f"[Orquestrador] Modo Social Selecionado: {selected_mode} (Imagens detectadas: {num_images}, Gatilho Avançado: {has_smart_instruction}, Reedição Contextual: {is_post_reedition})")
                
        # 3. Prosseguir com a geração no modo escolhido
        log_entry["target_destination"] = "social_flow_executor"
        log_entry["target_model"] = f"SocialMediaOrchestrator_{selected_mode}"

        output_text, tools_called_log = await execute_social_post_flow(
            user_prompt=user_prompt,
            last_user_msg=last_user_msg,
            messages=messages,
            mode=selected_mode
        )
        log_entry["tools_called"] = tools_called_log
        log_entry["status"] = "success"
        log_entry["response_preview"] = output_text[:250]
        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="SocialMediaOrchestrator",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def social_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": f"chatcmpl-soc-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "SocialMediaOrchestrator",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": f"chatcmpl-soc-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "SocialMediaOrchestrator",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": f"chatcmpl-soc-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "SocialMediaOrchestrator",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(social_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": f"chatcmpl-soc-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "SocialMediaOrchestrator",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # ROTA ESPECIAL 0.6: Exportador de Dados, Planilhas (CSV/Excel) e Gráficos (Seaborn)
    # ---------------------------------------------------------
    if category == "DATA_EXPORT_CHART":
        print(f"[Orquestrador] Executando Fluxo de Exportação de Dados & Gráficos ({category})...")
        log_entry["target_destination"] = "data_export_executor"
        log_entry["target_model"] = "DataExportOrchestrator"

        context_data = ""
        if is_followup:
            context_data = next((m.get("content", "") for m in reversed(messages[:-1]) if m.get("role") in ["assistant", "user"]), "")

        # Se a solicitação mencionar moedas, indicadores ou entidades reais, consulta a web/APIs antes
        realtime_check_words = [
            "dólar", "dolar", "euro", "bitcoin", "btc", "moeda", "cotação", "cotacao",
            "selic", "ipca", "inflação", "inflacao", "cdi", "ações", "acoes", "bolsa",
            "preço", "preco", "2026", "2025", "2024", "hoje", "ontem", "semana", "mês", "mes"
        ]
        needs_real_data = any(w in user_prompt.lower() for w in realtime_check_words)
        gathered_context = str(context_data)

        if needs_real_data:
            print(f"[DataExport Route] Consulta de dados do mundo real detectada. Buscando fontes factuais antes de gerar planilha...")
            search_text, search_tools = await execute_agent_flow(
                category="REALTIME_SEARCH",
                user_prompt=user_prompt,
                context_prompt=str(context_data),
                user_id=user_id
            )
            # Se a própria busca já tiver gerado o gráfico com os dados oficiais (via gancho integrado)
            if "![Gráfico Gerado]" in search_text:
                output_text = search_text
                tools_called_log = search_tools
            else:
                gathered_context = f"{gathered_context}\n\nFONTES FACTUAIS COLETADAS:\n{search_text}"
                output_text, tools_called_log = await execute_data_export_flow(
                    user_prompt=user_prompt,
                    context_text=gathered_context
                )
                tools_called_log = search_tools + tools_called_log
        else:
            output_text, tools_called_log = await execute_data_export_flow(
                user_prompt=user_prompt,
                context_text=gathered_context
            )
        log_entry["tools_called"] = tools_called_log
        log_entry["status"] = "success"
        log_entry["response_preview"] = output_text[:250]
        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="DataExportOrchestrator",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def data_export_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": f"chatcmpl-data-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "DataExportOrchestrator",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": f"chatcmpl-data-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "DataExportOrchestrator",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": f"chatcmpl-data-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "DataExportOrchestrator",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(data_export_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": f"chatcmpl-data-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "DataExportOrchestrator",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # ROTA ESPECIAL 0.7: Gerador de Vídeos Curtos Verticais (Reels / Shorts / TikTok)
    # ---------------------------------------------------------
    if category == "SHORT_VIDEO":
        print(f"[Orquestrador] Executando Fluxo de Vídeos Curtos ({category})...")
        log_entry["target_destination"] = "short_video_engine"
        log_entry["target_model"] = "ShortVideoOrchestrator"

        output_text, tools_called_log = await execute_short_video_flow(user_prompt=user_prompt)

        log_entry["tools_called"] = tools_called_log
        log_entry["status"] = "success"
        log_entry["response_preview"] = output_text[:250]
        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="ShortVideoOrchestrator",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def short_video_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": f"chatcmpl-vid-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "ShortVideoOrchestrator",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": f"chatcmpl-vid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ShortVideoOrchestrator",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": f"chatcmpl-vid-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "ShortVideoOrchestrator",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(short_video_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": f"chatcmpl-vid-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "ShortVideoOrchestrator",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # ROTA ESPECIAL 1: RAG Unificado (Documentos + Memória)
    # ---------------------------------------------------------
    if category == "DOCUMENT_RAG":
        print(f"[Orquestrador] Executando Busca Unificada Qdrant para: {category} (user_id={user_id})")
        log_entry["target_destination"] = "qdrant_vector_store"
        log_entry["target_model"] = "ComboFree_Smart (RAG Synthesis)"

        try:
            output_text, tools_called_log = await execute_rag_flow(user_prompt, context_prompt=last_context_prompt if is_followup else "", user_id=user_id)
            log_entry["tools_called"] = tools_called_log
            log_entry["status"] = "success"
            log_entry["response_preview"] = output_text[:250]
        except Exception as ex:
            print(f"[Erro no Fluxo RAG]: {ex}")
            output_text = f"Ocorreu um erro ao consultar a base de documentos: {str(ex)}"
            log_entry["status"] = f"error: {str(ex)}"
            log_entry["tools_called"] = []

        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="ComboFree_Smart (RAG Synthesis)",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        if log_entry.get("status") == "success":
            auto_curate_frequent_queries(
                user_id=user_id,
                prompt=user_prompt,
                category=category,
                response_text=output_text
            )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def rag_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": "chatcmpl-rag",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "KnowledgeBase-RAG",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": "chatcmpl-rag",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "KnowledgeBase-RAG",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": "chatcmpl-rag",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "KnowledgeBase-RAG",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(rag_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": "chatcmpl-rag",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "KnowledgeBase-RAG",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # ROTA ESPECIAL 2: Agente com Ferramentas (Tavily / Weather)
    # ---------------------------------------------------------
    if category in ["REALTIME_SEARCH", "WEATHER_INFO"] and (weather_tool or tavily_tool):
        print(f"[Orquestrador] Executando Agente com Ferramentas para Rota: {category}")
        log_entry["target_destination"] = "agent_executor"
        log_entry["target_model"] = "ComboFree_Smart (Tool Synthesis)"
        
        try:
            # Contextualiza a consulta de ferramentas caso seja uma pergunta de continuação (ex: 'E para Senador?')
            output_text, tools_called_log = await execute_agent_flow(
                category=category,
                user_prompt=user_prompt,
                context_prompt=last_context_prompt if is_followup else "",
                user_id=user_id
            )
            log_entry["tools_called"] = tools_called_log
            log_entry["status"] = "success"
            log_entry["response_preview"] = output_text[:250]
        except Exception as ex:
            print(f"[Erro no Fluxo do Agente]: {ex}")
            output_text = f"Ocorreu um erro ao consultar as ferramentas externas: {str(ex)}"
            log_entry["status"] = f"error: {str(ex)}"
            log_entry["tools_called"] = []

        log_entry["duration_seconds"] = round(time.time() - req_start_time, 3)
        write_log(log_entry)

        update_conversation_state(
            conversation_id=conversation_id,
            category=category,
            target_model="ComboFree_Smart (Tool Synthesis)",
            prompt=user_prompt,
            assistant_snippet=output_text[:300]
        )

        # Abastece automaticamente a memória de pesquisas recorrentes para o perfil do usuário
        if log_entry.get("status") == "success":
            auto_curate_frequent_queries(
                user_id=user_id,
                prompt=user_prompt,
                category=category,
                response_text=output_text
            )

        debug_reasoning = format_debug_block(log_entry) if is_debug_allowed_for_user(user_email=user_email, user_id=user_id) else None

        if stream:
            async def agent_stream():
                if debug_reasoning:
                    chunk_dbg = {
                        "id": "chatcmpl-agent",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": "SmartAgent-Tools",
                        "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": debug_reasoning}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_dbg)}\n\n".encode("utf-8")

                chunk_resp = {
                    "id": "chatcmpl-agent",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "SmartAgent-Tools",
                    "choices": [{"index": 0, "delta": {"content": output_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk_resp)}\n\n".encode("utf-8")
                done_resp = {
                    "id": "chatcmpl-agent",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "SmartAgent-Tools",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_resp)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(agent_stream(), media_type="text/event-stream")
        else:
            msg_payload = {"role": "assistant", "content": output_text}
            if debug_reasoning:
                msg_payload["reasoning_content"] = debug_reasoning
            return JSONResponse(content={
                "id": "chatcmpl-agent",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "SmartAgent-Tools",
                "choices": [{"index": 0, "message": msg_payload, "finish_reason": "stop"}]
            })

    # ---------------------------------------------------------
    # Seleção de Destinos via Serviços Modulares
    # ---------------------------------------------------------
    if category == "VISION":
        route_candidates = [
            (f"{LITELLM_BASE_URL}/chat/completions", "ComboVision", {"Content-Type": "application/json", "Authorization": f"Bearer {LITELLM_API_KEY}"}, False, "LiteLLM (ComboVision)")
        ]
    elif "CASUAL_CHAT" in category:
        route_candidates = get_chat_candidates()
    elif "CODING" in category:
        req_model = payload.get("model", "")
        # Para a interface do LibreChat (agent-orchestrator), resolve SEMPRE como FAST!
        tier = resolve_coding_tier(
            requested_model=req_model,
            user_prompt=user_prompt,
            messages=messages,
            is_dedicated_endpoint=is_dedicated_coding_alias
        )
        log_entry["coding_tier"] = tier
        route_candidates = get_coding_candidates(tier=tier)
        messages = prepare_coding_messages(messages, tier=tier)
    elif "LOCAL_THINKING" in category:
        route_candidates = get_local_thinking_candidates()
    elif "HEAVY_REASONING" in category:
        route_candidates = get_heavy_reasoning_candidates(user_prompt)
    else: # FACTUAL_GENERAL e demais
        route_candidates = get_factual_candidates()

    return await execute_candidates_pipeline(
        route_candidates=route_candidates,
        payload=payload,
        messages=messages,
        stream=stream,
        category=category,
        cache_lookup_key=cache_lookup_key,
        log_entry=log_entry,
        req_start_time=req_start_time,
        conversation_id=conversation_id,
        user_prompt=user_prompt,
        user_email=user_email,
        user_id=user_id
    )
