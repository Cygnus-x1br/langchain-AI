import os
import re
import time
from typing import Tuple, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun

from config import LITELLM_BASE_URL, LITELLM_API_KEY, SEARXNG_URL, clean_location_query
from prompts import build_agent_synthesis_prompt, build_rag_synthesis_prompt, FACT_AUDITOR_PROMPT
from rag_service import search_unified_rag
import httpx
import json

import asyncio
from scraper_service import scrape_url_to_markdown
from rag_service import ingest_user_text_memory

# Ferramentas do Agente
tavily_tool = None
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    tavily_tool = TavilySearchResults(max_results=3)

weather_tool = None
openweather_key = os.getenv("OPENWEATHERMAP_API_KEY")
if openweather_key:
    weather_wrapper = OpenWeatherMapAPIWrapper()
    weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

# Modelo para Síntese do Agente e RAG
agent_llm = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model="ComboFree_Smart",
    temperature=0.3,
    max_tokens=4096
)

# Modelos para Auditoria Reflexiva Anti-Alucinação com cache dinâmico
_auditor_llms: Dict[str, Any] = {}

def get_auditor_llm(model_name: str):
    """Retorna instância de LLM para o auditor de fatos (com cache)."""
    if model_name not in _auditor_llms:
        _auditor_llms[model_name] = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model=model_name,
            temperature=0.0,
            max_tokens=2048,
            timeout=10.0
        )
    return _auditor_llms[model_name]

auditor_llm = get_auditor_llm("Classifier_Fast")
auditor_llm_fallback = get_auditor_llm("Combo_Free_Chat")

async def audit_response_facts(
    user_prompt: str,
    preliminary_text: str,
    sources_text: str,
    flow_name: str = "RAG"
) -> Tuple[str, Dict[str, Any]]:
    """
    Audita reflexivamente a resposta preliminar contra as fontes factuais fornecidas.
    Se detectar alucinação ou contradição, substitui pelo texto revisado.
    """
    t_audit_start = time.time()
    audit_log = {
        "action": f"{flow_name.lower()}_fact_audit",
        "model": "Classifier_Fast",
        "status": "skipped",
        "hallucination_detected": False,
        "confidence_score": 1.0,
        "notes": "Auditoria dispensada por ausência de dados de fonte.",
        "duration_seconds": 0.0
    }

    # Limiar configurável de tamanho mínimo de fontes
    try:
        from admin_config_service import get_guardrail_value, get_prompt_text
        min_src_len = int(get_guardrail_value("fact_auditor_min_sources_length", 30))
        auditor_timeout = float(get_guardrail_value("fact_auditor_timeout_seconds", 8.0))
        auditor_template = get_prompt_text("FACT_AUDITOR_PROMPT") or FACT_AUDITOR_PROMPT
    except Exception:
        min_src_len = 30
        auditor_timeout = 8.0
        auditor_template = FACT_AUDITOR_PROMPT

    # Se não houver fontes ou resposta, mantém a resposta preliminar sem overhead
    if not sources_text or not preliminary_text or len(sources_text.strip()) < min_src_len:
        return preliminary_text, audit_log

    # Formata prompt do auditor com fallback seguro
    try:
        audit_prompt_text = auditor_template.format(
            sources_text=sources_text[:7000],
            user_prompt=user_prompt[:500],
            preliminary_text=preliminary_text[:4000]
        )
    except Exception:
        audit_prompt_text = FACT_AUDITOR_PROMPT.format(
            sources_text=sources_text[:7000],
            user_prompt=user_prompt[:500],
            preliminary_text=preliminary_text[:4000]
        )

    # Obtém modelos configurados no Dashboard (Padrão: Classifier_Fast -> Combo_Free_Chat)
    try:
        from admin_config_service import get_category_models
        configured_auditor_models = get_category_models("FACT_AUDITOR") or ["Classifier_Fast", "Combo_Free_Chat"]
    except Exception:
        configured_auditor_models = ["Classifier_Fast", "Combo_Free_Chat"]

    pri_auditor_model = configured_auditor_models[0] if len(configured_auditor_models) > 0 else "Classifier_Fast"
    fb_auditor_model = configured_auditor_models[1] if len(configured_auditor_models) > 1 else "Combo_Free_Chat"

    pri_auditor = get_auditor_llm(pri_auditor_model)
    fb_auditor = get_auditor_llm(fb_auditor_model)

    try:
        try:
            audit_res = await asyncio.wait_for(pri_auditor.ainvoke(audit_prompt_text), timeout=auditor_timeout)
        except Exception:
            audit_res = await asyncio.wait_for(fb_auditor.ainvoke(audit_prompt_text), timeout=auditor_timeout)

        raw_audit = audit_res.content if hasattr(audit_res, "content") else str(audit_res)
        if "</think>" in raw_audit:
            raw_audit = raw_audit.split("</think>", 1)[-1].strip()

        # Extrai JSON
        parsed_audit = None
        match = re.search(r"\{.*\}", raw_audit, re.DOTALL)
        if match:
            try:
                parsed_audit = json.loads(match.group(0))
            except Exception:
                pass

        if parsed_audit and isinstance(parsed_audit, dict):
            status = parsed_audit.get("status", "APPROVED")
            hallucination = bool(parsed_audit.get("hallucination_detected", False))
            revised_text = parsed_audit.get("revised_text", "").strip()
            score = float(parsed_audit.get("confidence_score", 1.0))
            notes = parsed_audit.get("audit_notes", "Auditoria concluída com sucesso.")

            audit_duration = round(time.time() - t_audit_start, 2)
            audit_log.update({
                "status": status,
                "hallucination_detected": hallucination,
                "confidence_score": score,
                "notes": notes,
                "duration_seconds": audit_duration
            })

            print(f"[Auditor de Fatos - {flow_name}] Status: {status} | Alucinação Detectada: {hallucination} | Score: {score} ({audit_duration}s)")
            if status == "CORRECTED" and revised_text and len(revised_text) > 30:
                print(f"[Auditor de Fatos - {flow_name}] Texto corrigido pelo auditor para eliminar inconsistências.")
                return revised_text, audit_log

    except Exception as audit_err:
        print(f"[Auditor de Fatos Aviso]: Falha não bloqueante na auditoria ({audit_err}). Mantendo resposta preliminar.")
        audit_log["status"] = f"error: {str(audit_err)}"
        audit_log["duration_seconds"] = round(time.time() - t_audit_start, 2)

    return preliminary_text, audit_log

async def async_searx_search(query: str, max_results: int = 4, user_id: str = "default_user") -> str:
    """Faz busca nativa no SearXNG e opcionalmente extrai o conteúdo profundo das páginas usando Crawl4AI."""
    if not SEARXNG_URL:
        return ""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"})
            if res.status_code == 200:
                data = res.json()
                results = []
                urls_to_scrape = []
                
                for item in data.get("results", [])[:max_results]:
                    url = item.get("url")
                    title = item.get("title", "")
                    snippet = item.get("content", item.get("snippet", ""))
                    results.append({
                        "url": url,
                        "content": snippet,
                        "title": title
                    })
                    if url and not url.endswith((".pdf", ".zip", ".exe", ".png", ".jpg")):
                        urls_to_scrape.append({"url": url, "title": title})
                
                # Scraping profundo simultâneo dos 2 melhores resultados para expandir o contexto
                if urls_to_scrape:
                    print(f"[Agente Search] Iniciando extração profunda em {min(2, len(urls_to_scrape))} URLs encontradas...")
                    async def fetch_and_ingest(u_info):
                        try:
                            md = await asyncio.wait_for(scrape_url_to_markdown(u_info["url"]), timeout=4.5)
                            if md and not md.startswith("[Erro"):
                                # Salva a página na memória em background sem bloquear a resposta do usuário!
                                asyncio.create_task(asyncio.to_thread(
                                    ingest_user_text_memory,
                                    text=f"Título: {u_info['title']}\nURL: {u_info['url']}\n\n{md[:6000]}",
                                    user_id=user_id,
                                    doc_title=f"Search: {u_info['title'][:30]}",
                                    ttl_days=1
                                ))
                                return {"url": u_info["url"], "deep_content": md}
                        except (asyncio.TimeoutError, Exception) as scrape_err:
                            print(f"[Aviso Scrape URL {u_info['url']}]: {scrape_err}")
                        return None
                    
                    # Roda a extração em paralelo nos top 2 com limite de tempo individual
                    tasks = [fetch_and_ingest(u) for u in urls_to_scrape[:2]]
                    deep_contents = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Substitui o snippet pelo conteúdo profundo se o scraping for bem-sucedido
                    for dc in deep_contents:
                        if isinstance(dc, dict) and dc.get("url") and dc.get("deep_content"):
                            for r in results:
                                if r["url"] == dc["url"]:
                                    r["content"] = dc["deep_content"]
                                    break

                # Formata os resultados de busca em texto legível para o LLM
                formatted_blocks = []
                for idx, r in enumerate(results, 1):
                    formatted_blocks.append(
                        f"--- [Resultado {idx}] {r['title']} ---\nURL: {r['url']}\nConteúdo:\n{r['content']}\n"
                    )
                return "\n".join(formatted_blocks)
    except Exception as e:
        print(f"[SearXNG Error]: {e}")
    return ""

# Ferramentas do Agente
tavily_tool = None
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    tavily_tool = TavilySearchResults(max_results=3)

weather_tool = None
openweather_key = os.getenv("OPENWEATHERMAP_API_KEY")
if openweather_key:
    weather_wrapper = OpenWeatherMapAPIWrapper()
    weather_tool = OpenWeatherMapQueryRun(api_wrapper=weather_wrapper)

# Modelo para Síntese do Agente e RAG
agent_llm = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model="ComboFree_Smart",
    temperature=0.3,
    max_tokens=4096
)

async def execute_agent_flow(category: str, user_prompt: str, context_prompt: str = "", user_id: str = "default_user") -> Tuple[str, List[Dict[str, Any]]]:
    """Executa a ferramenta adequada (SearXNG/Tavily/OpenWeather) com busca web em tempo real e síntese assertiva."""
    tools_called_log = []
    tool_data = ""
    tool_start = time.time()

    if category == "WEATHER_INFO":
        query_text = user_prompt
        if context_prompt and len(user_prompt.split()) <= 4:
            query_text = f"{context_prompt} {user_prompt}"
        location = clean_location_query(query_text)
        print(f"[Agente Tool] Consultando OpenWeather para localidade: '{location}'")
        
        weather_success = False
        if weather_tool:
            try:
                raw_data = await weather_tool.ainvoke(location)
                if "Unable to find the resource" not in str(raw_data) and "not found" not in str(raw_data).lower() and "invalid api key" not in str(raw_data).lower():
                    tool_data = str(raw_data)
                    weather_success = True
                    tools_called_log.append({
                        "tool": "open_weather_map",
                        "location": location,
                        "status": "success",
                        "duration_seconds": round(time.time() - tool_start, 3)
                    })
            except Exception as e:
                print(f"[Aviso OpenWeather]: {e}. Tentando fallback via Tavily.")

        if not weather_success:
            fallback_start = time.time()
            search_query = f"previsão do tempo hoje {location}"
            print(f"[Agente Tool Fallback] Consultando SearXNG/Tavily para: '{search_query}'")
            try:
                if SEARXNG_URL:
                    tool_data = await async_searx_search(search_query, user_id=user_id)
                    tool_name = "searxng_search_results"
                elif tavily_tool:
                    t_res = await tavily_tool.ainvoke(search_query)
                    tool_data = str(t_res)
                    tool_name = "tavily_search_results (weather fallback)"
                else:
                    raise Exception("No search engine available")
                    
                tools_called_log.append({
                    "tool": tool_name,
                    "query": search_query,
                    "status": "success",
                    "duration_seconds": round(time.time() - fallback_start, 3)
                })
            except Exception as t_err:
                tools_called_log.append({
                    "tool": "search_fallback",
                    "query": search_query,
                    "status": f"error: {t_err}",
                    "duration_seconds": round(time.time() - fallback_start, 3)
                })

    elif category == "REALTIME_SEARCH":
        # Extração inteligente de termos de busca via LLM rápido para evitar enviar instruções conversacionais brutas ao motor
        search_query = user_prompt
        try:
            query_extractor_llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model="Classifier_Fast",
                temperature=0.0
            )
            q_system = (
                "Você é um extrator de termos de busca na Web (Google/Bing/SearXNG).\n"
                "Converta o pedido do usuário em palavras-chave objetivas e assertivas de busca.\n"
                "Regras:\n"
                "1. Se o usuário mencionar um domínio ou site (ex: aquaticapaulista.org.br ou typesafe), inclua o nome ou domínio nas palavras-chave.\n"
                "2. Remova instruções conversacionais (ex: 'verifique a informação correta', 'procure nos calendários', 'me diga quando').\n"
                "3. Se houver pergunta anterior de contexto relevante, combine o assunto com o pedido atual.\n"
                "4. Responda APENAS as palavras-chave na mesma linha, sem aspas, sem pontuação final e sem explicações."
            )
            q_user = f"Contexto prévio: {context_prompt}\nMensagem do usuário: {user_prompt}" if context_prompt else f"Mensagem do usuário: {user_prompt}"
            extracted_msg = await query_extractor_llm.ainvoke([
                {"role": "system", "content": q_system},
                {"role": "user", "content": q_user}
            ])
            extracted_text = extracted_msg.content.strip().strip('"\'')
            if extracted_text and len(extracted_text) >= 3:
                search_query = extracted_text
        except Exception as q_err:
            print(f"[Aviso Extrator de Busca]: {q_err}. Usando fallback de query limpa.")
            if context_prompt and len(user_prompt.split()) <= 7:
                clean_prev = context_prompt.rstrip("?.!")
                search_query = f"{clean_prev} {user_prompt}"

        print(f"[Agente Tool] Rota REALTIME_SEARCH acionada. Termos de busca otimizados: '{search_query}'")
        tool_name = "searxng_search_results"
        tool_status = "success"

        # 1. Checagem prioritária em APIs Oficiais (Câmbio / BCB SGS) para precisão matemática total (0 erro de scraping)
        from finance_api_service import fetch_currency_rates, fetch_bcb_indicators
        from canonical_api_service import (
            fetch_cep_location,
            resolve_geographic_context,
            fetch_cnpj_data,
            fetch_fipe_vehicle_price
        )
        canonical_hit = False

        # A. Câmbio
        currency_data = await fetch_currency_rates(user_prompt)
        if currency_data:
            tool_data = f"--- DADOS OFICIAIS DE CÂMBIO (AwesomeAPI / Mercado Financeiro) ---\n{json.dumps(currency_data, ensure_ascii=False, indent=2)}\n"
            tool_name = "awesomeapi_currency_rates"
            canonical_hit = True
            print(f"[Agente Tool Finance] Cotação/Série oficial de câmbio obtida via AwesomeAPI ({currency_data.get('pair')}).")

        # B. Indicadores BCB SGS
        if not canonical_hit:
            bcb_data = await fetch_bcb_indicators(user_prompt)
            if bcb_data:
                tool_data = f"--- DADOS OFICIAIS DO BANCO CENTRAL DO BRASIL (BCB SGS) ---\n{json.dumps(bcb_data, ensure_ascii=False, indent=2)}\n"
                tool_name = "bcb_sgs_economic_indicators"
                canonical_hit = True
                print(f"[Agente Tool Finance] Indicador oficial obtido via BCB SGS ({bcb_data.get('indicador')}).")

        # C. CNPJ / Receita Federal
        if not canonical_hit:
            cnpj_data = await fetch_cnpj_data(user_prompt)
            if cnpj_data:
                tool_data = f"--- DADOS OFICIAIS CADASTRAIS DE EMPRESA (Receita Federal / BrasilAPI) ---\n{json.dumps(cnpj_data, ensure_ascii=False, indent=2)}\n"
                tool_name = "receita_federal_cnpj_lookup"
                canonical_hit = True
                print(f"[Agente Tool CNPJ] Dados cadastrais oficiais obtidos para CNPJ: {cnpj_data.get('cnpj')} ({cnpj_data.get('razao_social')}).")

        # D. CEP e Localização Geográfica
        if not canonical_hit:
            cep_data = await fetch_cep_location(user_prompt)
            if cep_data:
                tool_data = f"--- DADOS OFICIAIS DE ENDEREÇO E CEP (BrasilAPI / ViaCEP) ---\n{json.dumps(cep_data, ensure_ascii=False, indent=2)}\n"
                tool_name = "brasilapi_cep_location"
                canonical_hit = True
                print(f"[Agente Tool CEP] Endereço e geolocalização obtidos para CEP: {cep_data.get('cep')} ({cep_data.get('cidade')}/{cep_data.get('estado')}).")

        # E. Tabela FIPE (Veículos)
        if not canonical_hit:
            fipe_data = await fetch_fipe_vehicle_price(user_prompt)
            if fipe_data:
                tool_data = f"--- DADOS OFICIAIS DE PREÇO DE VEÍCULOS (Tabela FIPE / BrasilAPI) ---\n{json.dumps(fipe_data, ensure_ascii=False, indent=2)}\n"
                tool_name = "tabela_fipe_vehicle_price"
                canonical_hit = True
                print(f"[Agente Tool FIPE] Preço oficial FIPE obtido para: {fipe_data.get('modelo')} ({fipe_data.get('valor')}).")

        # Se não casou com APIs diretas, é uma busca Web geral.
        # Enriquecimento Geográfico Automático: se a pergunta citar cidade ou bairro (ex: 'Santo André'),
        # resolve o contexto territorial e injeta no prompt de síntese para enriquecer as sugestões locais.
        geo_context_block = ""
        if not canonical_hit:
            # Detecta menções a locais em buscas de comércios/restaurantes
            local_intent = any(k in user_prompt.lower() for k in [
                "restaurante", "pizzaria", "bar", "comida", "hamburgueria", "café",
                "onde comer", "onde fica", "hotéis", "passeios", "em ", "no ", "na "
            ])
            if local_intent:
                loc_cand = clean_location_query(user_prompt)
                if loc_cand and len(loc_cand.split()) <= 4:
                    geo_info = await resolve_geographic_context(loc_cand)
                    if geo_info:
                        geo_context_block = (
                            f"\nCONTEXTO TERRITORIAL & GEOGRÁFICO VALIDADO (OpenStreetMap):\n"
                            f"- Localidade: {geo_info.get('display_name')}\n"
                            f"- Coordenadas: Lat {geo_info.get('latitude')}, Lon {geo_info.get('longitude')}\n"
                            f"- Município: {geo_info.get('cidade')} | Estado: {geo_info.get('estado')}\n"
                            f"DIRETRIZ DE LOCALIZAÇÃO: Sugira estabelecimentos localizados prioritariamente dentro deste município/bairro "
                            f"ou nas suas imediações mais conhecidas.\n"
                        )
                        print(f"[Agente Tool Geo] Contexto geográfico resolvido para '{loc_cand}': {geo_info.get('cidade')}/{geo_info.get('estado')}.")

        if not canonical_hit:
            try:
                searx_success = False
                if SEARXNG_URL:
                    tool_data = await async_searx_search(search_query, user_id=user_id)
                    tool_name = "searxng_search_results"
                    # Verifica se retornou resultados reais (não apenas string vazia ou sem blocos de resultado)
                    if tool_data and "--- [Resultado" in tool_data:
                        searx_success = True
                    else:
                        print(f"[Agente Tool] SearXNG retornou sem resultados para '{search_query}'. Acionando fallback Tavily...")

                if not searx_success and tavily_tool:
                    t_res = await tavily_tool.ainvoke(search_query)
                    tool_data = str(t_res)
                    tool_name = "tavily_search_results (searxng fallback)" if SEARXNG_URL else "tavily_search_results"
                    
                    # Salva os snippets de busca retornados pelo Tavily na memória RAG do usuário em background
                    if isinstance(t_res, list):
                        for item in t_res[:3]:
                            if isinstance(item, dict) and item.get("content"):
                                asyncio.create_task(asyncio.to_thread(
                                    ingest_user_text_memory,
                                    text=f"Título: {item.get('title', '')}\nURL: {item.get('url', '')}\n\n{item.get('content', '')}",
                                    user_id=user_id,
                                    doc_title=f"Search: {item.get('title', 'Tavily')[:30]}",
                                    ttl_days=1
                                ))
                elif not searx_success and not tavily_tool:
                    tool_data = tool_data or "Nenhuma ferramenta de busca retornou resultados."
                    tool_name = "none"
            except Exception as e:
                print(f"[Erro Search API]: {e}. Tentando fallback...")
                if tavily_tool:
                    try:
                        t_res = await tavily_tool.ainvoke(search_query)
                        tool_data = str(t_res)
                        tool_name = "tavily_search_results (error fallback)"
                        if isinstance(t_res, list):
                            for item in t_res[:3]:
                                if isinstance(item, dict) and item.get("content"):
                                    asyncio.create_task(asyncio.to_thread(
                                        ingest_user_text_memory,
                                        text=f"Título: {item.get('title', '')}\nURL: {item.get('url', '')}\n\n{item.get('content', '')}",
                                        user_id=user_id,
                                        doc_title=f"Search: {item.get('title', 'Tavily')[:30]}",
                                        ttl_days=1
                                    ))
                    except Exception as t_err:
                        tool_data = f"Erro na consulta Web: {e} | Tavily: {t_err}"
                        tool_status = f"error: {e}"
                else:
                    tool_data = f"Erro na consulta Web: {e}"
                    tool_status = f"error: {e}"

            tools_called_log.append({
                "tool": tool_name,
                "query": search_query,
                "status": tool_status,
                "duration_seconds": round(time.time() - tool_start, 3)
            })
        else:
            tools_called_log.append({
                "tool": tool_name,
                "query": user_prompt,
                "status": "success",
                "duration_seconds": round(time.time() - tool_start, 3)
            })

        # Consulta híbrida prévia no RAG/Memória para enriquecer o contexto da Web com notícias pré-catalogadas
        rag_mem_block = ""
        try:
            rag_docs = await search_unified_rag(search_query, user_id=user_id, k=2)
            if rag_docs:
                mem_snippets = []
                for rd in rag_docs:
                    if float(rd.get("score", 0)) >= 0.52:
                        mem_snippets.append(f"- [{rd.get('source', 'Memória')}]: {rd.get('content', '')[:400]}")
                if mem_snippets:
                    rag_mem_block = "\nINFORMAÇÕES PRÉ-CATALOGADAS NA BASE LOCAL:\n" + "\n".join(mem_snippets) + "\n"
        except Exception as rag_err:
            print(f"[Aviso RAG na Busca Web]: {rag_err}")

    context_block = f"\nCONTEXTO DA PERGUNTA ANTERIOR:\n{context_prompt}\n" if context_prompt else ""
    if 'geo_context_block' in locals() and geo_context_block:
        context_block = geo_context_block + context_block
    if 'rag_mem_block' in locals() and rag_mem_block:
        context_block = rag_mem_block + context_block
    
    # Avaliação de risco/certeza dos dados da Web em tempo real
    has_meaningful_web_data = bool(tool_data and len(tool_data.strip()) > 80 and "Erro na consulta" not in tool_data and "Nenhuma ferramenta" not in tool_data)
    
    web_confidence_instruction = ""
    if not has_meaningful_web_data:
        web_confidence_instruction = """
DIRETRIZ DE INCERTEZA / DADOS NÃO CONCLUSIVOS:
Os dados obtidos nas ferramentas de busca em tempo real são insuficientes, vagos ou não tratam com clareza da entidade/assunto exato perguntado pelo usuário.
1. NÃO invente informações, dados históricos fictícios ou características não confirmadas.
2. Diga com transparência o que foi ou não foi encontrado nas fontes consultadas.
3. Conclua OBRIGATORIAMENTE perguntando ao usuário se era essa a informação esperada ou se ele pode fornecer mais contexto / link de referência para refinamento.
"""
    else:
        web_confidence_instruction = """
DIRETRIZ DE VALIDAÇÃO:
Apresente a resposta com base estritamente nos dados encontrados. Se houver alguma ambiguidade quanto a nomes, datas, versões ou intenção do usuário, conclua confirmando se esses dados atendem ao que ele esperava.
"""

    synthesis_prompt = f"""Você é um assistente prestativo e altamente atualizado.
Responda à pergunta do usuário baseando-se RIGOROSAMENTE nas informações e dados obtidos em tempo real abaixo.
Não invente fatos, não confunda ferramentas ou empresas e não traga informações de temas não relacionados.
Se as informações em tempo real trouxerem o significado exato ou o produto/tecnologia buscado, explique detalhadamente suas características, utilidades e como funciona.
{web_confidence_instruction}
DADOS OBTIDOS EM TEMPO REAL:
{tool_data}
{context_block}
PERGUNTA ATUAL DO USUÁRIO:
{user_prompt}

Forneça uma resposta completa, assertiva e em português:"""

    try:
        synth_response = await agent_llm.ainvoke(synthesis_prompt)
        raw_text = synth_response.content if hasattr(synth_response, "content") else str(synth_response)
        # Se o modelo expuser tag de reasoning aberta/fechada, extrai apenas o conteúdo final
        if "</think>" in raw_text:
            output_text = raw_text.split("</think>", 1)[-1].strip()
        else:
            output_text = raw_text.strip()
    except Exception as synth_err:
        print(f"[Erro na Síntese do Agente]: {synth_err}")
        output_text = f"Dados obtidos:\n{tool_data}"

    # Auditor Reflexivo Anti-Alucinação para Busca Web em Tempo Real
    if tool_data and category == "REALTIME_SEARCH":
        audited_text, audit_entry = await audit_response_facts(
            user_prompt=user_prompt,
            preliminary_text=output_text,
            sources_text=tool_data,
            flow_name="WebSearch"
        )
        output_text = audited_text
        tools_called_log.append(audit_entry)

    # ---------------------------------------------------------
    # Geração Integrada de Gráfico & Planilha se solicitado no fluxo normal
    # ---------------------------------------------------------
    chart_keywords = [
        "gráfico", "grafico", "planilha", "excel", "csv", "tabela comparativa",
        "plotar", "plote", "visualização gráfica", "visualizacao grafica"
    ]
    user_wants_visual_chart = any(k in user_prompt.lower() for k in chart_keywords)

    if user_wants_visual_chart and (tool_data or output_text):
        try:
            from data_export_service import execute_data_export_flow
            print(f"[Agente Normal Flow] Pedido de gráfico/planilha detectado na resposta da busca. Gerando imagem e downloads...")
            export_output, export_tools = await execute_data_export_flow(
                user_prompt=user_prompt,
                context_text=f"RESUMO / DADOS OBTIDOS:\n{output_text}\n\nFONTES BRUTAS:\n{tool_data[:4000]}"
            )
            tools_called_log.extend(export_tools)
            # Concatena a análise textual prévia com a seção de gráficos e downloads estruturados
            output_text = f"{output_text}\n\n---\n\n{export_output}"
        except Exception as chart_err:
            print(f"[Aviso Geração de Gráfico no Fluxo Normal]: {chart_err}")

    return output_text, tools_called_log

async def execute_rag_flow(user_prompt: str, context_prompt: str = "", user_id: str = "default_user") -> Tuple[str, List[Dict[str, Any]]]:
    """Busca trechos relevantes no Qdrant (KB + Memória Temporária) e sintetiza a resposta final com limiar de confiança."""
    start_time = time.time()
    
    search_query = user_prompt
    if context_prompt and len(user_prompt.split()) <= 7:
        clean_prev = context_prompt.rstrip("?.!")
        search_query = f"{clean_prev} - {user_prompt}"

    rag_results = await search_unified_rag(search_query, user_id=user_id, k=4)
    
    top_score = 0.0
    if not rag_results:
        context_str = "Nenhum documento ou memória relevante encontrado na base."
        sources_summary = []
        confidence_level = "LOW"
    else:
        context_blocks = []
        sources_summary = []
        scores = []
        for idx, doc in enumerate(rag_results, 1):
            src = doc["source"]
            mtype = doc.get("memory_type", "")
            page = f" (Pág. {doc['page']})" if doc.get("page") else ""
            type_tag = f" [{mtype}]" if mtype else ""
            sc = float(doc.get("score", 0.0))
            scores.append(sc)
            context_blocks.append(f"--- Documento [{idx}] {src}{page}{type_tag} (Relevância: {sc}) ---\n{doc['content']}")
            sources_summary.append(f"{src}{page}{type_tag}")
        context_str = "\n\n".join(context_blocks)
        top_score = max(scores) if scores else 0.0
        
        # Limiares de Confiança Semântica Configuráveis via Dashboard/Admin
        try:
            from admin_config_service import get_guardrail_value
            rag_thresh_high = float(get_guardrail_value("rag_confidence_threshold_high", 0.65))
            rag_thresh_med = float(get_guardrail_value("rag_confidence_threshold_medium", 0.50))
            rag_fallback_web_thresh = float(get_guardrail_value("rag_fallback_web_threshold", 0.64))
        except Exception:
            rag_thresh_high = 0.65
            rag_thresh_med = 0.50
            rag_fallback_web_thresh = 0.64

        if top_score >= rag_thresh_high:
            confidence_level = "HIGH"
        elif top_score >= rag_thresh_med:
            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"

    print(f"[RAG Confiança] Top Score: {top_score:.4f} -> Nível de Confiança: {confidence_level}")

    # Fallback Transparente para Web Search:
    # Se a base vetorial não possuir documentos com confiança (top_score < rag_fallback_web_thresh) ou for vazia,
    # não force resposta alucinatória nem responda 'não tenho dados'. Recorra à busca na Web!
    explicit_file_query = any(k in user_prompt.lower() for k in ["pdf", "anexo", "meu documento", "meus arquivos", "no manual", "no livro que enviei"])
    if (confidence_level in ["LOW", "MEDIUM"] and top_score < rag_fallback_web_thresh and not explicit_file_query) or not rag_results:
        print(f"[RAG Fallback Ativado] Relevância da base local insuficiente (score: {top_score:.4f}). Consultando fontes atualizadas na Web...")
        web_text, web_tools = await execute_agent_flow(
            category="REALTIME_SEARCH",
            user_prompt=user_prompt,
            context_prompt=context_prompt,
            user_id=user_id
        )
        tools_called_log = [{
            "tool": "qdrant_unified_vector_search",
            "user_id": user_id,
            "query": user_prompt,
            "sources_found": sources_summary,
            "top_score": round(top_score, 4),
            "confidence_level": "FALLBACK_TO_WEB",
            "duration_seconds": round(time.time() - start_time, 3)
        }] + web_tools
        return web_text, tools_called_log

    synthesis_prompt = build_rag_synthesis_prompt(user_prompt, context_str, confidence_level=confidence_level)

    try:
        synth_response = await agent_llm.ainvoke(synthesis_prompt)
        output_text = synth_response.content if hasattr(synth_response, "content") else str(synth_response)
        if "</think>" in output_text:
            output_text = output_text.split("</think>", 1)[-1].strip()
    except Exception as synth_err:
        print(f"[Erro Síntese RAG]: {synth_err}")
        output_text = f"Trechos encontrados na base de conhecimento e memórias:\n\n{context_str}"

    tools_called_log = [{
        "tool": "qdrant_unified_vector_store",
        "user_id": user_id,
        "query": user_prompt,
        "sources_found": sources_summary,
        "top_score": round(top_score, 4),
        "confidence_level": confidence_level,
        "duration_seconds": round(time.time() - start_time, 3)
    }]

    # Auditor Reflexivo Anti-Alucinação para RAG
    if rag_results:
        audited_text, audit_entry = await audit_response_facts(
            user_prompt=user_prompt,
            preliminary_text=output_text,
            sources_text=context_str,
            flow_name="RAG"
        )
        output_text = audited_text
        tools_called_log.append(audit_entry)

    # Geração Integrada de Gráfico & Planilha no RAG se solicitado
    chart_keywords = [
        "gráfico", "grafico", "planilha", "excel", "csv", "tabela comparativa",
        "plotar", "plote", "visualização gráfica", "visualizacao grafica"
    ]
    if any(k in user_prompt.lower() for k in chart_keywords) and context_str:
        try:
            from data_export_service import execute_data_export_flow
            print(f"[RAG Flow] Pedido de gráfico/planilha detectado na resposta RAG. Gerando imagem e downloads...")
            export_output, export_tools = await execute_data_export_flow(
                user_prompt=user_prompt,
                context_text=f"RESUMO / DADOS OBTIDOS:\n{output_text}\n\nDOCUMENTOS:\n{context_str[:4000]}"
            )
            tools_called_log.extend(export_tools)
            output_text = f"{output_text}\n\n---\n\n{export_output}"
        except Exception as chart_err:
            print(f"[Aviso Geração de Gráfico no RAG Flow]: {chart_err}")

    return output_text, tools_called_log
