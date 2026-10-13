import os
import re
import json
import time
import asyncio
from typing import Optional, Tuple, Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import ChatOpenAI
from config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    message_has_image,
    history_has_images
)
from cache_service import get_qdrant_client, embeddings

SEMANTIC_ROUTES_COLLECTION = os.getenv("SEMANTIC_ROUTES_COLLECTION", "semantic_routes_v2")
DEFAULT_VECTOR_THRESHOLD = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.78"))

# =====================================================================
# Sementes Canônicas Ricas para as 16 Categorias do Orquestrador
# =====================================================================
CANONICAL_SEEDS: Dict[str, List[str]] = {
    "CODING": [
        "como configurar acesso a servidor de arquivos smb em linux pelo windows sem precisar de senha",
        "configurar compartilhamento de rede smb ou samba no linux ubuntu debian",
        "comando powershell set-smbclientconfiguration enableinsecureguestlogons",
        "como montar partição ou disco de rede no fstab do linux ou windows",
        "escreva um script python para automação de tarefas e processamento de dados",
        "crie uma função typescript javascript com tipagem estática e regex",
        "comando no terminal bash para permissão de arquivos chmod chown e systemctl",
        "como fazer conexão ssh em servidor remoto e reiniciar serviço do sistema",
        "depurar erro de sintaxe ou exceção no código de programação",
        "configuração do docker docker-compose com containers de backend e banco",
        "escreva uma query sql complexa com joins agrupamento e índices",
        "como programar uma api rest em python fastapi ou node express",
        "corrija este bug na função e faça o refactor do código",
        "script bash de deploy e automação de infraestrutura em nuvem",
        "como resolver erro de compilação ou dependência de pacote npm pip",
        "meu desktop estava acessando discos compartilhados mas agora pede senha como resolver"
    ],
    "SOCIAL_MEDIA_POST": [
        "crie uma postagem para o instagram com gancho visual e chamada para ação",
        "escreva uma publicação para o linkedin sobre inovação e liderança de mercado",
        "copy persuasiva para rede social com hashtags estratégicas e emojis",
        "faça um post para feed com título chamativo e legenda completa",
        "anúncio para facebook e instagram focado em conversão e engajamento",
        "carrossel educativo com dicas para publicar nas redes sociais",
        "roteiro e legenda para post de divulgação de serviço ou produto",
        "crie uma legenda para foto no feed do instagram com cta para comentar",
        "estratégia de conteúdo e publicação para redes sociais com arte e texto",
        "escreva um post comercial para engajar clientes no linkedin"
    ],
    "IMAGE_GENERATION": [
        "gere uma imagem artística de um astronauta caminhando em marte ao pôr do sol",
        "crie uma ilustração 3d fotorrealista de uma cidade futurista cyberpunk",
        "desenhe uma foto realista de um cachorro da raça golden retriever no parque",
        "faça uma pintura digital de uma montanha com floresta no estilo aquarela",
        "gere uma foto em alta definição de uma xícara de café com grãos sobre a mesa",
        "desenhe um personagem de anime com armadura dourada e espada mágica"
    ],
    "IMAGE_EDIT": [
        "mude o fundo da imagem enviada para uma paisagem de praia ensolarada",
        "remova o objeto ou a pessoa indesejada no canto direito da foto",
        "altere a cor da camisa para azul marinho nesta imagem que anexei",
        "edite esta foto ajustando a iluminação e clareando o rosto da pessoa",
        "troque o fundo branco desta foto por um cenário de escritório moderno",
        "adicione um chapéu na pessoa desta imagem enviada"
    ],
    "SHORT_VIDEO": [
        "crie um vídeo curto e dinâmico para o tiktok sobre curiosidades da ciência",
        "gere um reels vertical com cortes rápidos e trilha sonora envolvente",
        "roteiro de vídeo vertical para youtube shorts com dicas rápidas de negócios",
        "produza um mini vídeo vertical de 30 segundos para instagram reels",
        "faça um vídeo curto no formato 9:16 para stories ou tiktok"
    ],
    "WEATHER_INFO": [
        "qual a previsão do tempo para são paulo hoje vai chover",
        "qual a temperatura atual e clima para o final de semana no rio de janeiro",
        "vai fazer frio amanhã em curitiba ou em porto alegre quantos graus",
        "previsão de chuva e meteorologia para salvador bahia",
        "como está o tempo agora em belo horizonte está ensolarado"
    ],
    "REALTIME_SEARCH": [
        "quais as últimas notícias sobre economia e política no brasil hoje",
        "qual a cotação do dólar comercial euro e bitcoin agora em tempo real",
        "quem venceu a partida de futebol ontem pelo campeonato brasileiro",
        "melhores restaurantes e pizzarias abertas no centro da cidade",
        "resultado oficial das eleições apuração das urnas e fatos recentes de 2026",
        "últimos lançamentos de filmes e séries nos cinemas e streaming este ano"
    ],
    "CASUAL_CHAT": [
        "olá bom dia como você está tudo bem",
        "muito obrigado pela ajuda você foi muito gentil e prestativo",
        "oi assistente inteligente quem é você e o que você faz",
        "boa noite até logo um abraço",
        "você é muito legal valeu pela força"
    ],
    "LOCAL_THINKING": [
        "se eu tenho cinco laranjas e como duas quantas sobram na cesta",
        "resolva este enigma simples dos três chapéus e dos prisioneiros",
        "problema de lógica simples envolvendo velocidade tempo e distância",
        "charada de raciocínio lógico sobre quem chegou primeiro na corrida"
    ],
    "HEAVY_REASONING": [
        "calcule o limite e a derivada parcial de segunda ordem desta equação",
        "demonstre a solução matemática para este sistema de equações diferenciais",
        "análise tática avançada de variantes e finais de torres no xadrez",
        "formulação teórica dos princípios da mecânica quântica e relatividade geral",
        "resolva este cálculo integral avançado com passo a passo formal"
    ],
    "FACTUAL_GENERAL": [
        "faça uma dissertação histórica sobre a revolução constitucionalista de 1932",
        "explique em detalhes o funcionamento biológico do processo de fotossíntese",
        "quem foi júlio césar e quais as causas da queda do império romano",
        "redação explicativa sobre a formação do sistema solar e dos planetas",
        "resuma o enredo e os temas principais do livro dom casmurro de machado de assis"
    ],
    "VISION": [
        "descreva o que você consegue ver nesta imagem enviada",
        "analise os detalhes elementos visuais e pessoas presentes na foto anexa",
        "leia e transcreva o texto ou tabela que aparece nesta foto enviada",
        "o que está escrito nesta placa ou documento da foto"
    ],
    "DATA_EXPORT_CHART": [
        "crie uma planilha em excel com projeção de despesas e receitas para download",
        "gere uma tabela comparativa com botão para baixar em formato csv ou xlsx",
        "plote um gráfico de barras comparando os resultados do primeiro e segundo semestre",
        "monte uma planilha financeira completa com fórmulas e download em excel",
        "gere um gráfico de pizza para demonstrar a distribuição de gastos"
    ],
    "MCP_FILESYSTEM": [
        "liste os arquivos e pastas presentes no diretório workspace seguro",
        "crie um arquivo chamado script.py com código dentro do workspace",
        "leia e exiba o conteúdo do arquivo dados.txt localizado no workspace",
        "apague o arquivo temporário teste.json da pasta do workspace",
        "busque arquivos com extensão .csv dentro do workspace"
    ],
    "MCP_FETCH": [
        "acesse a página https://example.com e extraia o conteúdo principal",
        "leia a url https://g1.globo.com/noticia e me traga um resumo do texto",
        "faça fetch da página web https://github.com/projeto e mostre os tópicos",
        "visite o site https://... e extraia os dados da tabela"
    ],
    "MCP_GIT": [
        "verifique o git status do repositório no workspace",
        "mostre o git log com os últimos commits realizados",
        "quais foram as alterações no git diff antes do commit",
        "faça um commit com a mensagem finalizando a implementação do serviço",
        "inicialize um repositório git init na pasta do projeto",
        "quais as branches disponíveis no git branch do workspace"
    ],
    "MCP_BRAVE_SEARCH": [
        "busque no brave search sobre as últimas notícias de tecnologia",
        "pesquise no brave as novidades da inteligência artificial",
        "faça uma busca web pelo mcp brave sobre cotação de moedas e economia",
        "procure no brave search por artigos recentes sobre kubernetes"
    ],
    "MCP_DATABASE": [
        "liste todas as tabelas e views presentes no banco de dados do workspace",
        "descreva a estrutura e colunas da tabela de clientes no banco de dados",
        "execute uma consulta sql select para buscar todos os registros com saldo positivo",
        "faça uma query no banco sqlite para contar quantos usuários foram cadastrados",
        "execute o comando sql para criar a tabela de produtos no workspace",
        "consulte os dados do arquivo csv analítico utilizando duckdb"
    ],
    "MCP_CODE_RUNNER": [
        "execute este código python inline e mostre o que sai no console",
        "rode o script python teste.py que está salvo no workspace",
        "execute este comando bash no terminal para verificar a versão do sistema",
        "rode este bloco de código e capture a saída stdout e stderr",
        "execute o script shell rotina.sh dentro da sandbox do workspace"
    ],
    "DOCUMENT_RAG": [
        "segundo o manual em formato pdf que enviei qual o procedimento de segurança",
        "de acordo com o relatório corporativo em anexo quais foram as metas atingidas",
        "consulte o documento anexado acima e responda a minha dúvida sobre as regras",
        "no pdf do livro que carreguei o que o autor diz no capítulo 3"
    ]
}

_is_collection_ready = False
_sync_lock = asyncio.Lock()


# =====================================================================
# Inicialização e Indexação das Sementes no Qdrant (0ms após boot)
# =====================================================================
async def ensure_semantic_routes_collection(qc: Optional[QdrantClient] = None):
    """
    Garante que a coleção de roteamento vetorial existe no Qdrant e contém
    as frases canônicas atualizadas para as 16 categorias.
    """
    global _is_collection_ready
    if _is_collection_ready:
        return

    async with _sync_lock:
        if _is_collection_ready:
            return

        client = qc or get_qdrant_client()
        try:
            cols = [c.name for c in client.get_collections().collections]
            total_seeds = sum(len(v) for v in CANONICAL_SEEDS.values())

            needs_seed = False
            if SEMANTIC_ROUTES_COLLECTION not in cols:
                client.create_collection(
                    collection_name=SEMANTIC_ROUTES_COLLECTION,
                    vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE)
                )
                print(f"[Roteador Semântico V2] Coleção '{SEMANTIC_ROUTES_COLLECTION}' criada no Qdrant.")
                needs_seed = True
            else:
                info = client.get_collection(SEMANTIC_ROUTES_COLLECTION)
                if (info.points_count or 0) < total_seeds:
                    needs_seed = True

            if needs_seed:
                print(f"[Roteador Semântico V2] Indexando {total_seeds} sementes canônicas vetoriais...")
                texts: List[str] = []
                payloads: List[Dict[str, Any]] = []

                for cat, phrases in CANONICAL_SEEDS.items():
                    for phrase in phrases:
                        texts.append(phrase)
                        payloads.append({
                            "category": cat,
                            "seed_text": phrase,
                            "created_at": time.time()
                        })

                # Gera embeddings em batch
                vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
                points = [
                    models.PointStruct(
                        id=idx + 1,
                        vector=vector,
                        payload=payloads[idx]
                    )
                    for idx, vector in enumerate(vectors)
                ]

                # Upsert em lote
                client.upsert(
                    collection_name=SEMANTIC_ROUTES_COLLECTION,
                    points=points
                )
                print(f"[Roteador Semântico V2] Indexação concluída! {len(points)} pontos salvos no Qdrant.")

            _is_collection_ready = True
        except Exception as e:
            print(f"[Aviso Roteador Semântico V2] Falha ao verificar/indexar coleção Qdrant: {e}")


# =====================================================================
# Camada 1: Roteamento Semântico Vetorial Instantâneo (5ms a 15ms)
# =====================================================================
async def match_semantic_vector(
    prompt: str,
    threshold: Optional[float] = None
) -> Optional[Tuple[str, str, float, str]]:
    """
    Calcula o embedding da mensagem e busca a categoria mais próxima por similaridade de cosseno.
    Retorna (categoria, método, score, seed_text) se score >= threshold, senão None.
    """
    if not prompt or len(prompt.strip()) < 3:
        return None

    try:
        from admin_config_service import get_guardrail_value
        min_score = float(get_guardrail_value("semantic_router_threshold", threshold or DEFAULT_VECTOR_THRESHOLD))
    except Exception:
        min_score = threshold or DEFAULT_VECTOR_THRESHOLD

    await ensure_semantic_routes_collection()
    client = get_qdrant_client()

    try:
        # Gera embedding da consulta
        query_vec = await asyncio.to_thread(embeddings.embed_query, prompt)

        # Suporta query_points (Qdrant 1.10+) e search (legado)
        if hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=SEMANTIC_ROUTES_COLLECTION,
                query=query_vec,
                limit=3
            )
            hits = res.points
        else:
            hits = client.search(
                collection_name=SEMANTIC_ROUTES_COLLECTION,
                query_vector=query_vec,
                limit=3
            )

        if hits and len(hits) > 0:
            top_hit = hits[0]
            top_score = round(top_hit.score, 4)
            cat = top_hit.payload.get("category")
            seed = top_hit.payload.get("seed_text", "")

            if top_score >= min_score and cat:
                method = f"v2_semantic_vector_match"
                print(f"[Roteador Semântico V2] Vetor casado: Categoria '{cat}' (Score: {top_score:.4f} >= {min_score}) | Semente: '{seed[:50]}...'")
                return cat, method, top_score, seed
            else:
                print(f"[Roteador Semântico V2] Top match '{cat}' com score {top_score:.4f} abaixo do limiar {min_score}. Acionando fallback estruturado.")
    except Exception as e:
        print(f"[Aviso Roteador Semântico V2] Erro na consulta vetorial Qdrant: {e}")

    return None


# =====================================================================
# Camada 2: Classificador LLM com JSON Schema Estruturado (Sem Alucinação)
# =====================================================================
CLASSIFIER_V2_SYSTEM_PROMPT = """Você é o Classificador Semântico e Guardrail de Roteamento de IA do Orquestrador Multi-Agente.
Sua missão é analisar a mensagem do usuário, o contexto anterior e classificar ESTRITAMENTE em uma das 16 categorias permitidas.

CATEGORIAS PERMITIDAS:
1. DOCUMENT_RAG: Dúvidas sobre documentos PDF, relatórios internos ou manuais anexados pelo usuário.
2. WEATHER_INFO: Previsão do tempo, clima, chuva ou temperatura.
3. REALTIME_SEARCH: Notícias recentes de 2026, cotações em tempo real, futebol, eventos atuais ou busca web geral.
4. CASUAL_CHAT: Saudações (oi, olá), agradecimentos e conversa informal.
5. CODING: Programação (Python, JS, SQL), scripts bash, comandos de terminal Linux/Windows, redes, servidores (SMB, Samba, NFS, Nginx), Docker, infraestrutura ou configuração de sistemas operacionais.
6. LOCAL_THINKING: Problemas matemáticos básicos e charadas lógicas simples.
7. HEAVY_REASONING: Matemática avançada (cálculo/integral), física teórica ou xadrez técnico.
8. FACTUAL_GENERAL: Conceitos enciclopédicos consolidados, história, dissertações e temas gerais.
9. VISION: Análise e descrição de imagens ou fotos anexadas.
10. IMAGE_GENERATION: Pedidos para criar ou desenhar novas imagens artísticas puras.
11. IMAGE_EDIT: Pedidos para alterar ou editar imagem existente.
12. SOCIAL_MEDIA_POST: Criação de publicações, posts, anúncios, carrosséis ou copies para redes sociais (Instagram, LinkedIn, etc.).
13. DATA_EXPORT_CHART: Geração de planilhas Excel/CSV ou gráficos de dados.
14. SHORT_VIDEO: Vídeos curtos verticais (Reels, TikTok, Shorts).
15. MCP_FILESYSTEM: Comandos explícitos para manipular arquivos e pastas dentro do diretório sandbox 'workspace'. NUNCA classifique dúvidas sobre servidores de rede ou Linux geral como MCP_FILESYSTEM!
16. MCP_FETCH: Acessar ou extrair conteúdo de páginas web via URL explícita no prompt.
17. MCP_GIT: Controle de versão Git dentro do workspace (git status, git diff, git log, git commit, git branch, git init, git checkout).
18. MCP_BRAVE_SEARCH: Busca web explicitamente solicitada através do motor independente Brave Search.
19. MCP_DATABASE: Comandos para consultar bancos de dados (SQLite, DuckDB, PostgreSQL), listar tabelas, descrever schemas de tabelas ou executar queries SQL.
20. MCP_CODE_RUNNER: Comandos explícitos para executar snippets de código (Python, Bash, scripts) na sandbox e capturar a saída de execução.

DIRETRIZ DE DETECÇÃO DE MUDANÇA DE TÓPICO (TOPIC SHIFT):
- Se a mensagem atual não tiver nenhuma relação com a última resposta do assistente (ex: assistente falou de rede social e o usuário perguntou sobre servidor SMB), 'topic_shift' DEVE ser true e 'is_followup' DEVE ser false!
- Se for continuação direta (ex: 'explique melhor', 'faça em python', 'e amanhã?'), 'topic_shift' é false e 'is_followup' é true.

REQUISITO OBRIGATÓRIO DE SAÍDA:
Você DEVE responder ESTRITAMENTE em formato JSON válido com as seguintes chaves:
{
  "category": "UMA_DAS_20_CATEGORIAS_ACIMA",
  "is_followup": true ou false,
  "topic_shift": true ou false,
  "rewritten_prompt": "pergunta completa expandida ou a própria mensagem se for auto-suficiente",
  "confidence": 0.0 a 1.0,
  "reason": "justificativa concisa em 1 frase"
}"""


async def classify_with_structured_json(
    user_prompt: str,
    context_str: str,
    last_category: Optional[str] = None,
    last_model: Optional[str] = None
) -> Tuple[str, str, str, Optional[str]]:
    """
    Invoca o modelo LLM configurado com response_format JSON estrito,
    eliminando 100% de textos conversacionais em Markdown e truncamento de tokens.
    """
    try:
        from admin_config_service import get_category_models
        models_cfg = get_category_models("SEMANTIC_CLASSIFIER") or ["Classifier_Fast", "Combo_Free_Chat"]
    except Exception:
        models_cfg = ["Classifier_Fast", "Combo_Free_Chat"]

    primary_model = models_cfg[0] if models_cfg else "Classifier_Fast"
    fallback_model = models_cfg[1] if len(models_cfg) > 1 else "Combo_Free_Chat"

    user_payload_text = f"""CONTEXTO DA SESSÃO:
- Categoria anterior: {last_category or 'Nenhuma'}
- Modelo anterior: {last_model or 'Nenhum'}
- Resumo da última interação: {context_str or 'Início da conversa'}

MENSAGEM DO USUÁRIO:
{user_prompt}"""

    for target_model in [primary_model, fallback_model]:
        try:
            llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model=target_model,
                temperature=0.0,
                max_tokens=250,
                timeout=8.0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )

            messages = [
                {"role": "system", "content": CLASSIFIER_V2_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload_text}
            ]

            response = await llm.ainvoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)

            # Limpa possíveis blocos markdown ```json ... ```
            clean_json = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
            clean_json = re.sub(r"\s*```$", "", clean_json.strip(), flags=re.MULTILINE)

            data = json.loads(clean_json)
            cat = str(data.get("category", "")).strip().upper()
            is_followup = bool(data.get("is_followup", False))
            rewritten = data.get("rewritten_prompt")
            topic_shift = bool(data.get("topic_shift", False))

            valid_categories = {
                "DOCUMENT_RAG", "WEATHER_INFO", "REALTIME_SEARCH", "CASUAL_CHAT",
                "CODING", "LOCAL_THINKING", "HEAVY_REASONING", "FACTUAL_GENERAL",
                "VISION", "IMAGE_GENERATION", "IMAGE_EDIT", "SOCIAL_MEDIA_POST",
                "DATA_EXPORT_CHART", "SHORT_VIDEO", "MCP_FILESYSTEM", "MCP_FETCH",
                "MCP_GIT", "MCP_BRAVE_SEARCH", "MCP_DATABASE", "MCP_CODE_RUNNER"
            }

            if cat in valid_categories:
                # Se o classificador apontou topic_shift, desativa herança de contexto
                if topic_shift:
                    is_followup = False

                method = f"v2_structured_json_{target_model}"
                print(f"[Roteador Semântico V2] Decisão LLM Estruturada ({target_model}): Rota '{cat}' | Follow-up: {is_followup} | Topic Shift: {topic_shift}")
                return cat, method, clean_json, rewritten
        except Exception as e:
            print(f"[Aviso Roteador Semântico V2] Modelo {target_model} falhou na saída estruturada: {e}. Tentando próximo...")

    # Fallback heurístico resiliente final
    p_low = user_prompt.lower()
    if any(k in p_low for k in ["smb", "samba", "linux", "servidor", "windows", "powershell", "python", "script", "código"]):
        cat = "CODING"
    else:
        cat = "FACTUAL_GENERAL"

    return cat, "v2_structured_json_fallback", "{}", None


# =====================================================================
# Função Principal do Motor V2 (classify_intent_v2)
# =====================================================================
async def classify_intent_v2(
    user_prompt: str,
    messages: list,
    last_user_msg: dict,
    conversation_id: str = None,
    user_id: str = None
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Motor Classificador V2 de Nova Geração:
    1. Regras estritas de títulos e anexos de imagem (0ms).
    2. Guardrail de Mudança de Tópico (Topic Shift) para isolar contaminações de chats antigos.
    3. Roteamento Vetorial Semântico via Qdrant Cosine Similarity (5ms - 15ms).
    4. Fallback com Classificador LLM estruturado via JSON Schema rígido (sem truncamento).
    """
    has_image_now = message_has_image(last_user_msg)
    has_image_in_history = history_has_images(messages)
    is_title_request = "Provide a concise, 5-word-or-less title" in user_prompt

    # 1. Requisição de Título interna do LibreChat
    if is_title_request:
        category = "VISION" if has_image_in_history else "CASUAL_CHAT"
        return category, "title_rule", None, None

    # 2. Análise de Imagem anexa direta (Visão, Edição ou Post Social)
    if has_image_now:
        prompt_lower = user_prompt.lower()
        # Se for pedido de criação de post/publicação ou inclusão de frases/textos na imagem
        social_post_keywords = r"\b(post|postar|postg|postagem|story|stories|feed|reels|carrossel|campanha|an[uú]nc|copy|publica[cç]|instagram|insta|linkedin|facebook|twitter|tiktok|frase|frases|texto|textos|legenda|dizeres|subt[ií]tulo)\b"
        if re.search(social_post_keywords, prompt_lower):
            return "SOCIAL_MEDIA_POST", "v2_social_media_image_attachment_rule", None, None

        # Se for pedido explícito de edição de imagem
        edit_keywords = r"\b(alter[ea]|mude|troque|edite|remova|coloque|substitua|modifique|apague|fundo|background)\b"
        if re.search(edit_keywords, prompt_lower):
            return "IMAGE_EDIT", "v2_image_edit_attachment_rule", None, None

        return "VISION", "v2_vision_attachment_rule", None, None

    # 3. Recuperação de Sessão & Topic Shift Guardrail
    from classifier import get_conversation_state, is_context_dependent_prompt
    session_state = get_conversation_state(conversation_id, messages)
    last_category = session_state.get("last_category")
    last_model = session_state.get("last_model")
    is_followup = is_context_dependent_prompt(user_prompt, messages)

    # Detecção de Topic Shift: se o chat anterior era SOCIAL_MEDIA_POST e a pergunta atual não tem nada social
    if last_category == "SOCIAL_MEDIA_POST":
        social_terms = r"\b(post\w*|postg\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|legenda\w*|carrossel\w*|copy\w*|story|stories|feed|instagram|insta|linkedin|facebook|twitter|tiktok|hashtag)\b"
        if not re.search(social_terms, user_prompt.lower()):
            # Desconecta do histórico social
            last_category = None
            is_followup = False
            print(f"[Roteador Semântico V2 Topic Shift] Desconectando histórico social anterior: nova pergunta é sobre outro domínio.")

    # 4. Roteamento Semântico Vetorial Instantâneo (Qdrant)
    vector_match = await match_semantic_vector(user_prompt)
    if vector_match:
        cat, method, score, seed = vector_match
        return cat, method, None, None

    # 5. Extração de Trecho da Resposta Anterior para Contexto
    last_assistant_snippet = ""
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if isinstance(c, str):
                last_assistant_snippet = c[:300]
            elif isinstance(c, list):
                t = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                last_assistant_snippet = " ".join(t)[:300]
            break

    context_str = f"Última resposta do assistente: '{last_assistant_snippet}'" if last_assistant_snippet else "Início da conversa."

    # 6. Fallback LLM Estruturado via JSON Schema
    cat, method, raw_json, rewritten = await classify_with_structured_json(
        user_prompt=user_prompt,
        context_str=context_str,
        last_category=last_category,
        last_model=last_model
    )

    return cat, method, raw_json, rewritten
