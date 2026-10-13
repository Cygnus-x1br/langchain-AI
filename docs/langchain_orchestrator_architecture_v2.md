# Arquitetura e Engenharia do Orquestrador Semântico v2 (LangChain & LiteLLM Gateway)

## 1. Visão Geral e Arquitetura

O objetivo desta solução é disponibilizar um **endpoint único, autônomo e transparente** compatível com o protocolo da OpenAI (`/v1/chat/completions`) para o **LibreChat**. O orquestrador atua como um gateway semântico inteligente de última geração, responsável por:
1. Inspecionar e sanitizar payloads recebidos do frontend.
2. Identificar a modalidade da entrada (texto puro vs. imagens anexadas).
3. Classificar a intenção da pergunta com abordagem híbrida (**Fast-Pass por Regex de 0ms** + **LLM Semântico**).
4. Despachar a requisição para a camada ideal de computação:
   - **Inferência local ultrarrápida** via Ollama (VM3) para tarefas cotidianas e código.
   - **Pools de inferência em nuvem** via LiteLLM / 9Router (VM2) para fatos densos e raciocínio de ponta.
   - **Agentes com Ferramentas em Tempo Real** (LangChain Tools) para dados meteorológicos e buscas na web.
5. Auditar e registrar em disco 100% dos eventos em formato estruturado (JSON Lines) para métricas de qualidade.

---

### Topologia de Infraestrutura (Homelab Proxmox)

A infraestrutura é distribuída em nós dedicados conectados em rede interna Gigabit (10.100.10.0/24):

| Host / VM | Função Principal | Serviços em Execução | IPs / Portas |
| :--- | :--- | :--- | :--- |
| **VM1** | Interface de Usuário / Frontend | LibreChat | `10.100.10.30` |
| **VM2** | Orquestrador & Gateway Multiprovedor | FastAPI (LangChain Router), LiteLLM Proxy, 9Router | `10.100.10.31:8000` (FastAPI)<br>`10.100.10.31:4000` (LiteLLM) |
| **VM3** | Inferência Local (16 Cores CPU, 32GB RAM) | Ollama Engine | `10.100.10.32:11434` |

---

## 2. Catálogo Consolidado de Modelos e Endpoints

### 2.1. Modelos Locais (Ollama - VM3)
Persistidos na memória RAM da CPU da VM3 com `OLLAMA_MAX_LOADED_MODELS=3` e `OLLAMA_KEEP_ALIVE=-1`. Operam primordialmente para **Embeddings RAG** e como **Camada de Contingência / Fallback Local Offline** (garantindo 0ms de downtime mesmo sem internet):
* **`llama3.2:3b` (~2,5 GB RAM):** Fallback local para saudações e conversas casuais (`CASUAL_CHAT`).
* **`qwen2.5-coder:7b` (~5,0 GB RAM):** Fallback local para geração e depuração de código (`CODING`).
* **`qwen3.5:9b` (~6,5 GB RAM):** Fallback local para reflexão lógica (`LOCAL_THINKING`).
* **Embeddings locais:** `nomic-embed-text` e `bge-m3` integrados ao Qdrant para indexação e busca vetorial RAG.

### 2.2. Grupos de Modelos em Nuvem (LiteLLM / 9Router - VM2)
* **`Combo_Free_Chat` (9Router - Round Robin Gratuito):** Pool conversacional de altíssima velocidade e inteligência para `CASUAL_CHAT`. Alterna entre **Groq** (`GPT-OSS-120b`, `GPT-OSS-20b`, `qwen-3.8-27b`) gerando a centenas de tokens/segundo, e **Cloudflare Workers AI** (`Llama-3.3-70B`, `Llama-3.1-70B`, `Mistral Small 24B`).
* **`Combo_Free_Coding` (9Router - Pool Primário Gratuito de Código):** Pool especializado em engenharia de software para `CODING`. Liderado pelo **`Qwen 2.5 Coder 32B`** e **`QwQ 32B`** (Cloudflare Workers AI), com contingência no **`GPT-OSS-120b`** (Groq) e **`North Mini Code`** (OpenRouter Free).
* **`Coding_DeepSeek_V3` (LiteLLM / OpenRouter):** DeepSeek-V3 (671B MoE) sem overhead de tokens de raciocínio. Custo ínfimo ($0.14 In / $0.28 Out por 1M tokens), resposta em ~1.5s, utilizado como contingência primária do nível Fast e líder de execução direta do nível Intermediário/Execução.
* **`Coding_Fast_Fallback` (LiteLLM / OpenAI):** `gpt-4o-mini` para contingência rápida de código (~1.0s).
* **`Reasoning_DeepSeek_R1` (LiteLLM / OpenRouter):** DeepSeek R1 para raciocínio analítico e planejamento arquitetural de software profundo (*Chain of Thought*).
* **`ComboFree_Fast`:** Pool leve do 9router (`cf/@cf/meta/llama-3.1-8b-instruct-fp8-fast`) utilizado como **Classificador Semântico** do orquestrador.
* **`ComboFree_Smart`:** Pool de modelos densos gratuitos (Llama 3.3 70B, GPT-OSS 120B, Qwen 3.8 27B) para conhecimento enciclopédico (`FACTUAL_GENERAL`) e síntese RAG.
* **`FrontierReasoning` (HEAVY_REASONING - Roteamento Inteligente & Multi-Tier Econômico):**
  Arquitetura multi-tier hierarquizada por custo-benefício, complexidade e qualidade, com rotação automática (*round-robin*) e contingência em cascata (*fallback*):
  * **Tier 1 (Econômico / CoT Profundo - Prioridade Primária):**
    * **DeepSeek R1** (`openrouter/deepseek/deepseek-r1` via OpenRouter): In: $0.70 / Out: $2.50 por 1M tokens. Raciocínio matemático e analítico de fronteira a custo ultra-baixo.
    * **OpenAI o3-mini** (`openai/o3-mini` via API direta): In: $1.10 / Out: $4.40 por 1M tokens. Excelente para STEM, lógica formal e velocidade.
    * **Moonshot Kimi K2 Thinking** (`openrouter/moonshotai/kimi-k2-thinking` via OpenRouter): In: $0.60 / Out: $2.50 por 1M tokens. Cadeia de pensamento (*CoT*) encadeada de altíssima profundidade.
  * **Tier 2 (Fronteira Avançada & Contexto Longo):**
    * **Google Gemini Pro Latest / 3.7 Flash Thinking** (`gemini/gemini-pro-latest` e `gemini/gemini-3.7-flash` via Google AI Studio direto): Janela de contexto de milhões de tokens, alta capacidade de síntese e raciocínio com fallback automático contra esgotamento de taxa.
    * **OpenAI ChatGPT 6.1 Sol** (`openai/gpt-6.1-sol` via API OpenAI direta): Alta capacidade conceitual e raciocínio sofisticado nativo.
    * **Moonshot Kimi K3** (`openrouter/moonshotai/kimi-k3` via OpenRouter): In: $0.67 / Out: $14.00 por 1M tokens.
  * **Tier 3 (Fronteira Máxima & Fallback de Elite):**
    * **Anthropic Claude Sonnet 5.5** (`openrouter/anthropic/claude-sonnet-5.5` via OpenRouter): In: $2.00-$3.00 / Out: $10.00-$15.00 por 1M tokens. Preservado como fallback de segurança máxima e extrema sutileza analítica/estilística, evitando consumo desnecessário de créditos.
* **`ComboVision`:** Pool multimodal para análise e descrição detalhada de imagens (`VISION`).
* **`ImageGen_Fast` / `ImageGen_Fast_SDXL`:** Geração rápida e econômica de imagens (Google Gemini Flash Lite Image e Together AI Stable Diffusion XL 1.0).
* **`ImageGen_Pro_OpenAI` / `ImageGen_Pro_Flux` / `ImageGen_Pro`:** Geração fotorrealista Pro (OpenAI GPT Image 2.5 Flare ~$0.013/img, Together AI FLUX.2-dev ~$0.015/img e xAI Grok Imagine 2.0).
* **`ImageEdit` / `FLUX.1-kontext-pro`:** Motor especializado de edição in-place de imagens (Together AI Black Forest Labs FLUX.1 Kontext Pro com contingência no Gemini Flash Lite Image e GPT Image 2.5 Sunburst).

### 2.3. Camada de Agentes Autônomos (LangChain Tools)
* **`open_weather_map` (OpenWeatherMap API):** Extrai em tempo real temperatura, vento, sensação térmica, umidade e cobertura de nuvens.
* **`searxng_search_results` (SearXNG Metasearch):** Metabusca auto-hospedada com privacidade agregando Google, Bing e DuckDuckGo.
* **`deep_crawler` (Crawl4AI / Scraper Service):** Extração profunda de Markdown a partir de URLs retornadas pela busca na web.
* **`tavily_search_results` (Tavily Search API):** Motor de busca otimizado para LLMs que obtém notícias de hoje, resultados esportivos, cotações financeiras e dados pós-2023.
* **`cron_rss_pipeline` (Feedparser + Qdrant):** Ingestão contínua de 43 feeds RSS catalogados a cada 30 minutos com TTLs diferenciados (7, 15 e 45 dias).

---

### 2.4. Tabela Consolidada de Agentes, Endpoints e Modelos

Abaixo encontra-se a matriz completa de todos os agentes implementados no ecossistema, os endpoints REST dedicados e a cadeia de modelos/serviços que eles acessam:

| Agente / Especialidade | Endpoints Dedicados | Modelos Primários e Fallbacks Acessados | Ferramentas / Motores Integrados | Finalidade Principal |
| :--- | :--- | :--- | :--- | :--- |
| **`agent-orchestrator`**<br>(Gateway Geral) | `POST /v1/chat/completions` | Dynamic Router ➔ Classifica em 12 categorias e despacha para o pool ideal | Cache SQLite + Qdrant Semântico | Ponto único de entrada para o LibreChat |
| **`agent-coding-fast`**<br>(Código Rápido / Scripts) | `POST /v1/coding/fast`<br>`POST /v1/opencode/fast`<br>`POST /v1/chat/completions` | 1º `Combo_Free_Coding` (Qwen 32B)<br>2º `Coding_DeepSeek_V3` (671B)<br>3º `Coding_Fast_Fallback` (GPT-4o-mini)<br>4º `qwen2.5-coder:7b` (Ollama) | Validador sintático `is_code_response_complete` | Automações, scripts rápidos e conversas de código no LibreChat |
| **`agent-coding-exec`**<br>(Código Execução / Sênior) | `POST /v1/coding/exec`<br>`POST /v1/opencode/exec`<br>`POST /v1/coding/completions` | 1º `Coding_DeepSeek_V3` (671B MoE)<br>2º `Combo_Free_Coding`<br>3º `Coding_Fast_Fallback`<br>4º `qwen2.5-coder:7b` (Ollama) | Validador sintático + System prompt de engenharia | Implementação de módulos completos, refatoração em IDEs |
| **`agent-coding-plan`**<br>(Código Arquitetura & CoT) | `POST /v1/coding/plan`<br>`POST /v1/opencode/plan` | 1º `Reasoning_DeepSeek_R1` (CoT)<br>2º `Reasoning_O3_Mini` (o3-mini)<br>3º `Coding_DeepSeek_V3`<br>4º `qwen2.5-coder:7b` (Ollama) | Raciocínio profundo com pensamento explícito | Planejamento de microsserviços, algoritmos complexos |
| **`agent-autocomplete-fim`**<br>(Preenchimento Inline) | `POST /v1/completions` | 1º `qwen2.5-coder:7b` (Ollama FIM)<br>2º `Combo_Free_Coding` (fallback chat) | Suporte a tokens `<\|fim_prefix\|>` | Autocomplete inline para Continue.dev, VSCode e Cline |
| **`agent-social`**<br>(Redes Sociais & Leads) | `POST /v1/social/create-post`<br>`POST /v1/social/completions`<br>`POST /v1/chat/completions` | 1º Pesquisa: `DeepSeek V3` (ou `R1` se complexo)<br>2º Criação: `ComboFree_Smart`<br>3º Visão: `ComboVision`<br>4º Arte: `FLUX.2-dev` / `Grok` / `GPT Image` | Tavily condicional + Salva anexo base64 em `/images/` + JSON briefing | Geração de postagens completas, copy, CTA, hashtags e artes por rede |
| **`agent-social-smart`**<br>(Composição Multimodal OpenAI) | `POST /v1/social/completions`<br>`POST /v1/chat/completions`<br>(quando passa múltiplas fotos) | 1º Direção Criativa: `DeepSeek V3`<br>2º Motor Visual: `gpt-image-2.5-sunburst` (OpenAI Direct API)<br>3º Fallback: `FLUX.1-kontext-pro` (Together) | OpenAI `/v1/images/edits` com array multipart `image[]` para múltiplas fotos + Pillow | Posters de alta complexidade para redes com fotos de atletas, logos e tipografia |
| **`agent-search-realtime`**<br>(Busca Web em Tempo Real) | `POST /v1/chat/completions`<br>(via rota `REALTIME_SEARCH`) | 1º Extrator: `Classifier_Fast` (Cloudflare 8B)<br>2º Síntese: `ComboFree_Smart` | SearXNG ➔ Deep Crawling Crawl4AI ➔ Fallback Tavily ➔ Auto-ingestão RAG | Notícias recentes, IA moderna, dados esportivos, recomendações locais |
| **`agent-weather`**<br>(Clima & Meteorologia) | `POST /v1/chat/completions`<br>(via rota `WEATHER_INFO`) | Síntese: `ComboFree_Smart` | OpenWeatherMap API ➔ Fallback SearXNG / Tavily | Previsão meteorológica, vento, umidade com limpeza de cidade |
| **`agent-rag-kb`**<br>(RAG Corporativo) | `GET /v1/rag/status`<br>`POST /v1/rag/ingest`<br>`POST /v1/chat/completions` | Embeddings: `nomic-embed-text` (Ollama)<br>Síntese: `ComboFree_Smart` | Qdrant `knowledge_base` + RecursiveCharacterSplitter + Sanitização XML | Consulta a manuais, PDFs e documentos corporativos com citação |
| **`agent-rag-memory`**<br>(Memória do Usuário & RSS) | `GET /v1/rag/user-memory/status`<br>`POST /v1/rag/user-memory/ingest`<br>`DELETE /v1/rag/user-memory/clear` | Embeddings: `nomic-embed-text`<br>Síntese: `ComboFree_Smart` | Qdrant `user_temp_memory` + Cron RSS (43 feeds) + Auto-Pruning TTL | Memória temporária contextual por usuário e acervo vivo de notícias |
| **`agent-image-gen`**<br>(Geração Gráfica) | `POST /v1/chat/completions`<br>(via rota `IMAGE_GENERATION`) | Fast: `Gemini Flash Lite` / `SDXL 1.0`<br>Pro: `FLUX.2-dev` / `Grok Imagine 2.0` / `GPT Image 2.5 Flare` | Servidor estático local `/images/{filename}` | Criação fotorrealista e ilustrações artísticas |
| **`agent-image-edit`**<br>(Edição In-Place) | `POST /v1/chat/completions`<br>(via rota `IMAGE_EDIT`) | Primário: `FLUX.1-kontext-pro` (Together AI)<br>Secundário: `Gemini Flash Lite Image` | Downloader Gigabit RFC1918 + conversor Inline Base64 | Edição preservativa de fotos existentes (troca de cor, fundo, objetos) |
| **`agent-vision`**<br>(Análise Multimodal) | `POST /v1/chat/completions`<br>(via rota `VISION`) | `ComboVision` (LiteLLM) | Extração e descrição de imagens anexadas | Leitura de gráficos, OCR e descrição de fotos |
| **`agent-audio-stt`**<br>(Reconhecimento de Fala & Transcrição) | `POST /v1/audio/transcriptions` | 1º `groq/whisper-large-v3-turbo` (0 custo, ~0.4s)<br>2º `groq/whisper-large-v3`<br>3º `gemini/gemini-3.5-transcribe` (0 custo Google)<br>4º `openai/whisper-1` (Fallback de alta resiliência) | Cascata de Fallback Automática + Endpoint 100% compatível OpenAI (`multipart/form-data`) | Entrada de áudio/voz nativa no LibreChat (microfone) para transcrição instantânea em Português |
| **`agent-mcp-filesystem`**<br>(MCP Filesystem Sandbox) | `POST /v1/chat/completions`<br>(via rota `MCP_FILESYSTEM`) | Executor Direto + Validador Canônico Sandbox | `mcp_read_file`, `mcp_write_file`, `mcp_list_directory`, `mcp_delete_file`, etc. Confinamento estrito em `/app/workspace` | Manipulação e criação de arquivos e scripts no workspace com isolamento total |
| **`agent-mcp-fetch`**<br>(MCP Fetch Web Scraper) | `POST /v1/chat/completions`<br>(via rota `MCP_FETCH`) | Extração HTTP/Crawl4AI + Síntese: `ComboFree_Smart` | Filtro Anti-SSRF (bloqueio de IPs RFC 1918 e metadados) + HTML/PDF para Markdown limpo | Leitura profunda de páginas web e links enviados pelo usuário |
| **`agent-mcp-git`**<br>(MCP Git Version Control) | `POST /v1/chat/completions`<br>(via rota `MCP_GIT`) | Executor Subprocess Seguro + Validador Sandbox | `mcp_git_status`, `mcp_git_diff`, `mcp_git_log`, `mcp_git_commit`, `mcp_git_branches`, etc. | Controle de versão de projetos e repositórios locais no workspace |
| **`agent-mcp-brave`**<br>(MCP Brave Search) | `POST /v1/chat/completions`<br>(via rota `MCP_BRAVE_SEARCH`) | Brave Search API v1 + Síntese: `ComboFree_Smart` | API independente Brave + Citações Markdown estruturadas | Camada resiliente e independente de busca na web sem filtros do Google/Bing |
| **`agent-mcp-database`**<br>(MCP Database SQL) | `POST /v1/chat/completions`<br>(via rota `MCP_DATABASE`) | Motores SQLite, DuckDB, PostgreSQL, MongoDB | `mcp_db_list_tables`, `mcp_db_describe_table`, `mcp_db_query`, `mcp_db_execute`. Confinamento estrito e proteção read_only | Consulta segura e introspecção de bancos relacionais e analíticos |
| **`agent-mcp-coderunner`**<br>(MCP Code Runner Sandbox) | `POST /v1/chat/completions`<br>(via rota `MCP_CODE_RUNNER`) | Subprocess Assíncrono com Timeout e Sanitização | `mcp_run_python`, `mcp_run_bash`, `mcp_run_script_file`. Expurgamento de chaves sensíveis e teto de saída 50 KB | Execução isolada e segura de scripts Python e comandos Bash no workspace |

---

### 2.5. Roadmap Completo das Buscas no Sistema (Search Architecture Roadmap)

O subsistema de busca em tempo real e alimentação contínua do orquestrador foi estruturado nas seguintes fases evolutivas:

```
[FASE 1: CONCLUÍDA ✅]       [FASE 2: CONCLUÍDA ✅]       [FASE 3: CONCLUÍDA ✅]       [FASE 4: EM ANDAMENTO 🔄]
Busca Externa Básica          Busca Híbrida & Deep Scrape   Alimentação RSS & Confiança  Reranking & Auto-Aprendizado
- Integração Tavily API       - SearXNG Metasearch local    - 43 Feeds RSS contínuos     - Reranker Cross-Encoder
- OpenWeather com fallback    - Extrator de termos LLM      - TTL categorizado (7/15/45) - Scraping dinâmico via Playwright
- Síntese com ComboFree_Smart - Crawl4AI Deep Scraping      - Limiares HIGH/MEDIUM/LOW   - Dashboard de telemetria
                              - Auto-ingestão RAG (24h)     - Pergunta de confirmação    - Busca semântica em vídeos
```

#### Detalhamento das Etapas do Roadmap de Busca:

1. **Fase 1: Busca Básica de Ferramentas (Concluída ✅)**
   * Conexão da API Tavily e OpenWeatherMap através de wrappers LangChain.
   * Síntese direta no modelo `ComboFree_Smart` com injeção em bloco de contexto.

2. **Fase 2: Arquitetura de Busca Híbrida e Extração Profunda (Concluída ✅)**
   * **Extrator de Query LLM:** Conversão de sentenças conversacionais ("me mostre o que é...") em palavras-chave limpas, retendo domínios explícitos (`openclaw.ia`, `aquaticapaulista.org.br`).
   * **Metabusca SearXNG:** Consulta em instâncias agregadoras sem rastreamento para obter URLs de alta precisão.
   * **Deep Web Scraping (Crawl4AI):** Leitura completa dos 2 primeiros sites retornados, extraindo o Markdown limpo e substituindo snippets curtos pelo conteúdo real da página.
   * **Auto-ingestão no RAG (Loop Fechado):** As páginas raspadas na busca web são automaticamente vetorizadas na memória temporária do usuário com TTL de 24 horas para reuso imediato em turnos seguintes.

3. **Fase 3: RAG Proativo com Feeds RSS e Confiança Semântica (Concluída ✅)**
   * **Pipeline Contínuo RSS (`cron_ingest_rss.py`):** Ingestão a cada 30 minutos de 43 fontes de nicho (OpenAI Blog, MIT Tech Review, Nature, FAPESP, Drauzio Varella, Porvir, canais UOL e imprensa internacional).
   * **TTL Diferenciado por Categoria:** Retenção prolongada de 45 dias para IA, Ciência, Saúde, Tecnologia e Educação; 15 dias para notícias gerais; e 7 dias para esportes/política efêmeros.
   * **Calibração de Risco e Confirmação Interativa:** Implementação dos limiares de cosseno `HIGH` ($\ge 0.65$), `MEDIUM` ($0.45 - 0.64$) e `LOW` ($< 0.45$). Se a similaridade for baixa ou os dados da web forem inconclusivos, o orquestrador não alucina e conclui **obrigatoriamente com uma pergunta de confirmação** perguntando se a resposta atende ou se necessita refinamento.

4. **Fase 4: Reranking Neural e Cobertura Multimodal (Próximos Passos 🔄)**
   * **Cross-Encoder Reranker (`bge-reranker-large`):** Reordenamento neural dos trechos recuperados do Qdrant e da web antes de enviar ao prompt de síntese, elevando a precisão semântica de 0.50 para mais de 0.85.
   * **Crawl dinâmico em SPAs complexas:** Expansão do scraper para lidar com páginas JavaScript pesadas que demandam interação (cliques em abas de calendários).
   * **Dashboard de Métricas de Busca:** Interface visual com distribuição de taxa de acerto do SearXNG vs. Tavily, tempo médio de scraping e índices de reaproveitamento do cache de busca.

---

### 2.6. Ecossistema Model Context Protocol (MCP) Integrado

O orquestrador implementa suporte nativo ao padrão aberto **Model Context Protocol (MCP)**, expondo ferramentas de sistema padronizadas com sandbox de segurança estrito e integração bidirecional (esquemas LangChain `@tool` e OpenAI JSON Tool Calling).

```
                      ┌────────────────────────────────────────────────────────┐
                      │              LibreChat / Agente Autônomo               │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
                                  Roteamento MCP ou Tool Calling
                                                  │
        ┌────────────────┬────────────────────────┼────────────────────────┬────────────────┐
        ▼                ▼                        ▼                        ▼                ▼
┌──────────────┐ ┌──────────────┐          ┌──────────────┐         ┌──────────────┐ ┌──────────────┐
│ MCP Filesystem│ │  MCP Fetch   │          │   MCP Git    │         │ MCP Database │ │ MCP Code     │
│ (Sandbox)    │ │ (Anti-SSRF)  │          │  (Workspace) │         │ (SQL/NoSQL)  │ │ Runner (Exec)│
└───────┬──────┘ └───────┬──────┘          └───────┬──────┘         └───────┬──────┘ └───────┬──────┘
        │                │                         │                        │                │
  Leitura/Escrita  Crawl4AI &             Controle de              SQLite, DuckDB,  Python & Bash
  em /workspace    Leitura Web            Versão Local             Postgres, Mongo  em Sandbox
```

#### 2.6.1. MCP Filesystem (`mcp_filesystem_service.py`)
- **Isolamento de Segurança:** Todas as operações são confinadas estritamente ao diretório `/app/workspace` (`/home/jean/workspace` no host). Qualquer tentativa de travessia de caminho (`path traversal` como `../` ou caminhos absolutos como `/etc/passwd`) é canonicamente rejeitada com `SecurityError`.
- **Ferramentas Expostas:**
  - `mcp_read_file(path, offset, limit)`: Leitura paginada e segura de arquivos.
  - `mcp_write_file(path, content)`: Escrita atômica de conteúdo.
  - `mcp_list_directory(path)`: Listagem estruturada de diretórios.
  - `mcp_get_file_info(path)`: Metadados, tamanho e data de modificação.
  - `mcp_delete_file(path)`: Exclusão com proteção de confirmação.

#### 2.6.2. MCP Fetch (`mcp_fetch_service.py`)
- **Proteção Anti-SSRF:** Bloqueio preventivo contra ataques SSRF; recusa requisições a endereços locais RFC 1918 (10.x.x.x, 192.168.x.x, 127.x.x.x) e endpoints de metadados em nuvem (ex: 169.254.169.254).
- **Extração Inteligente:** Converte HTML e documentos web em Markdown formatado, com cache e suporte integrado ao motor Crawl4AI.
- **Ferramenta Exposta:** `mcp_fetch_url(url, raw)`: Download e estruturação de páginas e documentações online.

#### 2.6.3. MCP Git (`mcp_git_service.py`)
- **Controle de Versão Confinado:** Execução de comandos Git sem shell (`shell=False`) e validação de parâmetros contra injeção de argumentos (`argument injection`). O repositório padrão é `/app/workspace`.
- **Ferramentas Expostas:**
  - `mcp_git_status(repo_path)`: Status detalhado da working tree e staging area.
  - `mcp_git_diff(repo_path, staged, file_path)`: Diffs unificados de alterações locais ou em staged.
  - `mcp_git_log(repo_path, max_count)`: Histórico cronológico de commits (`hash`, autor, data, mensagem).
  - `mcp_git_commit(message, repo_path, add_all)`: Criação segura de commits com mensagem explicativa.
  - `mcp_git_branches(repo_path)`: Listagem e identificação da branch ativa.
  - `mcp_git_checkout(branch_or_commit, repo_path, create_branch)`: Troca de branches ou criação de novas ramificações.
  - `mcp_git_init(repo_path)`: Inicialização de novos repositórios Git no workspace.
  - `mcp_git_show(object_ref, repo_path)`: Inspeção de commits ou objetos específicos.

#### 2.6.4. MCP Brave Search (`mcp_brave_search_service.py`)
- **Indexação Web Independente:** Integração direta com a API Brave Web Search v1, fornecendo resultados limpos, neutros e desassociados dos índices convencionais de Google e Bing.
- **Resiliência Multi-Tier:** Atua como motor de busca primário na categoria `MCP_BRAVE_SEARCH` e como terceiro nível de contingência (`Tier 3 Fallback`) no pipeline de busca em tempo real (`agent_service.py`), após SearXNG e Tavily.
- **Ferramenta Exposta:** `mcp_brave_web_search(query, count, country, search_lang)`: Retorna snippets ricos, títulos, URLs e metadados estruturados.

#### 2.6.5. MCP Database (`mcp_database_service.py`)
- **Consultas Multi-Motor e Introspecção Segura:** Suporte integrado a bancos relacionais e analíticos (SQLite, DuckDB, PostgreSQL, MongoDB). Arquivos SQLite e DuckDB são canonicamente resolvidos em `/app/workspace`.
- **Proteção Read-Only por Padrão:** Por padrão, consultas de leitura (`read_only=True`) bloqueiam comandos destrutivos (`DROP`, `DELETE`, `UPDATE`, `TRUNCATE`). Paginação estrita com teto de 200 linhas por consulta.
- **Ferramentas Expostas:**
  - `mcp_db_list_tables(database, engine)`: Listagem de tabelas e views.
  - `mcp_db_describe_table(table_name, database)`: Inspeciona colunas, tipos, chaves e DDL da tabela.
  - `mcp_db_query(query, database, engine, read_only, limit)`: Execução segura de consultas SQL com formatação tabular.
  - `mcp_db_execute(statement, database)`: Execução de scripts DDL/DML no banco do workspace.

#### 2.6.6. MCP Code Runner (`mcp_code_runner_service.py`)
- **Sandbox Computacional Confinada:** Execução controlada de código Python 3 e comandos Bash diretamente em `/app/workspace`.
- **Sanitização de Credenciais e Proteção:** Expurgamento preventivo de variáveis de ambiente sensíveis (`OPENAI_API_KEY`, chaves de API, URLs de banco) para que scripts não vazem segredos do orquestrador.
- **Limites Rígidos:** Timeout de 15 segundos (máx. 60s) com terminação forçada, truncamento de saída em 50 KB e bloqueio de comandos destrutivos (`rm -rf /`, fork bombs, escrita em `/dev`).
- **Ferramentas Expostas:**
  - `mcp_run_python(code, timeout_seconds)`: Execução de código Python inline.
  - `mcp_run_bash(command, timeout_seconds)`: Execução de comando shell seguro no workspace.
  - `mcp_run_script_file(file_path, timeout_seconds)`: Execução de scripts salvos no workspace (.py, .sh).

---

### 2.7. Hub de Domain Skills Especialistas & Manifesto OpenAPI para LibreChat

O orquestrador separa as habilidades em duas camadas: primitivas de infraestrutura (**MCP**) e inteligência verticalizada de negócios (**Domain Skills** em `domain_skills_service.py` e `skills_api.py`).

Para permitir integração transparente com o menu **Actions / Skills do LibreChat**, o orquestrador expõe um endpoint dedicado de manifesto OpenAPI:
👉 `GET /skills/openapi.json`

```
                      ┌────────────────────────────────────────┐
                      │    LibreChat: Menu Actions / Skills    │
                      │   (Importa /skills/openapi.json)       │
                      └──────────────────┬─────────────────────┘
                                         │ Chamada HTTP REST
                                         ▼
                      ┌────────────────────────────────────────┐
                      │    FastAPI: skills_api.py Router       │
                      └───────┬──────────────┬──────────────┬──┘
                              │              │              │
               ┌──────────────▼───┐   ┌──────▼──────┐   ┌───▼──────────────┐
               │ 📱 Social Media  │   │ 💻 DevOps   │   │ ⚙️ Programação    │
               │ - campaign_plan  │   │ - docker_ci │   │ - arch_designer  │
               │ - carousel_writer│   │ - net_troubl│   │ - security_audit │
               │ - visual_prompts │   │ - hardware  │   │ - test_suite     │
               │ - hashtag_seo    │   │             │   │ - clean_code     │
               └──────────────────┘   └─────────────┘   └──────────────────┘
```

#### 2.7.1. Vertical: Social Media & Marketing Digital
1. **`skill_social_campaign_planner`**: Estratégia completa de campanhas, ganchos magnéticos (*hooks*), calendário editorial diário ({1..30} dias) e CTAs.
2. **`skill_carousel_copywriter`**: Roteiro de carrosséis educativos para Instagram e LinkedIn ({4..10} slides) com gancho de capa e direção de arte.
3. **`skill_visual_art_director`**: Engenharia de prompts visuais profissionais calibrados para FLUX.1, FLUX.2 e Midjourney com enquadramento e iluminação.
4. **`skill_hashtag_seo_optimizer`**: Pirâmide de alcance de hashtags (alto, médio volume e cauda longa) e diretrizes de SEO na legenda contra shadowban.

#### 2.7.2. Vertical: DevOps, Infraestrutura de Redes e Hardware
1. **`skill_docker_cicd_generator`**: Geração de Dockerfile multi-stage enxuto (usuário não-root), `docker-compose.yml` e pipeline de CI/CD (GitHub Actions/GitLab CI).
2. **`skill_network_infra_troubleshooter`**: Diagnóstico de conectividade, Nginx, DNS e firewalls com comandos de diagnóstico exatos (`ss`, `dig`, `tcpdump`, `iptables`).
3. **`skill_hardware_capacity_planner`**: Dimensionamento de servidores (*Capacity Planning*): cálculo de CPU, RAM, VRAM para IA local, IOPS e storage NVMe com comparação de custos.

#### 2.7.3. Vertical: Programação e Design de Sistemas
1. **`skill_system_architecture_designer`**: Especificação técnica de arquiteturas escaláveis com diagramas **Mermaid** renderizáveis (sequence, flow, class, erDiagram).
2. **`skill_code_security_audit`**: Auditoria defensiva contra vulnerabilidades OWASP Top 10, injeções SQL/XSS, vazamento de credenciais e código corrigido.
3. **`skill_test_suite_generator`**: Geração de suítes de testes unitários automatizados (`pytest`, `unittest`, `jest`) com fixtures, mocks e edge cases.
4. **`skill_code_refactor_clean_code`**: Refatoração para Clean Code, princípios SOLID, tipagem estrita (*Type Hints*) e redução de complexidade ciclomática.

---

## 3. Fluxo de Roteamento Híbrido

```mermaid
graph TD
    A[LibreChat: Requisição HTTP POST /v1/chat/completions] --> B{Contém imagem ou pedido de edição?}
    B -- Edição solicitada (com anexo/histórico) --> C1[Rota: IMAGE_EDIT -> Together FLUX.1 Kontext Pro / Gemini]
    B -- Análise de imagem pura anexada --> C2[Rota: VISION -> LiteLLM ComboVision]
    B -- Não possui imagens --> D{É geração interna de título do LibreChat?}
    D -- Sim (com imagem no histórico) --> C2
    D -- Sim (sem imagem no histórico) --> E[Rota: CASUAL_CHAT -> 9Router Combo_Free_Chat / Ollama llama3.2:3b Fallback]
    D -- Não --> F{Fast-Pass Regex Match?}
    
    F -- Comando de geração de imagem --> G1[Rota: IMAGE_GENERATION -> FLUX.2-dev / Grok Imagine / GPT Image 2.5 Flare]
    F -- Post para Redes Sociais / Leads --> G5[Rota: SOCIAL_MEDIA_POST -> Agente Social (FLUX.2-dev + Copywriter)]
    F -- Pergunta de Documentos / Manuais --> G2[Rota: DOCUMENT_RAG -> Qdrant + nomic-embed-text]
    F -- Padrão de Clima detectado --> G3[Rota: WEATHER_INFO -> Agente OpenWeather]
    F -- Padrão de Notícia/Realtime detectado --> G4[Rota: REALTIME_SEARCH -> Agente Tavily]
    F -- Saudação exata detectada --> E
    
    F -- Sem correspondência por Regex --> H[Classificador LLM: ComboFree_Fast com Histórico]
    H --> I{Instrução de Continuação Contextual?}
    I -- Sim ('Classifique...', 'Ordene...', 'Resuma...') --> J[Guarda Anafórica: Mantém Categoria de Origem ex: FACTUAL_GENERAL]
    I -- Não --> K[Categoria Determinada pelo Classificador]
    
    K & J -->|CODING| L[Rota: CODING -> 9Router Combo_Free_Coding / Ollama qwen2.5-coder:7b Fallback]
    K & J -->|LOCAL_THINKING| M[LiteLLM: ComboFree_Smart / Ollama qwen3.5:9b Fallback]
    K & J -->|CASUAL_CHAT| E
    K & J -->|HEAVY_REASONING| N[LiteLLM: FrontierReasoning Multi-Tier DeepSeek R1 / o3-mini / Claude Fallback]
    K & J -->|FACTUAL_GENERAL / Fallback| O[LiteLLM: ComboFree_Smart / Ollama Fallback]
    K & J -->|SOCIAL_MEDIA_POST| G5
    K & J -->|IMAGE_GENERATION| G1
    K & J -->|DOCUMENT_RAG| G2
    K & J -->|WEATHER_INFO| G3
    K & J -->|REALTIME_SEARCH| G4

    G3 -->|Erro / Cidade não encontrada| G4
    G2 & G3 & G4 --> P[Síntese Contextual RAG com ComboFree_Smart]
    P & L & M & E & N & O & G1 & C1 --> Q[Resposta SSE Stream ou JSON para o LibreChat]
```

---

## 4. Histórico de Problemas Técnicos e Soluções Implementadas

### 4.1. Limitação Multimodal Local (mllama)
* **Problema:** Tentativa de rodar `llama3.2-vision:11b` no Ollama sem GPU causava erro `unknown model architecture: 'mllama'`.
* **Solução:** Tráfego visual delegado integralmente ao pool `ComboVision` no LiteLLM.

### 4.2. Incompatibilidade de Formato de Blocos no Ollama (Erro 400)
* **Problema:** O LibreChat envia blocos complexos `[{type: 'text', text: '...'}]`. O Ollama espera texto puro.
* **Solução:** Criação da função `sanitize_messages_for_ollama()` que extrai e une os textos limpos.

### 4.3. Exigência de Alternância de Papéis no LiteLLM (Erro 500)
* **Problema:** Provedores de nuvem rejeitavam conversas com mensagens consecutivas do mesmo autor (`user -> user`).
* **Solução:** Implementação de `ensure_alternating_roles()` que mescla conteúdos de turnos idênticos mantendo alternância estrita.

### 4.4. Efeito "Sticky Vision"
* **Problema:** Uma imagem enviada no início prendia todas as perguntas seguintes na rota de visão.
* **Solução:** Apenas a mensagem de usuário mais recente avalia o gatilho visual compulsório.

### 4.5. Latência Extrema por Thinking Chain em CPU (VM3)
* **Problema:** O `qwen3.5:9b` gerava mais de 600 tokens de raciocínio interno para responder `"Bom dia"`, levando 2m37s.
* **Solução:** Criação da rota `CASUAL_CHAT` direcionada ao `llama3.2:3b`, baixando a latência de saudações para 1-3 segundos.

### 4.6. Incompatibilidade de Tool Calling Nativo (Erro 400 no 9router)
* **Problema:** Provedores gratuitos no 9router (ex: Cloudflare Workers AI) não suportam mensagens do tipo `tool` (`ToolMessage`) do formato OpenAI.
* **Solução:** Adoção de arquitetura **Dynamic RAG / Tool-Augmented Generation**: o Python executa as ferramentas diretamente e passa os dados obtidos em tempo real para o modelo em um prompt contextual limpo.

### 4.7. Extração de Cidades para OpenWeather
* **Problema:** Ao perguntar *"Qual a previsão de tempo para hoje em Santo André/SP?"*, o OpenWeather recebia a frase inteira com preposição e falhava com `"Unable to find the resource"`.
* **Solução:** Criação de `clean_location_query()`, normalizando expressões e formatos de estado (ex: `Santo André/SP` -> `Santo André, BR`) e adição de **fallback transparente para Tavily Search**.

### 4.8. Esgotamento de Cota de Imagem e URLs Locais RFC1918 na Edição (IMAGE_EDIT)
* **Problema:** Ao enviar uma foto de um urso gigante com duas crianças e solicitar *"Altere a imagem do urso para azul vibrante"*, o sistema retornava uma imagem completamente nova em vez de editar a imagem original. Duas falhas foram identificadas:
  1. O LibreChat envia links de imagem da rede local (ex: `http://10.100.10.30:3080/images/...`), inacessíveis para modelos em nuvem pública (RFC1918).
  2. O modelo gratuito Gemini Flash Lite Image atingiu a cota diária (HTTP 429), caindo no fallback que gerava uma nova imagem do zero.
* **Solução:**
  1. Implementada a função `resolve_image_to_base64()`, que faz o download da imagem dentro da rede Gigabit local e converte para `data:image/png;base64,...` inline antes de despachar para as APIs.
  2. Integração do modelo especializado **`black-forest-labs/FLUX.1-kontext-pro`** via Together AI como motor primário de edição in-place, preservando sujeitos, crianças e cenários com fidelidade contextual em ~9s, mantendo o Gemini Flash Lite como contingência.

### 4.9. Perda de Contexto em Instruções Anafóricas/Continuação e Desvio para CODING
* **Problema:** O usuário solicitava a população dos estados do Brasil e em seguida enviava a instrução genérica *"Classifique em ordem crescente"*. O orquestrador enviava o prompt isolado para o classificador LLM, que citava a hipótese de programação (`CODING`) no raciocínio. A extração ingênua capturava `CODING` e direcionava para o `qwen2.5-coder:7b` no CPU (280s de espera), respondendo que precisava dos itens para ordenar.
* **Solução:**
  1. Injeção do histórico conversacional recente (`{context}`) no prompt do classificador com regra estrita para manter a categoria do assunto em andamento (`FACTUAL_GENERAL`).
  2. Criação da função `is_context_dependent_prompt()` com detecção de comandos anafóricos e de continuação (*"classifique"*, *"ordene"*, *"resuma"*, *"traduza"*).
  3. Guarda anafórica de proteção: se for instrução de continuação sem menção explícita a termos técnicos de programação (Python, SQL, regex, script), a requisição é redirecionada automaticamente para `FACTUAL_GENERAL`.

### 4.10. Falso Cache Hit e Contaminação Semântica Cruzada
* **Problema:** A resposta errônea do `qwen2.5-coder:7b` para *"Classifique em ordem crescente"* foi armazenada no cache com prompt genérico. Como o limiar semântico do Qdrant estava em `0.74` e não havia filtro de categoria, qualquer pergunta sequencial parecida colidia com essa resposta vazia.
* **Solução:**
  1. Chave de cache contextualizada: se `is_followup = True`, a chave armazena `[Ctx: <resumo_anterior>] <prompt>`, isolando conversas de temas diferentes.
  2. Bypass de busca semântica em follow-ups: perguntas de continuação só aceitam correspondência exata (`exact_cache`) com o mesmo contexto, evitando cruzamento de vetores frouxos.
  3. Filtro estrito de categoria: tanto SQLite quanto Qdrant agora filtram `category == row.category`.
  4. Calibração do limiar do Qdrant: elevado de `0.74` para `0.88`.
  5. As categorias visuais `IMAGE_GENERATION` e `IMAGE_EDIT` foram adicionadas à lista de bypass de cache para garantir unicidade das imagens geradas.

### 4.11. Fuso Horário dos Logs e Containers (UTC 0 -> America/Sao_Paulo)
* **Problema:** Os containers Docker e as instâncias Python operavam em UTC 0, gerando logs e arquivos diários com 3 horas de adiantamento em relação a São Paulo/BR.
* **Solução:** Adicionado `TZ=America/Sao_Paulo` e volumes `/etc/localtime` e `/etc/timezone` no `docker-compose.yml`. Criação da função `get_now_local()` com `zoneinfo.ZoneInfo("America/Sao_Paulo")` em `main.py` para sincronização completa de logs e nomes de arquivos particionados.

### 4.12. Truncamento Prematuro de Código na Cloudflare Workers AI & Orçamento `max_tokens`
* **Problema:** Scripts Python médios e longos solicitados no LibreChat estavam sendo cortados abruptamente na metade (ex: `print("Selecione a operação:\nprint("`). O provedor Cloudflare Workers AI (`@cf/qwen/qwen2.5-coder-32b-instruct` via 9Router) adota um teto padrão curto de ~256 a 512 tokens quando a requisição não especifica o parâmetro `max_tokens` ou envia valor nulo.
* **Solução:** No `main.py` (fluxos de streaming e síncrono), foi injetada uma garantia mandatória de orçamento: se `max_tokens` estiver ausente ou for inferior a 4096, o orquestrador define compulsoriamente `max_tokens = 8192` para as rotas `CODING` e `HEAVY_REASONING`, e `4096` para as demais, eliminando interrupções no meio da geração.

### 4.13. Poluição de Cache com Scripts Incompletos & Validador Sintático
* **Problema:** Quando uma geração de código era cortada (por timeout ou limite de tokens), o texto incompleto gerado até então era armazenado no cache exato (SQLite) e semântico (Qdrant). Ao pedir *"O código ficou incompleto, continue"*, o sistema retornava a própria resposta cortada do cache.
* **Solução:**
  1. Criação da função `is_code_response_complete()` no `coding_service.py`: verifica se a resposta possui blocos de código Markdown abertos sem fechamento (número ímpar de ```` ````) ou se termina com operadores incompletos (`print(`, `def `, `class `, `=`, etc.). Se a resposta for detectada como incompleta, o orquestrador **aborta a gravação em cache**.
  2. Expurgadas as chaves corrompidas do banco SQLite (`cache.db`) e pontos correspondentes na coleção vetorial do Qdrant.

### 4.14. Detecção de Anos Correntes (2026) e Classificador Fast-Pass
* **Problema:** A pergunta *"Quais os melhores nadadores brasileiros na categoria Juvenil 1 em 2026 até o momento?"* foi roteada erroneamente para `FACTUAL_GENERAL` e respondeu que não possuía dados atuais de 2026. A rota falhou por duas causas: o modelo `Classifier_Fast` no LiteLLM retornou erro 400 (container não reiniciado), ativando o fallback factual; e a regex de tempo real não possuía o ano corrente.
* **Solução:**
  1. Reiniciado o container do LiteLLM e validado o modelo `Classifier_Fast` com HTTP 200.
  2. Adicionados termos temporais explícitos na regex `realtime_pattern` do [classifier.py](file:///home/jean/langchain-orchestrator/classifier.py): `\b(em\s+2026|de\s+2026|at[eé]\s+o\s+momento|atualmente|neste\s+ano|neste\s+m[eê]s|este\s+ano|ano\s+atual|ranking\s+atual|melhores\s+nadadores)\b`.

### 4.15. Isolamento de Tiers: LibreChat (Sempre Fast) vs. Endpoints Dedicados de IDE (Fast, Exec, Plan)
* **Problema:** Modelos de raciocínio profundo (*Reasoning Chain of Thought*) como o DeepSeek R1 e o3-mini demoram entre 30s e 90s gerando reflexões internas. Na interface conversacional do LibreChat, essa espera para códigos triviais gerava sensação de lentidão e timeouts de navegador.
* **Solução:**
  1. O modelo geral do LibreChat (`agent-orchestrator`) foi travado compulsoriamente no nível **`FAST`**, utilizando o `Combo_Free_Coding` (Qwen 32B), `Coding_DeepSeek_V3` e `qwen2.5-coder:7b` local, sem reasoning tokens invisíveis.
  2. Os níveis **`EXEC`** (Intermediário / Execução com DeepSeek-V3 671B) e **`PLAN`** (Avançado / Planejamento com DeepSeek R1 e o3-mini) foram isolados exclusivamente para ferramentas de desenvolvimento de software (OpenCode, VSCode, Cline, Continue) via endpoints e aliases específicos.

### 4.16. Endpoints de Descoberta (`/v1/models`) e Autocompletação / FIM (`/v1/completions`)
* **Problema:** Extensões de IDE (como Cline, Continue e OpenCode) exigem o endpoint `GET /v1/models` para listar modelos disponíveis e `POST /v1/completions` com tokens FIM (`<|fim_prefix|>`, `<|fim_suffix|>`) para preenchimento de código inline (*autocomplete*).
* **Solução:**
  1. Implementado `GET /v1/models` no orquestrador retornando o schema oficial da OpenAI com todos os agentes e aliases por nível (`agent-coding-fast`, `agent-coding-exec`, `agent-coding-plan`, etc.).
  2. Implementado `POST /v1/completions` que conecta diretamente ao `qwen2.5-coder:7b` no Ollama com suporte FIM nativo, com fallback adaptativo para chat completion via `Combo_Free_Coding`.

### 4.17. Agente Especializado em Redes Sociais, Leads e Criação de Conteúdo (`SOCIAL_MEDIA_POST`)
* **Problema:** Usuários frequentemente solicitam postagens para redes sociais (Instagram, LinkedIn, X/Twitter, Facebook, Reddit), captação de leads ou anúncios. Tratar essa demanda como `IMAGE_GENERATION` comum gerava apenas uma imagem artística sem copy, hashtags ou estratégia. Tratar como `CASUAL_CHAT` gerava apenas texto sem imagem no aspect ratio ideal.
* **Solução:**
  1. Criado o módulo dedicado [social_service.py](file:///home/jean/langchain-orchestrator/social_service.py) e a 12ª rota do classificador: `SOCIAL_MEDIA_POST`.
  2. **Pipeline de 3 Etapas Integrado:**
     - **Estrategista & Copywriter LLM:** Gera JSON estruturado com Gancho (*Hook*), Corpo persuasivo, Chamada para Ação (*CTA*), Hashtags nichadas e Dica Estratégica de Publicação.
     - **Diretor de Arte Visual:** Converte o tema em um prompt em inglês de alta fidelidade e enquadra no *aspect ratio* exato de cada plataforma:
       * **Instagram:** Retrato `4:5` (`1080x1350`) para feed ou `1:1` para carrossel.
       * **LinkedIn:** Retangular `1.91:1` (`1200x627`) focado em tom corporativo e autoridade.
       * **X / Twitter:** Paisagem `16:9` (`1200x675`) focado em concisão e impacto rápido.
       * **Facebook:** Quadrado `1:1` (`1080x1080`) com tom comunitário.
       * **Reddit:** Paisagem `16:9` autêntico (sem jargões corporativos).
     - **Motor de Imagem Multi-Provedor:** Gera a imagem contextualizada via Together AI FLUX.2-dev, OpenAI GPT Image 2.5 Flare ou Grok Imagine 2.0, salvando em disco e entregando URL estática local.
  3. **Apresentação em Markdown com Caixa de Texto:** Entrega a arte visual acompanhada de bloco de código formatado para cópia rápida com 1 clique, decomposição estratégica e dica de algoritmo.
  4. **Endpoints Dedicados:** `POST /v1/social/create-post` (para automações externas com n8n, Make, scripts) e `POST /v1/social/completions` (OpenAI format).
  5. **Descoberta:** Registrado no `GET /v1/models` como `agent-social`.

### 4.18. Evolução do Agente Social: Pipeline Multiagente e Correção de Entrega de Imagens
* **Problemas observados em produção:**
  1. A legenda do Troféu Kim Mollo (torneio de natação) usou o termo "jogar": o modelo desconhecia o evento e o domínio.
  2. Com imagem anexa, o texto era entregue mas a imagem não: o LibreChat envia anexos como `data:image/png;base64,...`, que era inserido diretamente no Markdown e não era renderizado (além de inflar logs).
  3. Um erro de digitação (`postgem`) e a palavra "imagem" desviavam o pedido para `IMAGE_GENERATION`.
* **Solução:**
  1. **Dois agentes em [social_service.py](file:///home/jean/langchain-orchestrator/social_service.py):** o *Agente de Pesquisa e Direção Criativa* (planejador leve ➔ Tavily condicional ➔ análise da imagem anexa via `ComboVision` ➔ briefing JSON) e o *Agente Criador* (copy a partir do briefing, usando só fatos verificados e o vocabulário do domínio).
### 4.19. Alucinações sobre Novas Tecnologias de IA (OpenClaw, JEV, Hermes Agent)
* **Problema:** Ao ser questionado sobre tecnologias recentes de ponta como *OpenClaw*, *JEV* e *Hermes Agent*, o agente alucinava ou buscava informações fora de contexto histórico desatualizadas, mesmo quando o usuário mencionava explicitamente o domínio `openclaw.ia`.
* **Causas identificadas:**
  1. A pergunta era tratada com busca rasa ou passava termos conversacionais inteiros ("me diga o que é...", "verifique para mim") diretamente para a query da API de busca, degradando a precisão dos motores.
  2. O crawler não acessava a estrutura profunda do site do projeto.
* **Solução:**
  1. Criação do **Extrator Inteligente de Termos de Busca** via LLM rápido (`Classifier_Fast`), convertendo perguntas conversacionais em palavras-chave assertivas e preservando compulsoriamente domínios (`openclaw.ia`) e nomes próprios.
  2. Integração do **SearXNG** com fallback no **Tavily**.
  3. Adicionada extração profunda em paralelo (**Deep Web Scraping** via Crawl4AI) nos top 2 URLs retornados, substituindo meros resumos de metatags pelo conteúdo textual completo e real da documentação oficial da ferramenta.
  4. Auto-ingestão em background do conteúdo raspado na memória temporária vetorial do usuário (`ttl_days=1`).

### 4.20. Confusão Geográfica Local (Restaurantes em Santo André vs. São Paulo)
* **Problema:** Ao solicitar *"bons restaurantes em Santo André"*, o orquestrador classificava a pergunta como factual ou fazia buscas rasas que misturavam Shopping Iguatemi e bairros de São Paulo (capital).
* **Solução:**
  1. Adicionada regra Fast-Pass e reforço no prompt do classificador semântico para que intenções de recomendações locais geográficas (*"restaurantes em..."*, *"onde comer em..."*, *"hotéis em..."*) caiam compulsoriamente em `REALTIME_SEARCH`.
  2. O prompt de síntese do agente foi instruído com restrição estrita de delimitação geográfica: se o usuário pediu uma cidade específica da Região Metropolitana (ex: Santo André - ABC Paulista), estabelecimentos fora do município ou genéricos de shoppings da capital não são admitidos.

### 4.21. Falha de Extração em Calendários de Domínios Específicos (Aquática Paulista / Troféu Kim Mollo 2026)
* **Problema:** O usuário pediu a data de um campeonato no site `aquaticapaulista.org.br`, mas a busca web comum trazia dados de anos anteriores por não conseguir ler o conteúdo dinâmico da página do calendário.
* **Solução:**
  1. O extrator de busca identifica menções a URLs e websites específicos e gera queries filtradas de domínio (`site:aquaticapaulista.org.br calendário 2026`).
  2. O módulo [scraper_service.py](file:///home/jean/langchain-orchestrator/scraper_service.py) acessa diretamente o link retornado e renderiza o HTML via Crawl4AI / BeautifulSoup, extraindo tabelas de datas e enviando os dados brutos ao prompt de síntese.

### 4.22. Margem de Risco Semântico e Pergunta de Confirmação com Limiar de Confiança
* **Problema:** Quando o RAG ou a busca web retornavam similaridade matemática média ou baixa (ex: coeficiente de cosseno ~0.5093), o modelo apresentava a resposta com tom de certeza absoluta, induzindo o usuário ao erro caso houvesse ambiguidade na entidade pesquisada.
* **Solução:**
  1. Criada a métrica de calibração semântica em 3 faixas no `execute_rag_flow()` e diretrizes correspondentes no [prompts.py](file:///home/jean/langchain-orchestrator/prompts.py):
     * **Score >= 0.65 (`HIGH`):** Alta correlação factual, resposta assertiva e direta com citações de fonte.
     * **Score entre 0.45 e 0.64 (`MEDIUM`):** Correlação moderada; o modelo apresenta os dados mas inclui confirmação ao final se atende ao esperado.
     * **Score < 0.45 (`LOW`) ou Busca Web Vazia:** Alto risco de erro; o modelo expressa limitação, não inventa informações e formula **obrigatoriamente** uma pergunta de confirmação checando se era aquela a resposta esperada e oferecendo refinamento com novas referências.

### 4.23. Alimentação Contínua do RAG com Feeds RSS e TTL por Categoria
* **Problema:** Manter o RAG atualizado apenas por ingestão manual ou buscas sob demanda deixava o sistema "cego" a lançamentos diários e novidades que ainda não foram pesquisadas pelo usuário. Além disso, manter todas as notícias com o mesmo tempo de vida sobrecarregava o banco vetorial com notícias efêmeras (esportes, política) e descartava precocemente avanços de IA e ciência que demoram semanas para serem consultados.
* **Solução:**
  1. Criação do serviço [cron_ingest_rss.py](file:///home/jean/langchain-orchestrator/cron_ingest_rss.py) e arquivo declarativo [rss_feeds.json](file:///home/jean/langchain-orchestrator/rss_feeds.json) contendo 43 fontes (nicho em IA, engenharia de software, ciência, saúde, educação, subgêneros do UOL e agências internacionais).
  2. Implementação do cálculo de **TTL Diferenciado por Categoria**:
     * **45 dias:** IA, Tecnologia, Ciência, Saúde, Educação e Startups (alta durabilidade e catálogo de referência).
     * **15 dias:** Notícias gerais, economia e internacional.
     * **7 dias:** Política, esportes, mercados diários, cultura e entretenimento (alta volatilidade).
  3. A ingestão executa scraping profundo dos artigos via Crawl4AI e armazena na memória com expiração programada e auto-pruning.

---

## 5. Sistema de Auditoria e Logs Estruturados

Todas as requisições atendidas pelo orquestrador são registradas em formato **JSON Lines (`.jsonl`)** particionadas por data no fuso de São Paulo (`America/Sao_Paulo` - UTC-3) no diretório `./logs/` (montado via volume Docker no host):
👉 `logs/requests_YYYY-MM-DD.jsonl`

### Esquema do Registro de Log:
```json
{
  "timestamp": "2026-10-05T05:50:37.122931-03:00",
  "client_ip": "172.20.0.1",
  "user_id": "user_172_20_0_1",
  "user_email": "usuario@empresa.com",
  "conversation_id": "default_session",
  "prompt": "Altere a imagem do urso para azul vibrante.",
  "has_image": true,
  "is_followup": false,
  "routing_method": "image_edit_attachment_rule",
  "raw_classifier_output": null,
  "category": "IMAGE_EDIT",
  "stream": false,
  "auto_ingest_memory": null,
  "target_destination": "image_flow_executor",
  "target_model": "MultiProvider_ImageGen",
  "tools_called": [
    {
      "action": "image_edit",
      "model": "black-forest-labs/FLUX.1-kontext-pro",
      "provider": "Together AI",
      "status": "success",
      "duration_seconds": 9.81,
      "image_url": "http://10.100.10.31:8000/images/edit_20261005_055045_319b69.png"
    }
  ],
  "status": "success",
  "response_preview": "![Imagem Editada](http://10.100.10.31:8000/images/edit_20261005_055045_319b69.png)...",
  "duration_seconds": 9.812
}
```

---

## 6. TODO: Roadmap de Próximas Implementações

Abaixo estão listadas as próximas funcionalidades e melhorias necessárias para o ecossistema, divididas por prioridade técnica.

---

### 🟢 PRIORIDADE 0: Segurança e Higienização Pré-GitHub (CONCLUÍDO ✅)

Todas as medidas de segurança e isolamento foram executadas e validadas:

- [x] **1. Migração de Segredos para Arquivo `.env` (CONCLUÍDO):**
  - Todas as chaves literais (`TAVILY_API_KEY`, `OPENWEATHERMAP_API_KEY`, `LITELLM_API_KEY`) foram removidas do `docker-compose.yml` e isoladas no arquivo local `.env`.
  - Criado o arquivo modelo `.env.example` sem dados reais para versionamento seguro.
  - O `docker-compose.yml` foi reconfigurado com a diretiva `env_file: - .env` e porta parametrizada `${PORT:-8000}:8000`.

- [x] **2. Configuração de `.gitignore` Robusto (CONCLUÍDO):**
  - Criado `.gitignore` protegendo: `.env`, `.env.*` (exceto `.env.example`), arquivos de log `logs/*.jsonl`, caches Python `__pycache__/`, diretórios `.backup/`, `.venv/` e arquivos de IDE.

- [x] **3. Parametrização de Endereços de IP Internos (CONCLUÍDO):**
  - Os endereços do Proxmox (`10.100.10.31` e `10.100.10.32`) foram movidos para as variáveis `OLLAMA_BASE_URL` e `LITELLM_BASE_URL` no `.env`.
  - O código-fonte em `main.py` foi limpo para usar apenas `http://localhost:...` como fallback padrão de desenvolvimento.

- [x] **4. Auditoria e Varredura de Histórico / Diretório (CONCLUÍDO):**
  - Varredura por Regex (`grep -rnE`) executada em todo o repositório confirmando ausência de chaves de API expostas ou IPs privados hardcoded nos arquivos versionáveis.
  - Arquivos de backup legados (`main.py.v0.*`) foram isolados na pasta `.backup/` ignorada pelo Git.
  - Caches `__pycache__` foram limpos.


---

### 🟢 PRIORIDADE 1: Sistema de RAG (Retrieval-Augmented Generation) (CONCLUÍDO ✅)

A arquitetura desacoplada de RAG local com Qdrant foi implementada e testada com sucesso:

- [x] **1. Escolha e Deploy do Vector Database Dedicado (CONCLUÍDO):**
  - Adicionado o container `langchain_qdrant` (`qdrant/qdrant:latest`) isolado no `docker-compose.yml`, consumindo apenas ~50 MB de RAM.
  - Persistência mapeada no diretório `./qdrant_data:/qdrant/storage` e painel web nativo exposto em `http://10.100.10.31:6333/dashboard`.
  - Desacoplamento total garantido: o ciclo de vida do RAG não depende do LiteLLM ou de outros serviços.

- [x] **2. Pipeline de Embeddings Local e Ingestão de Documentos (CONCLUÍDO):**
  - Implementado o serviço `rag_service.py` utilizando o modelo `nomic-embed-text` no Ollama da VM3 (leve, rápido e com zero custo de tokens).
  - Suporte a múltiplos formatos de arquivos (`.pdf`, `.docx`, `.md`, `.txt`, `.csv`) através de fatiamento inteligente com `RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)`.
  - Mapeado o diretório `./documents:/app/documents` para recepção de arquivos de conhecimento.
  - Endpoints de controle criados no orquestrador:
    * `GET /v1/rag/status`: Informa a conectividade com o Qdrant e contagem de vetores indexados.
    * `POST /v1/rag/ingest`: Lê os arquivos em `/app/documents` e indexa automaticamente no banco vetorial.

- [x] **3. Rota Semântica `DOCUMENT_RAG` e Citação de Fontes (CONCLUÍDO):**
  - Adicionadas regras rápidas por Regex (*"segundo meus documentos"*, *"no manual"*, *"conforme os arquivos"*) e suporte no classificador semântico.
  - O fluxo recupera os trechos mais relevantes por similaridade de cosseno e injeta no modelo com instrução estrita de citar o nome dos arquivos e páginas consultadas.
  - Registrado com telemetria completa no sistema de logs `logs/requests_*.jsonl`.

- [x] **4. Sanitização Ativa de Documentos & Prompt Injection Guardrails (CONCLUÍDO):**
  - Implementada a função `sanitize_document_content()` no `rag_service.py` que bloqueia extensões perigosas (`.exe`, `.sh`, `.bat`, etc.).
  - Remoção automática de caracteres ocultos de largura zero (*Zero-Width Characters*) usados para ofuscação (`\u200B`, `\uFEFF`, etc.).
  - Filtro e desativação de tags ativas executáveis (`<script>`, `<iframe>`, `<object>`, `<embed>`).
  - Neutralização de padrões de injeção de prompt (*"ignore previous instructions"*, *"system override"*, *"you are now DAN/developer mode"*).
  - Delimitação XML estrita (`<documentos_brutos>...</documentos_brutos>`) com cláusula de sistema de dados passivos no prompt de síntese.

- [x] **5. Memória Temporária do Usuário com TTL (30 Dias) e Isolamento Multi-inquilino (CONCLUÍDO):**
  - Criada a coleção `user_temp_memory` no Qdrant separada da `knowledge_base` permanente.
  - Identificação automática do usuário a partir dos headers HTTP (`X-User-ID`, `X-User-Email`) ou payload OpenAI (`user`), isolando 100% dos dados para que um usuário não acesse os livros/arquivos de outro.
  - **Auto-indexação inteligente:** Textos e livros longos (> 1200 caracteres) enviados no chat são automaticamente sanitizados, fatiados em chunks de 800 caracteres e indexados com TTL de 30 dias (`expires_at`), deduplicados por hash SHA-256.
  - **Busca Unificada e Retomada de Contexto:** Consultas dias depois (ex: *"faça uma descrição dos personagens do livro que te mandei"*) recuperam instantaneamente os trechos arquivados no Qdrant em milissegundos sem necessidade de reenviar o livro.
  - **Auto-Pruning:** Expiração e limpeza automática de pontos com `expires_at < timestamp atual` no Qdrant sem necessidade de rotinas externas de manutenção.
  - Endpoints REST dedicados: `GET /v1/rag/user-memory/status`, `POST /v1/rag/user-memory/ingest` e `DELETE /v1/rag/user-memory/clear`.



---

### 🟢 PRIORIDADE 2: Geração e Edição de Imagens Multi-Tier (CONCLUÍDO ✅)

A arquitetura completa de geração textual e edição in-place de imagens foi implementada, integrada e validada:

- [x] **1. Roteamento Especializado `IMAGE_GENERATION` e `IMAGE_EDIT` (CONCLUÍDO):**
  - **Fast-Pass Regex:** Detecção imediata de intenções de criação (*"crie uma imagem"*, *"desenhe"*, *"gere uma foto"*, *"imagem de..."*) e edição (*"altere a imagem"*, *"mude a cor"*, *"substitua o fundo"*).
  - **Identificação de Anexos Visuais:** Se a requisição contiver imagem na mensagem atual ou recente combinada com pedido de modificação, direciona compulsoriamente para `IMAGE_EDIT`.

- [x] **2. Catálogo Multi-Tier de Modelos de Imagem (CONCLUÍDO):**
  - **Tier Fast (Econômico / ~1.8s a 3s):**
    - Google Gemini Flash Lite Image (`gemini/gemini-3.1-flash-lite-image` via LiteLLM) - Custo zero.
    - Together AI Stable Diffusion XL (`together_ai/stabilityai/stable-diffusion-xl-base-1.0`) - Contingência rápida (~$0.002/img).
  - **Tier Pro (Fotorrealismo / Alta Fidelidade / 4K / Estúdio):**
    - Together AI FLUX.2-dev (`together_ai/black-forest-labs/FLUX.2-dev`) - Modelo topo de linha da Black Forest Labs.
    - xAI Grok Imagine 2.0 (`xai/grok-imagine-image-2.0` via LiteLLM) - Balanceamento dinâmico em round-robin com FLUX.2-dev.
    - OpenAI GPT Image 2.5 Flare (`openai/gpt-image-2.5-flare` via `ImageGen_Pro_OpenAI`) - Substituto moderno e de alto custo-benefício do DALL-E 3 (~$0.013/img em 1024x1024 medium vs $0.040 do DALL-E 3 Standard).
    - OpenAI GPT Image 2.5 Sunburst (`openai/gpt-image-2.5-sunburst` via `ImageGen_Pro_Sunburst`) - Modelo especializado em edição e máxima fidelidade referencial.

- [x] **3. Motor Especializado de Edição In-Place (`IMAGE_EDIT`) (CONCLUÍDO):**
  - **Provedor Primário:** **`black-forest-labs/FLUX.1-kontext-pro`** (Together AI). Desenvolvido especificamente para edição preservativa (*in-context image-to-image*), mantendo sujeitos, crianças e cenários inalterados e alterando apenas o elemento solicitado em ~9s.
  - **Provedor Secundário:** Google Gemini Flash Lite Image via LiteLLM `ImageEdit`.
  - **Resolução Automática Base64:** A função `resolve_image_to_base64()` intercepta URLs internas do LibreChat (`http://10.100.10.30:3080/images/...`), faz download interno na rede Gigabit e converte em `data:image/png;base64,...`, contornando a restrição de redes privadas (RFC1918) de APIs externas.

- [x] **4. Servidor Estático de Imagens e Entrega Markdown (CONCLUÍDO):**
  - O orquestrador salva os binários gerados/baixados no diretório persistido `./generated_images` (`/app/generated_images`), servido estaticamente em `http://10.100.10.31:8000/images/{filename}`.
  - A resposta é entregue instantaneamente ao LibreChat com formato markdown padrão: `![Imagem Gerada/Editada](http://10.100.10.31:8000/images/...)` acompanhado de metadados ricos (provedor, modelo, tempo de criação, prompt e fidelidade).

---

### 🟢 PRIORIDADE 3: Subsistema Avançado de Código (Multi-Tier & IDE Gateway) (CONCLUÍDO ✅)

A rota `CODING` e os serviços de desenvolvimento foram totalmente reformulados em uma arquitetura de múltiplos níveis (multi-tier), separando a interface do LibreChat de ambientes profissionais de IDE:

- [x] **1. Isolamento de Tiers por Perfil de Uso (CONCLUÍDO):**
  - **Interface LibreChat (`agent-orchestrator`):** Travada compulsoriamente no nível **`FAST`**. Mesmo que o usuário pergunte sobre arquitetura ou sistemas concorrentes complexos, a resposta é entregue em 10-15s sem tokens de reflexão (*thinking traces*) e sem custo.
  - **Nível FAST (Quick / Scripts):** `Combo_Free_Coding` (Qwen 2.5 Coder 32B) ➔ `Coding_DeepSeek_V3` (DeepSeek-V3 671B MoE) ➔ `Coding_Fast_Fallback` (GPT-4o-mini) ➔ `qwen2.5-coder:7b` (Ollama Local 100% offline).
  - **Nível EXEC (Intermediário / Execução para OpenCode/IDEs):** `Coding_DeepSeek_V3` (DeepSeek-V3 671B MoE sem reasoning overhead, líder de execução) ➔ `Combo_Free_Coding` ➔ `Coding_Fast_Fallback` ➔ `qwen2.5-coder:7b` Local.
  - **Nível PLAN (Avançado / Planejamento Arquitetural):** `Reasoning_DeepSeek_R1` (DeepSeek R1 com CoT profundo) ➔ `Reasoning_O3_Mini` (o3-mini) ➔ `Coding_DeepSeek_V3` ➔ `Combo_Free_Coding` ➔ `qwen2.5-coder:7b` Local.

- [x] **2. Injeção de System Prompts Especializados de Engenharia de Software (CONCLUÍDO):**
  - Definidos `CODING_FAST_PROMPT`, `CODING_EXEC_PROMPT` e `CODING_PLAN_PROMPT` em `prompts.py`.
  - Exigência estrita de código 100% completo, sem placeholders (`// TODO`, `pass`), com tipagem estática, tratamento defensivo de erros (`try/except`), docstrings e retomada contextual sem repetição de imports.

- [x] **3. Endpoints Dedicados para Ferramentas de Desenvolvimento e IDEs (CONCLUÍDO):**
  - Sub-rotas diretas: `POST /v1/coding/fast`, `POST /v1/coding/exec`, `POST /v1/coding/plan` e seus equivalentes `/v1/opencode/*`.
  - Endpoint padrão OpenAI: `GET /v1/models` para autodescoberta por ferramentas como OpenCode, VSCode (Continue), Cline e Cursor.
  - Endpoint de Autocompletação: `POST /v1/completions` com suporte nativo a preenchimento no meio do código (*Fill-In-The-Middle - FIM*) via Ollama `qwen2.5-coder:7b`.

- [x] **4. Filtro de Integridade e Proteção contra Cache Incompleto (CONCLUÍDO):**
  - Função `is_code_response_complete()` no fluxo de gravação de cache: impede que códigos cortados por limite de tokens ou falhas sejam persistidos no SQLite ou no Qdrant.


---

### 🔵 PRIORIDADE 4: Manipulação e Análise de Vídeos Curtos

- [ ] **1. Análise de Vídeos Recebidos (Video Understanding):**
  - Quando o usuário anexar um clipe curto (MP4/WebM de até 60s), o orquestrador extrai keyframes (quadros-chave) via `ffmpeg` / `opencv-python` em intervalos de 1s.
  - Envia os frames selecionados como uma sequência de imagens para o `ComboVision` ou Gemini 2.5 Flash (que possui suporte nativo a vídeo via API).
- [ ] **2. Geração de Vídeos (Text-to-Video):**
  - Conexão com APIs externas (Luma Dream Machine, Kling AI, Runway Gen-3 via OpenRouter).
- [ ] **3. Dúvidas Técnicas a Avaliar:**
  - *Dúvida 1:* A interface do LibreChat tem suporte nativo para upload de arquivos de vídeo e player embutido?
  - *Dúvida 2:* Custo computacional e de largura de banda: a extração de frames por CPU na VM2 pode causar picos de consumo durante chats simultâneos.

---

### 🟣 PRIORIDADE 5: Otimizações de Engenharia e Cache

- [x] **1. Cache Híbrido de Respostas - SQLite Exato + Qdrant Semântico (CONCLUÍDO ✅):**
  - **Camada 1 (Exata via SQLite):** Base persistida em `/app/data/cache.db`. Normaliza espaços/pontuação e calcula SHA-256. Responde em 1 a 2 milissegundos sem invocar IA.
  - **Camada 2 (Semântica via Qdrant):** Coleção `semantic_cache` com embeddings `nomic-embed-text` e limiar rigoroso calibrado para português ($\ge 0.88$).
  - **Isolamento de Chaves Contextuais:** Para perguntas de continuação/anafóricas (`is_followup = True`), a chave armazena `[Ctx: <resumo_assistente>] <prompt>`, impedindo colisões de instruções genéricas (*"Classifique em ordem crescente"*) entre conversas distintas.
  - **Bypass de Busca Semântica em Follow-ups:** Consultas sequenciais só utilizam correspondência exata para evitar contaminações por vetores frouxos.
  - **Filtro Estrito por Categoria:** SQLite e Qdrant filtram compulsoriamente por `category`, garantindo que uma entrada de `CODING` nunca responda a `FACTUAL_GENERAL`.
  - **TTL de Expiração & Limpeza:** Padrão de 7 dias com auto-pruning.
  - **Bypass Inteligente:** Consultas meteorológicas (`WEATHER_INFO`), notícias (`REALTIME_SEARCH`), visão (`VISION`), geração (`IMAGE_GENERATION`) e edição (`IMAGE_EDIT`) nunca são congeladas em cache para garantir dados frescos e unicidade gráfica.
  - **Endpoints de Gestão:** `GET /v1/cache/status` e `POST /v1/cache/clear`.
- [ ] **2. Painel Web de Monitoramento de Logs:**
  - Criar um mini dashboard (Streamlit ou rota `/dashboard` no FastAPI) que leia os arquivos `logs/requests_*.jsonl` e exiba gráficos de:
    - Distribuição de rotas (quais modelos são mais acionados).
    - Latência média por modelo.
    - Taxa de sucesso e erros por backend.
    - Consumo estimado de tokens.

---

### 🟡 PRIORIDADE 6: Arquitetura Modular de Serviços e Endpoints Dedicados (CONCLUÍDO ✅)

A base de código monolítica original (`main.py` de ~1.615 linhas) foi completamente refatorada em uma **Arquitetura Modular Orientada a Serviços**, onde cada domínio técnico, prompt e cadeia de modelos possui responsabilidade única e isolada:

```
langchain-orchestrator/
├── config.py                 # Timezone São Paulo, URLs, chaves, logs e sanitizadores de mensagens
├── prompts.py                # Templates de classificação, síntese de agentes e RAG
├── classifier.py             # Filtros Regex Fast-Pass (0ms) e roteador semântico LLM
├── coding_service.py         # Pool de desenvolvimento de software (9Router Qwen Coder + Ollama)
├── chat_service.py           # Pool conversacional casual (Groq OSS 120B/20B + Cloudflare Llama 3.3)
├── factual_service.py        # Pool de conhecimento enciclopédico e fatos (ComboFree_Smart)
├── reasoning_service.py      # Rotação inteligente multi-tier de raciocínio profundo (DeepSeek R1/o3-mini/Claude)
├── image_service.py          # Motores de criação e edição in-place (FLUX.1 Kontext Pro, Grok, GPT Image)
├── agent_service.py          # Integração de ferramentas externas (OpenWeather, Tavily) e síntese contextual
├── rag_service.py            # Base de conhecimento e memória temporária TTL em Qdrant
├── cache_service.py          # Cache híbrido persistido (SQLite Exato + Qdrant Semântico)
├── main.py                   # Orquestrador FastAPI enxuto, rotas e pipeline unificado de inferência
└── docker-compose.yml        # Mapeamento e montagem dos módulos no container de produção
```

#### Endpoints Dedicados para Ferramentas de Desenvolvimento e IDEs:
- **`POST /v1/coding/completions` e `POST /v1/opencode/completions`:**
  - Gateway de entrada para ferramentas externas (OpenCode, VSCode, Cline, Continue).
  - Roteia dinamicamente para o tier desejado com base no modelo solicitado (`agent-coding-fast`, `agent-coding-exec`, `agent-coding-plan`).
- **Sub-rotas Explícitas por Nível:**
  - `POST /v1/coding/fast` e `/v1/opencode/fast`: Força o nível Fast.
  - `POST /v1/coding/exec` e `/v1/opencode/exec`: Força o nível Intermediário/Execução.
  - `POST /v1/coding/plan` e `/v1/opencode/plan`: Força o nível Avançado/Planejamento.
- **`POST /v1/completions` (FIM Autocomplete):**
  - Suporte ao preenchimento de código inline (*Fill-In-The-Middle*) usando `<|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|>`, executado nativamente pelo `qwen2.5-coder:7b` no Ollama.
- **`GET /v1/models` (Descoberta OpenAI):**
  - Retorna o catálogo oficial com os modelos `agent-orchestrator`, `agent-coding-fast`, `agent-coding-exec`, `agent-coding-plan`, `opencode-*`, etc.

#### Continuidade de Sessão e Herança Contextual de Rotas (CONCLUÍDO ✅):
- **Rastreamento de Sessão por `conversation_id`:** O orquestrador armazena o histórico do turno anterior (`last_category`, `last_model`, `last_prompt`, `last_assistant_snippet`).
- **Regra de Continuidade Contextual (0ms):** Quando o usuário faz perguntas de acompanhamento relativas (ex: *"E para Senador?"*, *"E amanhã?"*, *"E no Rio?"*, *"Explique melhor"*):
  - A rota anterior é compulsoriamente mantida se o tema for `REALTIME_SEARCH`, `WEATHER_INFO`, `DOCUMENT_RAG`, `CODING`, `HEAVY_REASONING`, `FACTUAL_GENERAL` ou `DATA_EXPORT_CHART`.
- **Contextualização Automática de Consultas de Ferramentas:**
  - Em `REALTIME_SEARCH`, perguntas de acompanhamento curtas são automaticamente fundidas com a pergunta anterior para consulta no Tavily/SearXNG.
- **Proteção do Classificador LLM com Cloudflare 8B:** O modelo `Classifier_Fast` foi apontado para o `cf/@cf/meta/llama-3.1-8b-instruct-fp8-fast`, eliminando erros 400 de modelos instáveis e garantindo respostas de classificação em menos de 1 segundo com contingência local.

---

### 🟡 PRIORIDADE 7: Exportador de Dados, Planilhas e Gráficos Visuais (`DATA_EXPORT_CHART`) (CONCLUÍDO ✅)

Implementado subsistema completo de Business Intelligence e visualização de dados orientado a fatos reais:

1. **Separação de Papéis (Dados Factuais vs Ferramenta de Diagramação):**
   - O gerador de planilhas e gráficos atua estritamente como **ferramenta complementar de pós-processamento visual**.
   - Quando o usuário solicita tabelas, planilhas ou gráficos sobre dados reais (dólar, euro, inflação, ações, esportes, etc.), o orquestrador consulta compulsoriamente os dados em tempo real antes de desenhar a planilha, impedindo alucinações numéricas.
2. **Integração de APIs Canônicas de Dados Financeiros e Econômicos (`finance_api_service.py`):**
   - **AwesomeAPI de Câmbio:** Cotações oficiais e séries temporais diárias históricas de até 90 dias para Dólar (`USD-BRL`), Euro (`EUR-BRL`), Bitcoin (`BTC-BRL`) e Libra (`GBP-BRL`).
   - **Banco Central do Brasil (API SGS / `api.bcb.gov.br`):** Séries oficiais da Meta Selic (Série 432 / 11), IPCA Acumulado e Mensal (Série 433), Taxa CDI (Série 12) e Poupança (Série 196).
3. **Diagramação e Estilização Profissional de Planilhas (`openpyxl` e CSV):**
   - **Excel (`.xlsx`):** Cabeçalhos corporativos em azul escuro (`#1F497D`) com texto branco em negrito, bordas sutis (`#D9D9D9`) e dimensionamento automático dinâmico da largura de cada coluna.
   - **CSV (`.csv`):** Formatação padronizada para o mercado brasileiro com separador de ponto e vírgula (`;`) e codificação `utf-8-sig` (compatível nativamente com Excel sem corromper acentuação).
4. **Gerador de Gráficos Visuais (`Seaborn` + `Matplotlib`):**
   - Geração de gráficos de linha, barras e dispersão com margens proporcionais dinâmicas nos eixos X e Y (`set_ylim`), evitando achatamento de dados com variações decimais sutis.
   - Anotações numéricas diretas nos pontos de dados para legibilidade máxima.
   - Sincronização automática via SCP para a VM1 (`10.100.10.30`) e entrega via links públicos com cache-busting (`?t={timestamp}`).

---

### 🟡 PRIORIDADE 8: Guardrails de RAG, Auditor Reflexivo Anti-Alucinação e Reescrita de Consultas (CONCLUÍDO ✅)

1. **Auditor Reflexivo Anti-Alucinação (`audit_response_facts`):**
   - Camada de verificação factual executada via `Classifier_Fast` (Llama 3.1 8B, ~0.8s) que confronta o rascunho de resposta preliminar diretamente com os trechos brutos recuperados (do RAG ou da Web).
   - Se detectar números, datas ou entidades inventadas não respaldadas pelas fontes, o auditor reescreve a resposta corrigindo a inconsistência antes do envio ao usuário.
2. **Reescrita Contextual de Prompts Incompletos (Query Rewriter com LLM):**
   - O classificador semântico avalia a completude do prompt do usuário (`IS_FOLLOWUP: TRUE/FALSE`).
   - Se o usuário enviar instruções elípticas ou relativas (ex: *"Faça uma dissertação completa sobre o tema."*, *"E no Rio?"*, *"Estava me referindo ao desenho"*), o modelo reconstitui o prompt completo incorporando a entidade da rodada anterior (ex: *"Faça uma dissertação completa sobre a Revolução Constitucionalista de 1932."*).
   - O prompt reescrito é propagado para todo o pipeline downstream, impedindo buscas vazias no Qdrant e alucinações desconexas.
3. **Guardrail Estrito de RAG vs Mídia/Cultura Pop:**
   - A rota `DOCUMENT_RAG` foi blindada: perguntas sobre filmes, séries de streaming (Netflix, HBO), jogos ou fatos históricos NUNCA são classificadas como `DOCUMENT_RAG` a menos que o usuário declare explicitamente ter anexado um arquivo sobre o tema.
4. **Fallback Transparente de RAG para Web Search:**
   - Se uma busca for direcionada ao Qdrant mas a confiança semântica dos documentos recuperados for insuficiente (`top_score < 0.64`), o sistema não responde *"não tenho a informação"* e não disserta sobre memórias temporárias aleatórias.
   - O orquestrador aciona automaticamente a rota `REALTIME_SEARCH` (SearXNG + Crawl4AI + Tavily), responde com dados atualizados da web e indexa o resultado na memória temporária do usuário em segundo plano.

---

### 🟡 PRIORIDADE 9: Hub de APIs Canônicas Públicas & Contextualização Territorial Geográfica (CONCLUÍDO ✅)

Implementado o módulo canônico público [canonical_api_service.py](file:///home/jean/langchain-orchestrator/canonical_api_service.py) e o enriquecimento de contexto territorial geográfico no agente:

1. **APIs Canônicas Oficiais Brasileiras (Precisão Determinística 100%):**
   - **CEP & Endereços (`fetch_cep_location`):**
     - Consulta direta na **BrasilAPI v2** com coordenadas de latitude e longitude, bairro, cidade, estado e código IBGE.
     - Fallback automático para o **ViaCEP** em caso de indisponibilidade.
     - Identificação instantânea por Regex Fast-Pass (0ms) no classificador para padrões como `09010-010`.
   - **CNPJ & Dados Cadastrais da Receita Federal (`fetch_cnpj_data`):**
     - Consulta oficial na **BrasilAPI / Receita Federal** para qualquer CNPJ de 14 dígitos (formatado ou não).
     - Retorna Razão Social, Nome Fantasia, Situação Cadastral, Data de Abertura, CNAE Principal, Natureza Jurídica, Capital Social, Endereço completo e Quadro de Sócios (QSA).
     - Resposta rápida (~1.0s) com 0% de alucinação de LLM.
   - **Tabela FIPE de Veículos (`fetch_fipe_vehicle_price`):**
     - Consulta oficial de preços e séries históricas de veículos na **BrasilAPI / Tabela FIPE**.
     - Identificação por código FIPE (ex: `004123-8` ou menções à Tabela FIPE).

2. **Memória de Localização & Contextualização Geográfica Territorial:**
   - **Mapeamento Territorial via OpenStreetMap (Nominatim):**
     - Função `resolve_geographic_context(location_name)` que resolve limites municipais, microrregiões, bairros e coordenadas GPS precisas (Lat/Lon).
   - **Injeção de Contexto Territorial em Buscas Locais:**
     - Quando o usuário faz perguntas de comércios ou conveniências locais (ex: *"Me dê sugestões de restaurantes em Santo André"*, *"Onde comer em Moema"*), o agente:
       1. Extrai a cidade/bairro da consulta através de normalização sintática.
       2. Valida as coordenadas e delimitações geográficas do município via OpenStreetMap.
       3. Injeta um bloco estruturado de **DIRETRIZ DE LOCALIZAÇÃO GEOGRÁFICA** no prompt de síntese do LLM.
       4. O modelo sintetiza recomendações de estabelecimentos situados estritamente dentro daquele município/bairro, detalhando os endereços e bairros de cada opção.
   - **Proteção de Latência (Scraping Bounded):**
     - O web crawler profundo (`async_searx_search`) possui timeout estrito de 4.5s por URL com `asyncio.wait_for`, garantindo que sites externos lentos não atrasem as buscas locais.





---

### 🟡 PRIORIDADE 10: Memória Episódica Persistente de Longo Prazo, Perfis Comportamentais & Curadoria de RAG Recorrente (CONCLUÍDO ✅)

Implementado o subsistema episodic_memory_service.py e a camada analítica de usuários:

1. **Perfis Comportamentais & Arquétipos de Usuário (user_profiles.json):**
   - Rastreamento contínuo das preferências de cada user_id e user_email.
   - **Detecção Automática de Arquétipos:**
     - Desenvolvedor de Software / Engenheiro: respostas técnicas, concisas e código executável.
     - Criador de Conteúdo / Social Media: tom persuasivo, engajador com hooks e copies.
     - Pesquisador Acadêmico / Estudos Densos: dissertações estruturadas, analíticas e fundamentadas.
     - Explorador de Fatos & Atualidades: síntese factual baseada em fontes recentes.
   - Extração de tópicos frequentes e termos recorrentes por usuário.

2. **Memória Episódica Permanente no Qdrant (user_episodic_memory):**
   - Coleção vetorial permanente (**sem expiração / sem TTL**), indexada por user_id e metadados.
   - **Extração Heurística Instantânea de Fatos Biográficos:**
     - Detecta nome (*"meu nome é..."*), cidade/residência (*"moro em..."*), profissão/cargo (*"trabalho como..."*) e preferências explícitas.
   - **Injeção Transparente de Contexto de Longo Prazo:**
     - A cada nova interação (mesmo sem histórico na sessão!), o orquestrador recupera as memórias episódicas do usuário e personaliza a resposta com seu nome, localização e contexto de carreira.

3. **Curadoria Automática de RAG Pessoal por Recorrência:**
   - Função auto_curate_frequent_queries: quando um usuário pesquisa assuntos alinhados aos seus tópicos frequentes, a síntese consolidada é automaticamente arquivada na coleção episódica permanente, acelerando turnos futuros e abastecendo RAGs individuais.

4. **Painel de Monitoramento na Dashboard Web (dashboard.html):**
   - Nova aba dedicada: **🧠 Perfis & Memória Episódica**.
   - Lista completa de usuários cadastrados com visualização de arquétipos, histórico de acessos, distribuição de rotas, tags de tópicos frequentes e inspeção de memórias permanentes com pontuação vetorial.

---

### 🟢 PRIORIDADE 11: Painel Operacional de Ações, Prompts Dinâmicos & Gestão Segura de Guardrails (CONCLUÍDO ✅)

Implementada a infraestrutura completa de governança e administração em tempo de execução via Dashboard Web e API REST (`admin_config_service.py`), permitindo manutenção ativa do sistema sem necessidade de rebuild de containers e com garantia estrita de resiliência e estabilidade:

1. **Ações Operacionais Rápidas (One-Click Operations):**
   - **Limpeza Geral de Cache (`POST /v1/admin/cache/clear`):** Purga instantânea do SQLite (cache exato) e da coleção Qdrant `semantic_cache`.
   - **Reorganização / Reindexação RAG (`POST /v1/admin/rag/reindex`):** Dispara releitura de arquivos em `/app/documents` recalculando embeddings vetoriais na coleção `knowledge_base`.
   - **Purga de Memória Expirada (`POST /v1/admin/rag/prune`):** Remove registros temporários do Qdrant com TTL vencido (`expires_at < now`).
   - **Sincronização de Feedbacks (`POST /v1/admin/feedback/sync`):** Puxa imediatamente avaliações de polegares (positivo/negativo) do LibreChat MongoDB para alimentar a memória reflexiva.

2. **Gestão Segura de Prompts Padrão:**
   - **Templates Gerenciáveis:** Classificador Semântico (13 rotas), Síntese RAG, Síntese de Agente em Tempo Real, Auditor Reflexivo Anti-Alucinação, Níveis de Código (Fast, Exec, Plan) e Prompts de Redes Sociais (Planejador, Pesquisador, Criador).
   - **Mecanismo Fallback Seguro (*Zero Breakage*):** O sistema utiliza compulsoriamente os prompts nativos imutáveis de `prompts.py` caso um campo seja deixado vazio, omitido ou se ocorrer qualquer erro de formatação em tempo de execução (`.format()`).

3. **Configuração e Limiares de Guardrails Numéricos:**
   - **Limiares de Confiança RAG:** `rag_confidence_threshold_high` (padrão 0.65) e `rag_confidence_threshold_medium` (padrão 0.50).
   - **Fallback Transparente para Web Search:** `rag_fallback_web_threshold` (padrão 0.64) — aciona busca web quando a base local não possui relevância suficiente.
   - **Limiares do Auditor Reflexivo:** `fact_auditor_min_sources_length` (padrão 30 chars) e `fact_auditor_timeout_seconds` (padrão 8.0s).
   - **Retenção & Scraping:** `user_memory_default_ttl_days` (30 dias), `deep_scrape_timeout_seconds` (6.0s) e `search_engine_max_results` (4 resultados).
   - **Validação de Faixas:** Valores numéricos recebidos via API são estritamente delimitados a faixas seguras antes da persistência em `/app/data/runtime_config.json`.

4. **Nova Aba na Dashboard Web (`dashboard.html`):**
   - **⚙️ Administração & Guardrails:** Interface visual responsiva com cartões de ações (com confirmação de segurança via modal), chips de seleção rápida de prompts, editor com destaque de variáveis dinâmicas, sliders numéricos com visualização em tempo real e botão para **Restaurar Padrões de Fábrica**.

---

### 🟣 PRIORIDADE 12: Motor Autônomo de Vídeos Curtos Verticais (Reels / Shorts / TikTok) com OpenAI TTS & Renderização FFmpeg (CONCLUÍDO ✅)

Implementado o módulo dedicado `video_service.py` e integrado ao pipeline orquestrado (`SHORT_VIDEO` como 14ª categoria do sistema):

1. **Roteirização Cinematográfica & Storyboard Dinâmico:**
   - Prompt do Diretor Executivo de Vídeos Verticais (`SHORT_VIDEO_DIRECTOR_PROMPT`): estrutura rígida para alta retenção em formato 9:16 (1080x1920), com gancho forte nos primeiros 3 segundos, ritmo dinâmico e chamada para ação (*CTA*).
   - Divide a narrativa em 2 a 4 cenas curtas sequenciais (4 a 6 segundos cada), com duração total de 12 a 25 segundos.
   - Gera prompts visuais descritivos em inglês fotorrealista e frases curtas de legenda para sobreposição na tela.

2. **Locução de Voz Ultra-Realista via OpenAI TTS:**
   - Integração direta com a OpenAI Audio API (`tts-1` com voz `nova` em velocidade 1.05x).
   - Voz humana fluida, sincronizada ao texto exato do roteiro, sem desvios robóticos.

3. **Geração Visual das Cenas com Fallback Inteligente:**
   - Geração de imagens verticais fotorrealistas via **Google Gemini Flash Lite Image**, **OpenAI GPT Image 2.5 Flare** e **Together AI SDXL/FLUX**.
   - Fallback de gradiente estético moderno em alta resolução via Pillow para continuidade garantida.

4. **Composição e Renderização de Vídeo com FFmpeg:**
   - Renderização local acelerada com **FFmpeg 7.1.5** dentro do container do orquestrador.
   - Aplicação de efeito de movimento contínuo cinematográfico (*Ken Burns / zoompan*).
   - Sobreposição de faixa escura translúcida e legendas estilizadas centralizadas na tela.
   - Concatenação automática e muxing de áudio H.264/AAC no padrão 1080x1920 vertical.

5. **Disponibilização e Integração com LibreChat:**
   - O arquivo MP4 gerado é salvo em `/app/generated_images` e sincronizado com o LibreChat VM1 (`10.100.10.30`).
   - Retorno imediato no chat com **Player HTML5 `<video>` embutido**, link de download direto em MP4, roteiro detalhado e descrição das cenas.
   - Rota dedicada de API: `POST /v1/video/short` e roteamento semântico automático via `/v1/chat/completions`.

