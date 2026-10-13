# LangChain Orchestrator & Smart Multi-Provider AI Gateway (v2.0)

Gateway inteligente e autônomo de orquestração de Inteligência Artificial, construído com **FastAPI**, **LangChain**, **LiteLLM** e **Qdrant**. Compatível com a especificação da **OpenAI API** (`/v1/chat/completions`, `/v1/models`, `/v1/completions`), projetado para servir como backend de alta disponibilidade para o **LibreChat** e ferramentas avançadas de desenvolvimento de software (**OpenCode**, **VSCode / Continue**, **Cline**, **Cursor**).

---

## 1. Topologia de Infraestrutura (Homelab Proxmox)

A arquitetura opera distribuída em rede interna Gigabit (`10.100.10.0/24`):

| Nó / VM | IP e Portas | Função Principal | Serviços em Execução |
| :--- | :--- | :--- | :--- |
| **VM1** | `10.100.10.30:3080` | Interface Conversacional | **LibreChat Web UI** (MongoDB, MeiliSearch) |
| **VM2** | `10.100.10.31:8000`<br>`10.100.10.31:4000`<br>`10.100.10.31:6333`<br>`10.100.10.31:20128` | Orquestrador, RAG e Gateways | **LangChain Orchestrator** (porta `8000`)<br>**LiteLLM Proxy** (porta `4000`)<br>**Qdrant Vector DB** (porta `6333`)<br>**9Router** (porta `20128`) |
| **VM3** | `10.100.10.32:11434` | Inferência Local Offline (CPU 16 Cores, 32GB RAM) | **Ollama Engine** (`llama3.2:3b`, `qwen2.5-coder:7b`, `qwen3.5:9b`, `nomic-embed-text`) |

---

## 2. Estrutura Modular da Aplicação

O repositório adota uma **Arquitetura Modular Orientada a Serviços**, onde cada domínio de tarefa, prompt e cadeia de modelos possui isolamento funcional:

```
langchain-orchestrator/
├── config.py                 # Timezone São Paulo, URLs, chaves de API, logs e sanitizadores de mensagens
├── prompts.py                # Templates de classificação, síntese de agentes, RAG e prompts de código
├── classifier.py             # Filtros Regex Fast-Pass (0ms) e classificador semântico LLM
├── coding_service.py         # Subsistema de código multi-tier (Fast, Exec, Plan) e validação sintática
├── chat_service.py           # Pool conversacional casual (Groq OSS 120B/20B + Cloudflare Llama 3.3)
├── factual_service.py        # Pool de conhecimento enciclopédico e fatos (ComboFree_Smart)
├── reasoning_service.py      # Roteamento multi-tier de raciocínio profundo (DeepSeek R1, o3-mini, Claude)
├── image_service.py          # Motores de criação e edição in-place (FLUX.1 Kontext Pro, Grok, GPT Image)
├── social_service.py         # Subsistema de Redes Sociais, Leads, Copywriting e Imagens por Plataforma
├── finance_api_service.py    # Consultas oficiais em tempo real: Câmbio (AwesomeAPI) e BACEN SGS
├── data_export_service.py    # Geração de planilhas formatadas (.xlsx, .csv) e gráficos visuais (Seaborn)
├── feedback_service.py       # Sincronização de feedbacks negativos do MongoDB e guardrail reflexivo
├── agent_service.py          # Ferramentas em tempo real (OpenWeatherMap, SearXNG, Tavily) e auditor de fatos
├── rag_service.py            # Base corporativa e memória temporária com TTL no Qdrant
├── cache_service.py          # Cache híbrido de alta velocidade (SQLite Exato + Qdrant Semântico)
├── canonical_api_service.py  # APIs Canônicas Brasileiras (BrasilAPI: CEP, CNPJ, FIPE, IBGE)
├── video_service.py           # Criação autônoma de vídeos curtos verticais (Reels/Shorts/TikTok) com OpenAI TTS + FFmpeg
├── mcp_filesystem_service.py # MCP Filesystem: Sandbox seguro de manipulação de arquivos (/app/workspace)
├── mcp_fetch_service.py      # MCP Fetch: Extração de páginas web, HTML para Markdown e proteção SSRF
├── mcp_git_service.py        # MCP Git: Controle de versão, commits, status, log e diffs seguros no workspace
├── mcp_brave_search_service.py # MCP Brave Search: Busca web independente e resiliente via Brave Search API
├── mcp_database_service.py   # MCP Database: Consultas SQL seguras em SQLite, DuckDB e PostgreSQL
├── mcp_code_runner_service.py # MCP Code Runner: Execução segura de scripts Python e Bash em sandbox
├── domain_skills_service.py  # Habilidades Especialistas: Social Media, DevOps, Engenharia de Software
├── skills_api.py             # Router REST e Manifesto OpenAPI (/skills/openapi.json) para o LibreChat
├── main.py                   # FastAPI: endpoints REST, pipeline unificado e streaming SSE
├── docker-compose.yml        # Orquestração do Orchestrator e Qdrant
├── Dockerfile                # Imagem de produção Python 3.11-slim
├── requirements.txt          # Dependências do projeto
├── .env                      # Variáveis de ambiente e segredos locais (não versionado)
├── .env.example              # Modelo versionável de variáveis de ambiente
├── data/cache.db             # Banco SQLite do cache exato (persistido)
├── documents/                # Diretório de arquivos PDF/DOCX/TXT para ingestão RAG
├── generated_images/         # Imagens geradas/editadas e gráficos servidos estaticamente
└── logs/                     # Auditoria em JSON Lines (requests_YYYY-MM-DD.jsonl)
```

---

## 3. Catálogo Completo de Endpoints da API

O servidor expõe na porta `8000` uma API REST compatível com OpenAI e rotas de controle:

| Método | Endpoint | Descrição | Destinatário Ideal |
| :--- | :--- | :--- | :--- |
| **`POST`** | `/v1/chat/completions` | Endpoint principal de chat com roteamento dinâmico, streaming SSE e cache | **LibreChat** / Clientes Gerais |
| **`GET`** | `/v1/models` | Catálogo de modelos compatível com o schema oficial OpenAI | **IDEs** / LibreChat |
| **`POST`** | `/v1/completions` | Autocompletação de código e *Fill-In-The-Middle* (FIM) | Extensões de IDE (Autocomplete) |
| **`POST`** | `/v1/coding/completions` | Gateway dedicado para desenvolvimento de software | **OpenCode** / **VSCode** / **Cline** |
| **`POST`** | `/v1/opencode/completions` | Alias dedicado para ferramentas OpenCode | **OpenCode** |
| **`POST`** | `/v1/social/create-post` | Endpoint dedicado de criação de posts e leads (JSON estruturado) | **n8n** / **Make** / Automações |
| **`POST`** | `/v1/social/completions` | Alias OpenAI-compatible para o Agente de Redes Sociais | Clientes OpenAI / LibreChat |
| **`POST`** | `/v1/video/short` | Geração completa de vídeos curtos verticais (Reels/Shorts/TikTok) com roteiro, narração e renderização | **LibreChat** / Automações / Marketing |
| **`POST`** | `/v1/chat/completions` *(MCP_FILESYSTEM)* | Manipulação segura de arquivos e pastas no sandbox do workspace | **LibreChat** / Agentes |
| **`POST`** | `/v1/chat/completions` *(MCP_FETCH)* | Extração de páginas web e conversão segura para Markdown limpo | **LibreChat** / Agentes |
| **`POST`** | `/v1/chat/completions` *(MCP_GIT)* | Controle de versão Git seguro no workspace (status, log, diff, commit, branch) | **LibreChat** / IDEs |
| **`POST`** | `/v1/chat/completions` *(MCP_BRAVE_SEARCH)* | Busca web independente em tempo real via Brave Search API | **LibreChat** / Agentes |
| **`POST`** | `/v1/coding/fast` | Sub-rota direta para o nível **FAST** de código | Scripts rápidos e edições diretas |
| **`POST`** | `/v1/coding/exec` | Sub-rota direta para o nível **EXEC** (Intermediário / Execução) | Implementação e refatoração |
| **`POST`** | `/v1/coding/plan` | Sub-rota direta para o nível **PLAN** (Avançado / Arquitetura) | Planejamento e raciocínio profundo |
| **`POST`** | `/v1/opencode/fast` | Alias OpenCode para o nível **FAST** | Scripts rápidos e testes simples |
| **`POST`** | `/v1/opencode/exec` | Alias OpenCode para o nível **EXEC** | Criação de módulos e classes |
| **`POST`** | `/v1/opencode/plan` | Alias OpenCode para o nível **PLAN** | Arquitetura de microsserviços |
| **`GET`** | `/v1/cache/status` | Estatísticas de volume e acertos do cache híbrido | Monitoramento |
| **`POST`** | `/v1/cache/clear` | Purga total do SQLite e da coleção do Qdrant | Manutenção |
| **`GET`** | `/v1/rag/status` | Status da coleção permanente `knowledge_base` | RAG Administrativo |
| **`POST`** | `/v1/rag/ingest` | Processa e indexa arquivos da pasta `./documents` | RAG Administrativo |
| **`GET`** | `/v1/rag/user-memory/status` | Contagem de memórias temporárias ativas do usuário | Memória Temporária |
| **`POST`** | `/v1/rag/user-memory/ingest` | Ingestão manual de texto na memória temporária (TTL 30d) | Memória Temporária |
| **`GET`** | `/dashboard` | Interface Web completa com Métricas, Logs, Perfis e Painel de Administração | Navegadores Desktop/Mobile |
| **`GET`** | `/v1/admin/config` | Consulta prompts padrão ativos, guardrails e metadados descritivos | Dashboard / Automação |
| **`POST`** | `/v1/admin/config` | Salva alterações seguras de prompts e guardrails com validação | Dashboard / Automação |
| **`POST`** | `/v1/admin/config/reset` | Restaura todos os prompts e guardrails para os padrões de fábrica | Dashboard / Automação |
| **`POST`** | `/v1/admin/cache/clear` | Ação administrativa de purga total dos caches (SQLite e Qdrant) | Dashboard / Manutenção |
| **`POST`** | `/v1/admin/rag/reindex` | Ação de reindexação e reorganização da base documental RAG | Dashboard / Manutenção |
| **`POST`** | `/v1/admin/rag/prune` | Ação de purga de memórias temporárias expiradas por TTL | Dashboard / Manutenção |
| **`POST`** | `/v1/admin/feedback/sync` | Força sincronização imediata dos feedbacks de polegares do LibreChat | Dashboard / Manutenção |
| **`GET`** | `/images/{filename}` | Arquivos estáticos de imagens geradas/editadas | Renderização Markdown |

---

## 4. Subsistema de Código: Arquitetura de 3 Níveis

Para equilibrar agilidade na interface de chat e máxima sofisticação em ambientes de desenvolvimento, o subsistema de código foi dividido em 3 perfis:

```
                            ┌─────────────────────────────────┐
                            │      Requisição de Código       │
                            └────────────────┬────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             │                                                               │
    [Interface LibreChat]                                        [Ferramentas de IDE]
   (model: agent-orchestrator)                                (OpenCode, VSCode, Cline, Continue)
             │                                                               │
             ▼                                                               ▼
┌─────────────────────────┐                            ┌───────────────────────────────────────────┐
│     Sempre Nível FAST   │                            │      Nível Selecionado pela Ferramenta    │
└────────────┬────────────┘                            └─────────────────────┬─────────────────────┘
             │                                                               │
             │                                ┌──────────────────────────────┼──────────────────────────────┐
             │                                │                              │                              │
             ▼                                ▼                              ▼                              ▼
┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│       NÍVEL FAST        │      │       NÍVEL FAST        │    │       NÍVEL EXEC        │    │       NÍVEL PLAN        │
│    (Scripts Rápidos)    │      │    (Edição & Scripts)   │    │ (Interm. / Execução)    │    │ (Avançado / Arquitetura)│
├─────────────────────────┤      ├─────────────────────────┤    ├─────────────────────────┤    ├─────────────────────────┤
│ 1º Combo_Free_Coding    │      │ 1º Combo_Free_Coding    │    │ 1º Coding_DeepSeek_V3   │    │ 1º Reasoning_DeepSeek_R1│
│    (Qwen 2.5 Coder 32B) │      │    (Qwen 2.5 Coder 32B) │    │    (DeepSeek V3 671B)   │    │    (DeepSeek R1 CoT)    │
│ 2º Coding_DeepSeek_V3   │      │ 2º Coding_DeepSeek_V3   │    │ 2º Combo_Free_Coding    │    │ 2º Reasoning_O3_Mini    │
│    (DeepSeek V3 671B)   │      │    (DeepSeek V3 671B)   │    │    (Qwen 2.5 Coder 32B) │    │    (OpenAI o3-mini)     │
│ 3º Coding_Fast_Fallback │      │ 3º Coding_Fast_Fallback │    │ 3º Coding_Fast_Fallback │    │ 3º Coding_DeepSeek_V3   │
│    (GPT-4o-mini)        │      │    (GPT-4o-mini)        │    │    (GPT-4o-mini)        │    │    (Fallback Execução)  │
│ 4º qwen2.5-coder:7b     │      │ 4º qwen2.5-coder:7b     │    │ 4º qwen2.5-coder:7b     │    │ 4º qwen2.5-coder:7b     │
│    (Ollama Local)       │      │    (Ollama Local)       │    │    (Ollama Local)       │    │    (Ollama Local)       │
└─────────────────────────┘      └─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
```

### 4.1. Nível FAST (Scripts Rápidos, Regex e Automações)
* **Objetivo:** Velocidade máxima (10 a 15 segundos), custo zero ou irrisório e zero tempo de pensamento (*thinking*).
* **Uso:** Utilizado **obrigatoriamente pelo LibreChat** (mesmo se o usuário perguntar de arquitetura ou sistemas concorrentes) e em tarefas rápidas nas IDEs.
* **Modelos:** `Combo_Free_Coding` (Qwen 32B) ➔ `Coding_DeepSeek_V3` ➔ `Coding_Fast_Fallback` (GPT-4o-mini) ➔ `qwen2.5-coder:7b` Local.
* **Identificadores:** `/v1/coding/fast`, `/v1/opencode/fast`, `model: "agent-coding-fast"`.

### 4.2. Nível EXEC (Intermediário / Execução de Código em Produção)
* **Objetivo:** Máxima precisão na escrita de código, criação de módulos inteiros, classes thread-safe, tipagem rigorosa e refatoração profunda sem atrasos de CoT.
* **Uso:** Ferramentas de desenvolvimento conectadas ao orquestrador.
* **Modelos:** `Coding_DeepSeek_V3` (DeepSeek-V3 671B MoE sem reasoning tokens — líder mundial de geração direta de código) ➔ `Combo_Free_Coding` ➔ `Coding_Fast_Fallback` ➔ `qwen2.5-coder:7b` Local.
* **Identificadores:** `/v1/coding/exec`, `/v1/opencode/exec`, `model: "agent-coding-exec"`, `model: "agent-coding"`, `model: "opencode"`.

### 4.3. Nível PLAN (Avançado / Planejamento Arquitetural & Raciocínio Profundo)
* **Objetivo:** Resolução de problemas de alta complexidade algorítmica, design de sistemas distribuídos (*raft/consensus*), análise de concorrência e planejamento de projetos com cadeia de pensamento (*Chain of Thought*).
* **Uso:** Ferramentas de arquitetura e sessões de planejamento de software.
* **Modelos:** `Reasoning_DeepSeek_R1` (DeepSeek R1 CoT) ➔ `Reasoning_O3_Mini` (OpenAI o3-mini) ➔ `Coding_DeepSeek_V3` (fallback) ➔ `Combo_Free_Coding` ➔ `qwen2.5-coder:7b` Local.
* **Identificadores:** `/v1/coding/plan`, `/v1/opencode/plan`, `model: "agent-coding-plan"`, `model: "agent-coding-pro"`.

---

## 5. Subsistema de Redes Sociais, Leads e Imagens Contextualizadas (`agent-social` e `agent-social-smart`)

O orquestrador inclui um módulo especializado para criação de postagens completas de marketing, conversão de leads e posters visuais avançados: o [social_service.py](file:///home/jean/langchain-orchestrator/social_service.py) e o [social_mode_smart.py](file:///home/jean/langchain-orchestrator/social_mode_smart.py).

### 5.1. Modos de Operação do Agente Social
* **Modo FAST (`agent-social`):** Focado em copywriting persuasivo veloz, enquadramento de fotos existentes ou geração rápida de imagem temática, com sobreposição tipográfica.
* **Modo SMART (`agent-social-smart`):** Composição Multimodal de Alta Fidelidade com **`gpt-image-2.5-sunburst`** (OpenAI). Permite passar **múltiplas fotos** simultâneas (atletas, troféus, logotipos) e instruções de arte, gerando um poster esportivo/institucional completo integrado diretamente pela OpenAI.

### 5.2. Pipeline Multiagente
```
Pedido ─► [Planejador leve] ─► decide pesquisa web (Tavily) + domínio do assunto
                                         │
   Imagem anexa? ─► [Agente de Visão: ComboVision descreve a imagem]
                                         ▼
          [AGENTE 1: Pesquisa & Direção Criativa]  (DeepSeek V3; R1 se o pedido for complexo)
          └─► BRIEFING JSON: fatos verificados, vocabulário correto/evitado, tom, elementos visuais,
              imagens complementares, ícones, emojis, tipografia, paleta HEX, layout, prompt visual
                                         ▼
          [AGENTE 2: Criador]  (ComboFree_Smart ➔ DeepSeek V3 ➔ Combo_Free_Chat)
          └─► Hook, corpo, CTA, hashtags (usa apenas fatos do briefing)
                                         ▼
          Imagem: reuso do anexo do usuário (base64 ➔ URL pública local) ou geração nova (FLUX/GPT Image/Grok)
```
* **Pesquisa condicional:** só ocorre quando o pedido cita eventos, nomes próprios, datas ou marcas.
* **Anexos do LibreChat** chegam em base64; o orquestrador os salva em `/images/` para que a imagem seja renderizada no chat.
* **Reasoning adaptativo:** R1 é usado para campanhas, multi-plataforma ou pedidos longos; V3 nos demais casos.
* **Contingência:** se qualquer agente falhar, há briefing/copy mínimos e fallback de modelos em cada etapa.
* **Limitação atual:** tipografia, paleta e layout são entregues como orientação; a imagem não recebe texto sobreposto (previsto para a fase 2).

### 5.2. Mapeamento Técnico de Formato e Resoluções por Plataforma

| Plataforma | Aspect Ratio | Dimensões Ideais | Foco do Copywriting e Tom | Estratégia de Hashtags |
| :--- | :--- | :--- | :--- | :--- |
| **Instagram** | `4:5` (ou `1:1`) | `1080x1350` (Feed Retrato) | Gancho magnético inicial, storytelling dinâmico, espaçamento limpo e CTA para engajamento | 5 a 8 hashtags nichadas |
| **LinkedIn** | `1.91:1` (ou `1:1`) | `1200x627` (Banner de Artigo) | Tom profissional, liderança de pensamento (*thought leadership*), foco em negócios e dados | 3 a 5 hashtags corporativas |
| **X (Twitter)** | `16:9` | `1200x675` (Landscape) | Concisão máxima, gancho imediato de impacto e leitura em segundos | 1 a 2 hashtags estratégicas |
| **Facebook** | `1:1` | `1080x1080` (Feed Quadrado) | Tom acolhedor e comunitário, perguntas abertas estimulando comentários | 2 a 4 hashtags |
| **Reddit** | `16:9` | `1200x675` (Autêntico) | Anti-corporativês, foco em valor e solução prática, sem tom de anúncio | Sem hashtags |

### 5.3. Apresentação ao Usuário
* **Arte Visual Pronta:** Renderizada e servida localmente (`/images/...`).
* **Caixa de Texto Formatada:** Bloco de código com gancho, corpo, chamada para ação (CTA) e hashtags pronto para copiar e colar com 1 clique.
* **Decomposição Estratégica:** Explicação do gancho, estilo visual utilizado e métricas de tempo.
* **Dica de Crescimento:** Melhores horários e práticas para impulsionar a postagem no algoritmo.

---

## 6. Como Configurar em IDEs (OpenCode, VSCode, Cline, Continue)

### 5.1. OpenCode / Cline / Cursor
Configure o provedor como **OpenAI Compatible**:
* **Base URL:** `http://10.100.10.31:8000/v1`
* **API Key:** `sk-antigravity-orchestrator` (ou qualquer string não vazia)
* **Modelos Recomendados:**
  * Para desenvolvimento e refatoração: `agent-coding-exec` (ou `opencode-exec`)
  * Para planejamento de arquitetura: `agent-coding-plan` (ou `opencode-plan`)
  * Para scripts rápidos: `agent-coding-fast`

### 5.2. Continue.dev (VSCode) - `config.json`
```json
{
  "models": [
    {
      "title": "Orchestrator - Coding Exec (DeepSeek V3)",
      "provider": "openai",
      "model": "agent-coding-exec",
      "apiBase": "http://10.100.10.31:8000/v1",
      "apiKey": "sk-antigravity"
    },
    {
      "title": "Orchestrator - Coding Plan (DeepSeek R1)",
      "provider": "openai",
      "model": "agent-coding-plan",
      "apiBase": "http://10.100.10.31:8000/v1",
      "apiKey": "sk-antigravity"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Orchestrator FIM Autocomplete",
    "provider": "openai",
    "model": "qwen2.5-coder:7b",
    "apiBase": "http://10.100.10.31:8000/v1",
    "apiKey": "sk-antigravity"
  }
}
```

---

## 7. Exemplos de Uso via cURL

### 7.1. Chat Geral (LibreChat / Fast Coding Automático)
```bash
curl -X POST http://10.100.10.31:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-orchestrator",
    "messages": [{"role": "user", "content": "Crie um script em Python para calcular o hash SHA-256 de um arquivo."}],
    "stream": true
  }'
```

### 7.2. Endpoint Dedicado - Nível Execução (`/v1/coding/exec`)
```bash
curl -X POST http://10.100.10.31:8000/v1/coding/exec \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-coding-exec",
    "messages": [{"role": "user", "content": "Implemente uma classe ThreadSafeLRUCache com Lock e tipagem estática."}],
    "stream": false
  }'
```

### 7.3. Endpoint Dedicado - Nível Planejamento Arquitetural (`/v1/coding/plan`)
```bash
curl -X POST http://10.100.10.31:8000/v1/coding/plan \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agent-coding-plan",
    "messages": [{"role": "user", "content": "Projete a arquitetura de replicação de logs e eleição de líder do Raft."}],
    "stream": true
  }'
```

### 7.4. Automação de Redes Sociais & Leads (`/v1/social/create-post`)
```bash
curl -X POST http://10.100.10.31:8000/v1/social/create-post \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "5 dicas essenciais de cibersegurança e senhas fortes para empresas",
    "platform": "instagram"
  }'
```

### 7.5. Autocompletação / FIM (`/v1/completions`)
```bash
curl -X POST http://10.100.10.31:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def calcular_media(valores):\n    return",
    "max_tokens": 32
  }'
```

### 7.6. Descoberta de Modelos (`/v1/models`)
```bash
curl http://10.100.10.31:8000/v1/models
```

---

## 8. Sistema de Cache Híbrido & Proteção de Código

O orquestrador implementa uma arquitetura de cache em dois estágios:
1. **Camada 1 (Exata via SQLite - `data/cache.db`):** Hash SHA-256 do prompt normalizado com resposta em ~1ms.
2. **Camada 2 (Semântica via Qdrant - `semantic_cache`):** Embeddings com `nomic-embed-text` e limiar calibrado ($\ge 0.88$).
3. **Validador Sintático de Código (`is_code_response_complete`):**
   * Impede compulsoriamente a gravação de respostas de código que possuam blocos de código Markdown abertos sem fechamento (quantidade ímpar de ```` ````) ou comandos cortados no final (`print(`, `def `, `=`, etc.).
   * Evita a poluição do cache e respostas truncadas persistentes.
4. **Bypass Inteligente:** Consultas de notícias (`REALTIME_SEARCH`), clima (`WEATHER_INFO`), visão (`VISION`), redes sociais (`SOCIAL_MEDIA_POST`) e geração/edição de imagem (`IMAGE_GENERATION`, `IMAGE_EDIT`) nunca são cacheadas.

---

## 9. Ingestão RAG, Feeds RSS Contínuos e Memória do Usuário

O ecossistema de dados para RAG opera em três frentes complementares:

1. **Base Corporativa Permanente (`knowledge_base`):**
   * Arquivos colocados em `./documents` (.pdf, .docx, .md, .txt, .csv) são fatiados e indexados via `POST /v1/rag/ingest`.
2. **Ingestão Autônoma de Feeds RSS Contínuos (`cron_ingest_rss.py`):**
   * Executado a cada 30 minutos em background no container do orquestrador.
   * Lê o arquivo de configuração declarativa [rss_feeds.json](file:///home/jean/langchain-orchestrator/rss_feeds.json) contendo 43 feeds selecionados (fontes de nicho em IA, Tecnologia, Ciência, Saúde, Educação, subgêneros do UOL e veículos internacionais).
   * **Política de TTL Inteligente por Categoria:**
     * **45 dias:** IA, Tecnologia, Ciência, Saúde, Educação e Startups (assuntos de referência e alta durabilidade temporal).
     * **15 dias:** Noticiário geral, economia ampla e internacional.
     * **7 dias:** Política, esportes, cultura e entretenimento (alta volatilidade).
   * Realiza scraping profundo via [scraper_service.py](file:///home/jean/langchain-orchestrator/scraper_service.py) (Crawl4AI / BeautifulSoup fallback) e indexa os artigos automaticamente.
3. **Memória do Usuário & Auto-Ingestão de Buscas Web (`user_temp_memory`):**
   * Textos longos (> 1200 caracteres) enviados em conversas são fatiados e salvos com TTL de 30 dias por `user_id`.
   * **Ciclo Fechado de Busca:** Resultados profundos obtidos pelo agente `REALTIME_SEARCH` na web são injetados automaticamente na memória do usuário (TTL de 24h), permitindo reutilização instantânea em perguntas subsequentes sem re-executar crawling.
4. **Calibração de Confiança Semântica e Pergunta de Confirmação:**
   * **Score >= 0.65 (HIGH):** Alta similaridade vetorial; resposta direta e conclusiva com citação de fontes.
   * **Score 0.45 a 0.64 (MEDIUM):** Similaridade moderada; resposta acompanhada de checagem educada de alinhamento com a intenção do usuário.
   * **Score < 0.45 (LOW):** Baixa similaridade; resposta cautelosa indicando o melhor esforço e **obrigatoriamente** formulando uma pergunta de confirmação para validar se era a informação esperada ou se necessita refinamento.

---

## 10. Pipeline e Roadmap de Buscas em Tempo Real (`REALTIME_SEARCH`)

Para evitar alucinações e respostas desatualizadas, o fluxo de busca foi desenhado com múltiplas camadas de resiliência:

```
Mensagem do Usuário
       │
       ▼
[Extrator de Termos LLM] ──► Extrai palavras-chave objetivas, nomes próprios, datas e domínios citados
       │
       ▼
[SearXNG Local (Porta 8888)] ──► Metabusca agregada sem rastreamento (Google, Bing, DuckDuckGo, etc.)
       │
       ├──► [Resultados Encontrados?]
       │         │
       │         ├──► SIM: [Scraping Profundo em Paralelo dos Top-2 URLs] (Crawl4AI ➔ Markdown limpo)
       │         │           │
       │         │           └──► [Auto-Ingestão em Background no RAG] (TTL 24h)
       │         │
       │         └──► NÃO / VAZIO / TIMEOUT:
       │                   │
       │                   ▼
       │             [Fallback Tavily Search API] ──► Top 3 snippets de alta relevância + Auto-ingestão RAG
       ▼
[Enriquecimento Híbrido com RAG Local] ──► Busca simultânea em KB/Feeds RSS (vetores locais pré-catalogados)
       │
       ▼
[Avaliador de Certeza & Diretriz de Validação]
       │
       ▼
[Síntese com ComboFree_Smart] ──► Resposta assertiva com citação de fontes e checagem de alinhamento
```

## 11. Subsistema de Exportação de Dados, Planilhas e Gráficos (`DATA_EXPORT_CHART`)

O orquestrador inclui uma suíte especializada de Business Intelligence e visualização gráfica de dados:

1. **Separação de Papéis e Fidedignidade de Dados:**
   - O gerador de dados atua estritamente como ferramenta complementar de formatação e análise visual.
   - Quando o usuário pede cotações, moedas ou estatísticas do mundo real, o sistema consulta compulsoriamente fontes oficiais antes de gerar o arquivo, impedindo alucinações numéricas.
2. **APIs Canônicas Integradas (`finance_api_service.py`):**
   - **AwesomeAPI de Câmbio:** Cotações oficiais em tempo real e séries temporais históricas diárias (até 90 dias) para Dólar (`USD-BRL`), Euro (`EUR-BRL`), Bitcoin (`BTC-BRL`) e Libra (`GBP-BRL`).
   - **Banco Central do Brasil (SGS):** Séries oficiais de Selic (Série 432 / 11), IPCA Acumulado e Mensal (Série 433), CDI (Série 12) e Poupança (Série 196).
3. **Diagramação e Estilização de Planilhas:**
   - **Excel (`.xlsx` via `openpyxl`):** Cabeçalho profissional em azul escuro (`#1F497D`) com texto branco em negrito, linhas com grade sutil (`#D9D9D9`) e dimensionamento automático dinâmico da largura de cada coluna.
   - **CSV (`.csv`):** Codificação `utf-8-sig` com separador ponto e vírgula (`;`), nativamente compatível com o Microsoft Excel em português.
4. **Gráficos com Seaborn & Matplotlib:**
   - Gráficos de linha, barra e dispersão com margens dinâmicas (`set_ylim`) para evitar distorções de escala, pontos anotados com valores numéricos, sincronização automática via SCP para `10.100.10.30` e cache-busting nos links (`?t={timestamp}`).

---

## 12. Guardrails Semânticos, Auditor Reflexivo e Reescrita Contextual

Para assegurar máxima robustez em conversas contínuas e eliminar respostas erradas:

1. **Reescrita Contextual de Consultas (Query Rewriter com LLM):**
   - O classificador semântico (`Classifier_Fast`) detecta se a mensagem atual do usuário é incompleta ou anafórica (`IS_FOLLOWUP: TRUE`).
   - Se for uma frase elíptica (ex: *"Faça uma dissertação completa sobre o tema."*, *"E no Rio?"*), o modelo reconstrói o prompt autônomo completo integrando o tema anterior (ex: *"Faça uma dissertação completa sobre a Revolução Constitucionalista de 1932."*).
   - O prompt reescrito é propagado para o pipeline, impedindo que o modelo downstream alucine tópicos desconexos.
2. **Auditor Reflexivo Anti-Alucinação (`audit_response_facts`):**
   - Avalia a conformidade de dados gerados contra as fontes brutas retornadas do RAG e da busca web.
   - Corrige automaticamente datas, nomes e valores inventados antes de entregar a resposta final.
3. **Guardrail Estrito de RAG vs Mídia/Cultura Pop:**
   - O modelo é expressamente impedido de classificar séries de TV, streaming (Netflix, HBO), jogos ou fatos históricos como `DOCUMENT_RAG`, a menos que o usuário declare explicitamente ter anexado um arquivo sobre o assunto.
4. **Fallback Transparente do RAG para Web Search:**
   - Quando uma consulta é direcionada ao Qdrant mas a relevância dos documentos recuperados é baixa (`top_score < 0.64`), o sistema não responde *"não tenho a informação"*.
   - O fluxo transita automaticamente para `REALTIME_SEARCH`, busca dados frescos na Web (SearXNG + Crawl4AI + Tavily) e indexa o resultado na memória temporária do usuário.

---

## 13. Ecossistema Model Context Protocol (MCP) Integrado

O orquestrador implementa serviços modulares alinhados à especificação **Model Context Protocol (MCP)**, expondo ferramentas nativas tanto via **LangChain Tools (`@tool`)** quanto em **Schemas OpenAI Function Calling**:

### 13.1. MCP Filesystem (`mcp_filesystem_service.py`)
* **Objetivo:** Manipulação segura e estritamente confinada de arquivos e pastas dentro de `WORKSPACE_PATH` (`/app/workspace`).
* **Ferramentas Expostas:** `mcp_read_file`, `mcp_write_file`, `mcp_append_file`, `mcp_list_directory`, `mcp_create_directory`, `mcp_delete_file`, `mcp_search_files`, `mcp_get_file_info`.
* **Segurança Zero-Trust:** Resolução canônica de caminho (`Path.resolve()`), verificação estrita `relative_to()` impedindo qualquer tentativa de Path Traversal (`../`), bloqueio de deleção de raiz e limite de leitura (max 5 MB).

### 13.2. MCP Fetch (`mcp_fetch_service.py`)
* **Objetivo:** Extração, raspagem e conversão de páginas web e documentos públicos em Markdown estruturado.
* **Ferramentas Expostas:** `mcp_fetch_page`.
* **Segurança Zero-Trust (Anti-SSRF):** Validação estrita de esquema (apenas `http`/`https`), resolução de DNS com bloqueio preventivo de IPs privados (RFC 1918), loopback (`127.0.0.1`), link-local e metadados de nuvem (`169.254.169.254`). Integração com Crawl4AI para SPAs dinâmicas e fallback com síntese de IA.

### 13.3. MCP Git (`mcp_git_service.py`)
* **Objetivo:** Controle de versão completo para projetos de desenvolvimento de software dentro do workspace.
* **Ferramentas Expostas:** `mcp_git_status`, `mcp_git_diff`, `mcp_git_log`, `mcp_git_commit`, `mcp_git_branches`, `mcp_git_init`, `mcp_git_checkout`, `mcp_git_show`.
* **Segurança:** Execução de subprocesso segura sem `shell=True`, confinamento estrito no workspace, limite de tamanho de diff e suporte a commit semântico automático por linguagem natural.

### 13.4. MCP Brave Search (`mcp_brave_search_service.py`)
* **Objetivo:** Camada de busca web independente através do índice da Brave Search API.
* **Ferramentas Expostas:** `mcp_brave_web_search`.
* **Resiliência e Redundância:** Atua como terceira camada de busca web independente (SearXNG ➔ Tavily ➔ Brave Search), contornando bloqueios do Google/Bing e rate limits de terceiros.

### 13.5. MCP Database (`mcp_database_service.py`)
* **Objetivo:** Consulta e gerenciamento de bancos de dados relacionais e analíticos (SQLite, DuckDB, PostgreSQL, MongoDB).
* **Ferramentas Expostas:** `mcp_db_list_tables`, `mcp_db_describe_table`, `mcp_db_query`, `mcp_db_execute`.
* **Segurança Zero-Trust:** Modo somente-leitura por padrão (`read_only=True`), sanitização contra injeção SQL em metadados, resolução canônica de arquivos restrita a `/app/workspace` e paginação com teto rígido (máx. 200 registros por query).

### 13.6. MCP Code Runner (`mcp_code_runner_service.py`)
* **Objetivo:** Execução controlada e confinada de snippets de código Python 3, rotinas Bash e scripts existentes no workspace.
* **Ferramentas Expostas:** `mcp_run_python`, `mcp_run_bash`, `mcp_run_script_file`.
* **Segurança Zero-Trust:** Expurgamento automático de credenciais de ambiente (chaves de API sensíveis não vazam para o subprocesso), timeout estrito (padrão 15s, máx 60s) com terminação de processos zumbis, bloqueio de comandos destrutivos e teto rígido de buffer de saída (50 KB de stdout/stderr).

---

## 14. Catálogo de Skills Especialistas & Manifesto OpenAPI para o LibreChat

O orquestrador implementa uma camada de **Domain Skills Especialistas** expostas via endpoints REST padronizados (`skills_api.py`) e servidas através de um manifesto limpo pronto para importação no menu de **Actions / Skills do LibreChat**:
👉 **URL do Manifesto:** `http://10.100.10.31:8000/skills/openapi.json`

### 14.1. Vertical: Social Media & Marketing Digital
* **`POST /v1/skills/social/campaign-planner`**: Planeja estratégia de campanhas completas com ganchos magnéticos (*hooks*), calendário diário ({1..30} dias) e CTAs orientadas a conversão.
* **`POST /v1/skills/social/carousel-copywriter`**: Roteiro slide a slide para Instagram/LinkedIn ({4..10} slides) com gancho de capa, corpo didático e direcionamento de arte.
* **`POST /v1/skills/social/visual-art-director`**: Gera prompts cinemáticos ultra-detalhados para FLUX.1, FLUX.2 e Midjourney com lentes, iluminação de estúdio e color grading.
* **`POST /v1/skills/social/hashtag-seo`**: Otimizador de hashtags balanceadas (alto, médio volume e cauda longa) e diretrizes de SEO na legenda contra shadowban.

### 14.2. Vertical: DevOps, Infraestrutura de Redes e Hardware
* **`POST /v1/skills/devops/docker-cicd`**: Gera Dockerfile multi-stage enxuto e seguro (usuário não-root), `docker-compose.yml` e pipeline de CI/CD (GitHub Actions/GitLab CI).
* **`POST /v1/skills/devops/network-troubleshooter`**: Diagnóstico detalhado de problemas de rede, proxy reverso (Nginx), DNS, portas e firewalls com comandos exatos de diagnóstico (`ss`, `dig`, `tcpdump`, `iptables`).
* **`POST /v1/skills/devops/hardware-planner`**: Planejamento de capacidade de servidores (*Capacity Planning*): dimensionamento de CPU, RAM, VRAM para IA local, IOPS e storage NVMe com comparação de custos.

### 14.3. Vertical: Programação e Design de Sistemas
* **`POST /v1/skills/coding/architecture-designer`**: Desenvolve a especificação técnica de arquiteturas escaláveis com diagramas **Mermaid** renderizáveis (sequence, flow, class, erDiagram) e análise de trade-offs.
* **`POST /v1/skills/coding/security-audit`**: Auditoria estrita de código contra OWASP Top 10, injeções SQL/XSS, vazamento de chaves e código refatorado com segurança defensiva.
* **`POST /v1/skills/coding/test-suite`**: Geração de suítes de testes unitários automatizados (`pytest`, `unittest`, `jest`) com fixtures, mocks de bancos/APIs e cobertura de cenários de borda.
* **`POST /v1/skills/coding/refactor-clean-code`**: Refatoração de código legado para Clean Code, princípios SOLID, tipagem estrita (*Type Hints*) e redução de complexidade ciclomática.

---

## 15. Hub de APIs Canônicas Oficiais & Contextualização Territorial Geográfica

Para eliminar alucinações e atender consultas regulatórias, empresariais e locais brasileiras:

1. **APIs Públicas Canônicas Oficiais (`canonical_api_service.py`):**
   * **CEP e Endereços:** Consulta direta via **BrasilAPI v2** (retornando latitude/longitude, IBGE, logradouro e bairro) com fallback transparente para o **ViaCEP**.
   * **CNPJ e Receita Federal:** Consulta de dados cadastrais oficiais para CNPJs de 14 dígitos (Razão Social, Nome Fantasia, CNAE, Situação Cadastral, Capital Social, Endereço e Quadro Societário QSA) com 100% de exatidão.
   * **Tabela FIPE:** Consulta de preços médios e dados de veículos por código FIPE.
2. **Contextualização Geográfica Territorial (OpenStreetMap Nominatim):**
   * Quando o usuário faz perguntas locais (ex: *"Me dê sugestões de restaurantes em Santo André"*, *"Pizzarias em Moema"*), o agente resolve as coordenadas, microrregião e delimitações territoriais do município.
   * O prompt de síntese recebe uma **Diretriz Territorial Estrita**, garantindo que as sugestões de estabelecimentos e comércios pertençam estritamente ao município e aos bairros corretos.

---

## 16. Memória Episódica Persistente de Longo Prazo & Perfis Analíticos de Usuário

O orquestrador implementa um sistema completo de cognição de longo prazo e perfilamento comportamental ([episodic_memory_service.py](file:///home/jean/langchain-orchestrator/episodic_memory_service.py)):

1. **Perfil Analítico do Usuário (`user_profiles.json`):**
   * Rastreia preferências, distribuição de categorias e padrões de busca a cada requisição.
   * **Detecção Automática de Arquétipos:**
     * *Desenvolvedor de Software / Engenheiro:* respostas diretas com código executável.
     * *Criador de Conteúdo / Social Media:* foco em copywriting magnético e engajamento.
     * *Pesquisador Acadêmico / Estudos Densos:* dissertações estruturadas e aprofundadas.
     * *Explorador de Fatos & Atualidades:* respostas baseadas em dados em tempo real.
   * Extração de tags de tópicos frequentes e estilo ideal de resposta.

2. **Memória Episódica Permanente no Qdrant (`user_episodic_memory`):**
   * Armazenamento perpétuo (**sem expiração / sem TTL**) de fatos biográficos declarados pelo usuário (nome, localização, profissão/empresa e preferências).
   * **Injeção Transparente de Contexto:** Mesmo em conversas novas sem histórico de mensagens prévias, o orquestrador recupera as preferências do usuário via similaridade semântica e personaliza as respostas.

3. **Curadoria Automática de RAG Pessoal por Recorrência:**
   * Quando o usuário pesquisa temas alinhados aos seus tópicos frequentes, a síntese consolidada é automaticamente arquivada na coleção episódica permanente, criando uma base de conhecimento personalizada de alta velocidade.

4. **Painel na Dashboard Web:**
   * Aba dedicada **🧠 Perfis & Memória Episódica** para inspecionar métricas comportamentais, tópicos frequentes e memórias vetoriais de cada usuário cadastrado.

---

## 17. Auditoria e Telemetria

Todas as transações atendidas pelo orquestrador são registradas em disco em formato estruturado:
👉 `logs/requests_YYYY-MM-DD.jsonl`

Cada linha contém um JSON com:
* `timestamp` no fuso horário `America/Sao_Paulo` (UTC-3).
* `user_id`, `conversation_id`, `client_ip`.
* `routing_method` (`fast_pass_rule`, `llm_semantic_classifier`, `direct_model_alias`, `contextual_continuity_rule`).
* `category` e `coding_tier` (`FAST`, `EXEC`, `PLAN`).
* `target_destination` e `target_model`.
* `tools_called` (tempo de busca, URLs extraídas, score semântico, auditorias).
* `duration_seconds` e status (`success` ou falhas detalhadas).

---

## 18. Como Executar

### 18.1. Pré-requisitos
* Docker e Docker Compose instalados no host.
* Instância do Ollama acessível em `10.100.10.32:11434`.
* Instância do LiteLLM Proxy acessível em `10.100.10.31:4000`.
* Instância do SearXNG acessível em `10.100.10.31:8888`.

### 16.2. Subindo os Serviços
```bash
cd /home/jean/langchain-orchestrator
docker compose up -d
```

### 16.3. Verificando Logs e Saúde
```bash
docker logs -f langchain_orchestrator
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/rag/status
```

