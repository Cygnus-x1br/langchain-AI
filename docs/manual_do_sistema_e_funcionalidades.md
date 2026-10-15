# Manual Geral de Funcionalidades, Arquitetura e Endpoints do Sistema (LibreChat & Orquestrador IA)

> **Documento Oficial de Conhecimento do Sistema**  
> **Destinação:** Base de Conhecimento Permanente (RAG) do Orquestrador Multi-Agente.  
> **Objetivo:** Fornecer suporte detalhado a todas as dúvidas de usuários sobre capacidades do chat, comandos, ferramentas e configurações de integração técnica (ex.: OpenCode, Cline, VS Code).

---

## 1. Visão Geral do Sistema

O sistema é um **Orquestrador Inteligente Multi-Agente & RAG Corporativo** de alta performance, conectado como backend de IA do **LibreChat**. Ele conta com classificação semântica instantânea (Modo Vetorial Qdrant + Fallback Estruturado LLM), cascata resiliente de modelos (Nuvem, Provedores Rápidos e Fallback Local Offline no Ollama) e uma suíte completa de ferramentas utilitárias.

### Infraestrutura do Homelab (Nós e Endereçamento)
O cluster Proxmox opera na rede `10.100.10.0/24`:
- **Nó de Interface (VM1 - `10.100.10.30`):**
  - Roda o frontend web do **LibreChat** e agentes do OpenClaw.
- **Nó de Orquestração e RAG (VM2 - `10.100.10.31`):**
  - **FastAPI Smart Router (Orquestrador Principal):** Porta `8000` (URL base: `http://10.100.10.31:8000` ou público `https://ai.jmarc.com.br`).
  - **Qdrant Vector Database:** Porta `6333` (Dashboard web nativo e coleções vetoriais `knowledge_base`, `semantic_routes_v2`, `user_temp_memory`).
  - **LiteLLM Proxy:** Porta `4000` (Gateway para modelos OpenAI, DeepSeek, Claude, Google Gemini).
  - **9Router:** Porta `20128` (Roteamento balanceado de modelos abertos e Cloudflare).
- **Nó de Inferência Local (VM3 - `10.100.10.32`):**
  - **Ollama Engine:** Porta `11434` (Resiliente a quedas de internet).
  - Modelos locais ativos: `llama3.2:3b` (chat rápido), `qwen2.5-coder:7b` (programação), `qwen3.5:9b` (raciocínio local), `nomic-embed-text` (embeddings vetoriais).

---

## 2. O Que Posso Fazer Neste Sistema? (Catálogo de Capacidades)

Caso o usuário pergunte *"O que posso fazer nesse sistema?"*, *"Quais as funcionalidades disponíveis no LibreChat?"* ou *"O que este chat sabe fazer?"*, o sistema oferece um ecossistema abrangente de 21 capacidades integradas:

### 1. Conversação Geral, Escrita e Explicações Factuais
- Respostas sobre história, ciência, literatura, filosofia e conceitos enciclopédicos.
- Redação de textos, dissertações, e-mails, relatórios executivos e resumos de qualquer tema.
- **Modelos:** Gemini Flash, GPT-4o-mini, Combo_Free_Chat e LLaMA local.

### 2. Pesquisa em Tempo Real na Web e Dados de 2026
- Consultas factuais atualizadas do mundo real, notícias recentes de 2026, eventos esportivos e apurações.
- Consulta de cotações financeiras em tempo real (Dólar, Euro, Bitcoin, Selic, IPCA, CDI).
- Consulta pública de dados oficiais de CEP, CNPJ de empresas e Tabela FIPE de veículos.
- **Exemplos de prompt:** *"Qual a cotação do dólar e do bitcoin hoje?"*, *"Consulte o CNPJ 00.000.000/0001-91"*, *"Quais as últimas notícias de tecnologia?"*.

### 3. Previsão do Tempo e Clima
- Previsão meteorológica detalhada, temperatura atual, sensação térmica, umidade e probabilidade de chuva para qualquer cidade do Brasil e do mundo.
- **Exemplos de prompt:** *"Previsão do tempo para São Paulo hoje"*, *"Vai chover em Curitiba amanhã?"*.

### 4. Engenharia de Software, Programação e DevOps
- Escrita, depuração e refatoração de código em Python, JavaScript, TypeScript, SQL, Bash, Go, Rust, C++, Java, PHP e HTML/CSS.
- Criação de scripts de automação, pipelines Docker/Docker-Compose e CI/CD.
- Configuração de servidores Linux, infraestrutura de redes, permissões (chmod/chown/systemctl) e compartilhamento de arquivos em rede local (servidores SMB, Samba, NFS).
- **Endpoints dedicados para IDEs** (OpenCode, Cline, VS Code).
- **Modelos:** Qwen 2.5 Coder 32B, DeepSeek V3 (671B MoE) e Qwen Coder 7B local.

### 5. Raciocínio Profundo, Lógica e Matemática Avançada
- Resolução de problemas matemáticos complexos (cálculo diferencial/integral, álgebra linear, probabilidade, física teórica) e charadas lógicas com cadeia de pensamento (CoT).
- **Modelos:** DeepSeek R1, OpenAI o3-mini e Qwen 3.5 9B local.

### 6. Geração de Documentos em PDF Formatado (PDF_EXPORT)
- **Conversão de qualquer resumo, dissertação ou resposta Markdown anterior em um PDF diagramado profissionalmente** com padrão executivo A4, margens elegantes, tabelas zebradas, caixas de destaque azuis, código formatado e paginação dinâmica (`Página X de Y`).
- Link direto para download gerado em segundos.
- **Exemplos de prompt:**
  * *"Gere um PDF desse resumo"*
  * *"Converta o texto anterior em PDF"*
  * *"Faça um resumo sobre a Revolução Industrial e anexe em PDF para download"*
- **Endpoint direto:** `POST /v1/skills/export/markdown-to-pdf`.

### 7. Criação de Posts e Gestão de Redes Sociais (Social Media Studio v3)
- Geração de publicações completas para **Instagram** (Feed, Reels, Carrossel, Stories), **LinkedIn**, **Facebook**, **X (Twitter)** e **TikTok**.
- Criação de copy persuasiva com título chamativo (headline/gancho), texto de engajamento, chamadas para ação (CTA) e hashtags estratégicas.
- **Composição Visual Multimodal Integrada:**
  - Gera automaticamente artes com imagens conceituais de alta qualidade (DALL-E 3, Flux 1.1 Pro).
  - Aceita envio de fotos próprias pelo usuário, compondo a imagem final com o texto exato desejado.
  - **Revisão Inteligente:** Quando você pede para ajustar uma imagem ou frase já gerada, o orquestrador preserva a coerência visual acionando prioritariamente o modelo multimodal de alta fidelidade (GPT-2.5-sunburst).
  - **Guardrail de Frases:** Extrai estritamente a frase desejada caso o usuário a coloque entre aspas ou comandos diretos, evitando poluição visual com o texto do prompt.

### 8. Geração e Edição Artística de Imagens
- Criação de ilustrações 3D, arte digital, fotos realistas e renderizações conceituais a partir de prompts descritivos.
- Edição de imagens existentes (troca de fundo, ajuste de iluminação, remoção de objetos indesejados).
- **Modelos:** DALL-E 3, Flux 1.1 Pro, Imagen 3, Recraft v3.

### 9. Exportação de Planilhas Excel, CSV e Gráficos Estatísticos
- Conversão de dados tabulares e relatórios em arquivos para download:
  - Planilhas `.xlsx` (Excel) com estilos formatados.
  - Arquivos `.csv` delimitados.
  - Gráficos estatísticos gerados via Seaborn/Matplotlib (barras, linhas, pizza, dispersão) e exibidos inline com link de download.
- **Exemplos de prompt:** *"Gere uma planilha Excel com as 10 maiores empresas de tecnologia"*, *"Plote um gráfico comparativo da inflação dos últimos anos"*.

### 10. Consulta a Documentos e RAG Permanente / Temporário
- **Base Permanente (/app/documents):** Consulta a manuais internos, guias de rede e documentações arquivadas na coleção `knowledge_base` do Qdrant.
- **Memória Temporária de Textos/Livros:** Leitura e análise profunda de livros em PDF, contratos, relatórios técnicos ou romances anexados na conversa pelo usuário.

### 11. Suíte de Ferramentas MCP (Model Context Protocol & Sandbox)
- **MCP Filesystem:** Listagem, leitura, criação e edição de arquivos dentro da sandbox de workspace do servidor (`/app/workspace`).
- **MCP Fetch:** Extração e leitura de páginas web e artigos via URL direta informada pelo usuário.
- **MCP Git:** Execução de rotinas de versionamento no workspace (`git status`, `git diff`, `git log`, `git commit`, `git branch`).
- **MCP Database:** Inspeção de esquemas e execução de consultas SQL analíticas em bancos locais SQLite e DuckDB.
- **MCP Code Runner:** Execução direta de snippets de código Python ou comandos de terminal Bash em ambiente isolado, capturando stdout e stderr.
- **MCP Brave Search:** Pesquisa web independente através da API Brave.

### 12. Reconhecimento de Fala (STT) e Voz no Chat
- Envio de áudios gravados no LibreChat com transcrição automática por cascata de alta precisão (Groq Whisper $\rightarrow$ Gemini Flash $\rightarrow$ OpenAI Whisper-1 oficial).

### 13. Memória Episódica Persistente e Personalização
- O assistente memoriza fatos biográficos declarados pelo usuário (nome, profissão, empresa, preferências de resposta e tecnologias favoritas) e aplica esse contexto sutilmente nas respostas futuras.
- Possui circuito fechado de feedback: avaliações negativas no LibreChat (thumbs down) alimentam guardrails em tempo real.

---

## 3. Conexão de IDEs: OpenCode, Cline, Continue, Cursor e VS Code

### "Qual o endpoint para conectar o OpenCode ao sistema e como faço a configuração?"

Para integrar o **OpenCode**, **Cline**, **Continue**, **Cursor** ou extensões do **VS Code** ao orquestrador inteligente, utilize as informações abaixo:

### Endpoints Disponíveis
O orquestrador disponibiliza rotas dedicadas de alta velocidade com otimização para código e suporte a streaming (SSE):

| Rota / Endpoint | Descrição / Finalidade | Modelos Principais |
| :--- | :--- | :--- |
| `http://10.100.10.31:8000/v1/coding/completions` | **Endpoint Principal Recomendado para IDEs** (Seleção automática do tier ideal por complexidade) | DeepSeek V3, Qwen 2.5 Coder 32B |
| `http://10.100.10.31:8000/v1/opencode/completions` | Alias direto para ferramentas OpenCode | DeepSeek V3, Qwen 2.5 Coder 32B |
| `http://10.100.10.31:8000/v1/coding/fast` | Tier 1: Edições rápidas de arquivos, pequenas funções e scripts de automação (0 ms overhead) | Qwen 2.5 Coder 32B, Ollama Local |
| `http://10.100.10.31:8000/v1/coding/exec` | Tier 2: Implementação de módulos completos, refatoração de classes e testes | DeepSeek V3 671B MoE, Qwen Coder |
| `http://10.100.10.31:8000/v1/coding/plan` | Tier 3: Planejamento arquitetural, desenho de sistemas complexos e raciocínio CoT | DeepSeek R1, OpenAI o3-mini |
| `http://10.100.10.31:8000/v1/chat/completions` | Endpoint universal compatível com o padrão OpenAI | Orquestrador Geral |

> **Nota de Acesso Externo:** Se você estiver acessando de fora da rede local através do túnel público, substitua `http://10.100.10.31:8000` por `https://ai.jmarc.com.br`.

---

### Passo a Passo de Configuração por Ferramenta

#### 1. OpenCode
No arquivo de configuração de provedores do OpenCode (`settings.json` ou interface do plugin):
- **Provider:** `OpenAI-Compatible` (ou Custom Provider)
- **Base URL:** `http://10.100.10.31:8000/v1` (ou sub-rota: `http://10.100.10.31:8000/v1/coding`)
- **API Key:** Qualquer string (ex.: `sk-homelab-orchestrator` ou `none`).
- **Model Name:** `agent-coding` (ou `agent-coding-fast`, `agent-coding-exec`, `agent-coding-plan`).
- **Stream:** `true` habilitado.

#### 2. Cline (Extensão do VS Code)
Nas configurações da extensão Cline no VS Code:
1. Abra as configurações do Cline (ícone de engrenagem).
2. Selecione **API Provider:** `OpenAI Compatible`.
3. Preencha os campos:
   - **Base URL:** `http://10.100.10.31:8000/v1`
   - **API Key:** `local` (não é necessária chave proprietária)
   - **Model ID:** `agent-coding`
4. Clique em **Done / Salvar**.

#### 3. Continue (Extensão do VS Code / JetBrains)
No arquivo `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Orquestrador IA - Coding Exec",
      "provider": "openai",
      "model": "agent-coding",
      "apiBase": "http://10.100.10.31:8000/v1"
    },
    {
      "title": "Orquestrador IA - Fast Edits",
      "provider": "openai",
      "model": "agent-coding-fast",
      "apiBase": "http://10.100.10.31:8000/v1"
    },
    {
      "title": "Orquestrador IA - Architectural Plan",
      "provider": "openai",
      "model": "agent-coding-plan",
      "apiBase": "http://10.100.10.31:8000/v1"
    }
  ],
  "tabAutocompleteModel": {
    "title": "Orquestrador Autocomplete",
    "provider": "openai",
    "model": "agent-coding-fast",
    "apiBase": "http://10.100.10.31:8000/v1"
  }
}
```

#### 4. Cursor IDE
Em **Cursor Settings $\rightarrow$ Models**:
1. Ative **OpenAI API Key** (coloque qualquer valor, ex: `custom`).
2. Marque **Override OpenAI Base URL** e insira: `http://10.100.10.31:8000/v1`.
3. Adicione o modelo customizado: `agent-coding`.

---

## 4. Guia Rápido de Comandos e Dicas de Uso no LibreChat

| Objetivo | Como Solicitar no Chat (Exemplo) | O Que Acontece nos Bastidores |
| :--- | :--- | :--- |
| **Exportar para PDF** | *"Gere um PDF desse resumo"* ou *"Faça um resumo de X e anexe em PDF"* | O orquestrador recupera o Markdown anterior, compila via WeasyPrint em layout executivo A4 e devolve o link de download imediato. |
| **Criar Post Social** | *"Crie um post para o Instagram sobre inteligência artificial nos negócios"* | Aciona o Social Media Studio v3: gera copy estratégica, hashtags, carrossel e arte conceitual. |
| **Ajustar Imagem de Post** | *"Troque a frase da imagem anterior para: 'Inovação Contínua'"* | Identifica que é uma revisão, aciona o modelo multimodal de referência (GPT-2.5-sunburst) e altera apenas a frase solicitada. |
| **Gerar Planilha/Gráfico** | *"Gere uma planilha Excel com as vendas do trimestre e monte um gráfico"* | Rota `DATA_EXPORT_CHART`: gera o código analítico, compila o arquivo `.xlsx` e gera o gráfico Seaborn para download. |
| **Consultar Notícias Atuais** | *"Quais as notícias de hoje sobre o setor elétrico?"* | Rota `REALTIME_SEARCH`: faz busca externa na web, filtra fontes e sintetiza as informações mais recentes de 2026. |
| **Dúvidas sobre o Sistema** | *"O que posso fazer nesse sistema?"* ou *"Qual o endpoint do opencode?"* | Rota `DOCUMENT_RAG`: consulta este manual oficial no banco vetorial Qdrant e responde com todos os detalhes e exemplos. |
| **Manipular Arquivos Sandbox** | *"Liste os arquivos disponíveis no workspace"* | Rota `MCP_FILESYSTEM`: inspeciona o diretório seguro `/app/workspace` e retorna a listagem. |
| **Executar Código de Teste** | *"Rode este script python no workspace: print(sum(range(100)))"* | Rota `MCP_CODE_RUNNER`: executa o código na sandbox e retorna a saída exata do interpretador. |

---

## 5. Dicionário de Rotas Semânticas do Orquestrador

O sistema conta com 21 categorias de roteamento autônomo:
1. `CASUAL_CHAT`: Saudações e conversa leve.
2. `FACTUAL_GENERAL`: Conceitos consolidados e história.
3. `REALTIME_SEARCH`: Notícias, eventos recentes e cotações financeiras.
4. `WEATHER_INFO`: Previsão meteorológica global.
5. `DOCUMENT_RAG`: Busca em manuais e documentos da base Qdrant.
6. `CODING`: Programação, infraestrutura, Docker e servidores.
7. `LOCAL_THINKING`: Lógica leve e contas diretas.
8. `HEAVY_REASONING`: Matemática avançada e raciocínio profundo.
9. `VISION`: Análise e interpretação de imagens anexas.
10. `IMAGE_GENERATION`: Desenho e criação artística de novas imagens.
11. `IMAGE_EDIT`: Alteração e edição de fotos enviadas.
12. `SOCIAL_MEDIA_POST`: Criação completa de posts para redes sociais.
13. `DATA_EXPORT_CHART`: Planilhas Excel/CSV e gráficos estatísticos.
14. `SHORT_VIDEO`: Roteiros de vídeo vertical (TikTok/Reels).
15. `MCP_FILESYSTEM`: Operações de arquivos na sandbox.
16. `MCP_FETCH`: Extração de conteúdo de páginas web via link.
17. `MCP_GIT`: Operações e versionamento Git na sandbox.
18. `MCP_BRAVE_SEARCH`: Busca independente via motor Brave.
19. `MCP_DATABASE`: Consultas a bancos de dados SQLite/DuckDB.
20. `MCP_CODE_RUNNER`: Execução de códigos Python e scripts shell.
21. `PDF_EXPORT`: Diagramação e conversão de textos em documentos PDF executivos.
