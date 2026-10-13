"""
Domain Skills Service - Habilidades Especialistas por Domínio
=============================================================
Implementa as regras de negócio, prompts profundos e pipelines de processamento
para 3 verticais de alta especialização:
1. Social Media & Marketing Digital
2. DevOps, Infraestrutura de Redes e Hardware
3. Programação e Design de Sistemas de Alta Performance

Cada skill é projetada para ser invocada como ferramenta pelo LibreChat (via OpenAPI Actions)
ou diretamente pelo orquestrador LangChain.
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI

from config import LITELLM_BASE_URL, LITELLM_API_KEY

# Modelos padrão utilizados para raciocínio em skills
SMART_MODEL = "ComboFree_Smart"
CODE_MODEL = "Combo_Free_Coding"
CHAT_MODEL = "Combo_Free_Chat"


async def invoke_skill_llm(
    prompt: str,
    preferred_model: str = SMART_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 2500
) -> str:
    """Invoca o modelo com cascata transparente de fallback entre modelos rápidos."""
    candidates = [preferred_model, "Combo_Free_Coding", "ComboFree_Smart", "Combo_Free_Chat"]
    seen = set()
    models_to_try = [m for m in candidates if not (m in seen or seen.add(m))]

    for model_name in models_to_try:
        try:
            llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=20.0
            )
            res = await llm.ainvoke([{"role": "user", "content": prompt}])
            text = res.content if hasattr(res, "content") else str(res)
            if text and len(text.strip()) > 10:
                return text.strip()
        except Exception as e:
            print(f"[Skills Service] Modelo {model_name} falhou: {e}. Tentando próximo candidato...")
            continue
    raise RuntimeError("Todos os modelos candidatos falharam ao executar a skill.")


# =========================================================================
# 1. VERTICAL: SOCIAL MEDIA & MARKETING DIGITAL
# =========================================================================

async def execute_social_campaign_planner(
    theme: str,
    target_audience: str,
    platforms: List[str],
    goal: str = "Engajamento e Aquisição de Leads",
    duration_days: int = 7
) -> Dict[str, Any]:
    """Planeja uma campanha completa de conteúdo com calendário diário, ganchos e CTAs."""
    t0 = time.time()
    plat_str = ", ".join(platforms) if platforms else "Instagram e LinkedIn"

    prompt = f"""Você é um Diretor de Criação e Estrategista Chefe de Social Media.
Crie um plano estratégico de campanha detalhado para o seguinte briefing:
- TEMA PRINCIPAL: {theme}
- PÚBLICO-ALVO / PERSONA: {target_audience}
- PLATAFORMAS: {plat_str}
- OBJETIVO DA CAMPANHA: {goal}
- DURAÇÃO: {duration_days} dias

ESTRUTURA OBRIGATÓRIA DA RESPOSTA EM MARKDOWN:
1. 🎯 **Visão Geral & Posicionamento:** Conceito central e mensagem única de valor.
2. 🪝 **Ganchos Psicológicos (Hooks):** 5 variações magnéticas de abertura para reter atenção nos 3 primeiros segundos.
3. 📅 **Calendário Editorial Dia a Dia ({duration_days} dias):**
   Para cada dia detalhe:
   - Formato (Reels, Carrossel, Post Único, Thread)
   - Título do Post
   - Mensagem-chave
   - Chamada para Ação (CTA) específica
4. 🚀 **Métricas de Sucesso (KPIs):** Como medir o ROI e engajamento.

Seja extremamente prático, criativo e orientado a conversão."""

    output_text = await invoke_skill_llm(prompt, preferred_model=SMART_MODEL, temperature=0.3, max_tokens=3000)

    return {
        "status": "success",
        "theme": theme,
        "platforms": platforms,
        "duration_days": duration_days,
        "plan_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_carousel_copywriter(
    topic: str,
    slides_count: int = 7,
    platform: str = "Instagram",
    cta_action: str = "Salve para consultar depois e compartilhe com seu time"
) -> Dict[str, Any]:
    """Cria roteiro e copy completa de carrossel slide a slide com direcionamento visual."""
    t0 = time.time()
    slides_n = min(max(4, slides_count), 10)

    prompt = f"""Você é o redator sênior especializado em Carrosséis Virais para {platform}.
Crie a cópia completa de um carrossel educativo de alto impacto de {slides_n} slides sobre:
TEMA: {topic}
CHAMADA PARA AÇÃO FINAL: {cta_action}

DIRETRIZES TÉCNICAS DE ESTRUTURA:
- Slide 1: A CAPA (Gancho irresistível, subtítulo instigante e elemento visual sugerido).
- Slides 2 até {slides_n - 1}: O CONTEÚDO (1 conceito único por slide, títulos fortes, texto curto em tópicos, analogias fáceis).
- Slide {slides_n}: A CONCLUSÃO & CTA (Resumo em 3 pontos + chamada para ação direta).

FORMATO DE RESPOSTA OBRIGATÓRIO PARA CADA SLIDE:
---
### 🖼️ Slide [X]: [Título do Slide]
* **Texto em Destaque:** "[Frase de impacto curta]"
* **Corpo do Slide:** [1 a 2 parágrafos concisos ou tópicos objetivos]
* **Direção de Arte / Imagem:** [Sugestão visual: ícone, gráfico, foto ou ilustração]
---"""

    output_text = await invoke_skill_llm(prompt, preferred_model=SMART_MODEL, temperature=0.3, max_tokens=2500)

    return {
        "status": "success",
        "topic": topic,
        "platform": platform,
        "slides_count": slides_n,
        "carousel_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_visual_art_director(
    subject: str,
    mood: str = "Cinematográfico e profissional",
    style: str = "Fotografia realista 8k",
    aspect_ratio: str = "4:5",
    lighting: str = "Iluminação dramática de estúdio com rim light sutil"
) -> Dict[str, Any]:
    """Gera prompts de imagem profissionais calibrados para FLUX, Midjourney e Grok."""
    t0 = time.time()

    prompt = f"""Você é um Diretor de Fotografia e Especialista em Engenharia de Prompts para IA Visual (FLUX.1-kontext, FLUX.2-dev e Midjourney v6).
Transforme a seguinte ideia de cena em 3 variações de prompts em inglês de nível ultra-profissional:
- SUJEITO/CENA: {subject}
- CLIMA/MOOD: {mood}
- ESTILO: {style}
- ILUMINAÇÃO: {lighting}
- PROPORÇÃO (ASPECT RATIO): {aspect_ratio}

REQUISITOS TÉCNICOS:
1. Use termos precisos de cinematografia: tipo de lente (ex: 85mm lens, f/1.4 aperture), distância focal, textura da pele, color grading.
2. Forneça:
   - Opção 1: Foco Comercial / Editorial Limpo
   - Opção 2: Foco Narrativo / Emocional Dramático
   - Opção 3: Conceitual / Minimalista de Alto Contraste"""

    output_text = await invoke_skill_llm(prompt, preferred_model=SMART_MODEL, temperature=0.3, max_tokens=2000)

    return {
        "status": "success",
        "subject": subject,
        "aspect_ratio": aspect_ratio,
        "prompts_guide_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_hashtag_seo_optimizer(
    niche: str,
    content_summary: str,
    platform: str = "Instagram"
) -> Dict[str, Any]:
    """Gera grupos de hashtags balanceadas e otimiza o texto para SEO interno das redes."""
    t0 = time.time()

    prompt = f"""Você é um Especialista em SEO e Algoritmos de Distribuição de Redes Sociais ({platform}).
Analise o conteúdo abaixo e produza uma estratégia de ranqueamento e indexação:
- NICHO: {niche}
- RESUMO DO CONTEÚDO: {content_summary}
- PLATAFORMA: {platform}

FORNEÇA:
1. 🏷️ **Conjunto Balanceado de Hashtags:**
   - 3 a 5 de Alto Volume / Nicho Amplo
   - 5 a 8 de Médio Volume / Segmentadas
   - 3 a 5 de Cauda Longa / Ultra-específicas
2. 🔍 **Palavras-chave de Busca (SEO na Legenda):** Termos que devem aparecer naturalmente nas duas primeiras linhas da legenda.
3. 🚫 **Hashtags Proibidas (Banned Check):** Alertas sobre tipos de tags que causam shadowban."""

    output_text = await invoke_skill_llm(prompt, preferred_model=SMART_MODEL, temperature=0.2, max_tokens=1500)

    return {
        "status": "success",
        "niche": niche,
        "platform": platform,
        "seo_guide_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


# =========================================================================
# 2. VERTICAL: DEVOPS, INFRAESTRUTURA DE REDES E HARDWARE
# =========================================================================

async def execute_docker_cicd_generator(
    app_type: str,
    runtime_version: str = "Python 3.11",
    port: int = 8000,
    database: Optional[str] = "PostgreSQL",
    ci_platform: str = "GitHub Actions"
) -> Dict[str, Any]:
    """Gera Dockerfile multi-stage, docker-compose.yml e pipeline de CI/CD para produção."""
    t0 = time.time()

    prompt = f"""Você é um Engenheiro DevOps Sênior / Especialista SRE.
Crie a infraestrutura de conteinerização e automação de deploy profissional para:
- APLICAÇÃO: {app_type}
- RUNTIME / VERSÃO: {runtime_version}
- PORTA INTERNA: {port}
- BANCO DE DADOS: {database or 'Nenhum'}
- PLATAFORMA DE CI/CD: {ci_platform}

FORNEÇA:
1. 🐳 **Dockerfile Multi-Stage Otimizado:**
   - Estágio de build limpo e estágio final mínimo.
   - Usuário não-root (segurança) e healthcheck configurado.
2. 📦 **docker-compose.yml:**
   - Serviços isolados, volumes persistentes, healthchecks e rede customizada.
3. 🚀 **Pipeline de CI/CD ({ci_platform}):**
   - Lint, testes automatizados, build da imagem Docker e push."""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.1, max_tokens=3000)

    return {
        "status": "success",
        "app_type": app_type,
        "runtime_version": runtime_version,
        "ci_platform": ci_platform,
        "devops_config_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_network_infra_troubleshooter(
    issue_description: str,
    service_type: str = "Nginx Proxy / DNS",
    os_environment: str = "Linux Ubuntu 22.04 LTS",
    error_log: Optional[str] = None
) -> Dict[str, Any]:
    """Diagnostica problemas de rede, firewalls, DNS, roteamento e proxies reversos."""
    t0 = time.time()
    log_section = f"\nLOGS / MENSAGENS DE ERRO FORNECIDAS:\n```{error_log}\n```" if error_log else ""

    prompt = f"""Você é um Administrador de Redes e Engenheiro de Infraestrutura Sênior (CCIE/Linux Sênior).
Diagnostique e resolva o seguinte incidente de rede e conectividade:
- DESCRIÇÃO DO PROBLEMA: {issue_description}
- SERVIÇO / PROTOCOLO AFETADO: {service_type}
- SISTEMA OPERACIONAL: {os_environment}
{log_section}

FORNEÇA UMA SOLUÇÃO ESTRUTURADA:
1. 🩺 **Diagnóstico da Causa Raiz:** Hipóteses mais prováveis.
2. 🛠️ **Comandos de Diagnóstico Imediato:** Comandos exatos no terminal (ss, dig, curl, iptables, traceroute).
3. 🔧 **Procedimento Passo a Passo de Correção:** Configurações exatas a editar.
4. 🛡️ **Ação Preventiva:** Como monitorar para impedir recorrência."""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.1, max_tokens=2500)

    return {
        "status": "success",
        "service_type": service_type,
        "troubleshooting_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_hardware_capacity_planner(
    workload_type: str,
    expected_users_or_concurrency: str = "100 usuários simultâneos",
    ai_model_specs: Optional[str] = None,
    storage_profile: str = "Alta taxa de I/O em banco de dados"
) -> Dict[str, Any]:
    """Dimensiona recursos de hardware (CPU, VRAM, RAM, IOPS e Storage) para servidores."""
    t0 = time.time()
    ai_section = f"- MODELOS DE IA / LLM CONSIDERADOS: {ai_model_specs}\n" if ai_model_specs else ""

    prompt = f"""Você é um Arquiteto de Sistemas de Infraestrutura e Hardware Corporativo.
Efetue o dimensionamento e capacidade de hardware (*Capacity Planning*) para:
- TIPO DE CARGA / APLICAÇÃO: {workload_type}
- CONCORRÊNCIA ESTIMADA: {expected_users_or_concurrency}
{ai_section}- PERFIL DE STORAGE / BANCO: {storage_profile}

FORNEÇA UMA ESPECIFICAÇÃO TÉCNICA FORMAL:
1. 💻 **CPU / Processamento:** Número de Cores/Threads e arquitetura recomendada.
2. 🧠 **Memória RAM & VRAM (se aplicável):** Dimensionamento e quantizações.
3. 💾 **Subsistema de Armazenamento:** Capacidade bruta, tipo de storage (NVMe PCIe), RAID e IOPS.
4. 🌐 **Largura de Banda e Interface de Rede:** Requisitos de portas (1 Gbps vs 10 Gbps)."""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.2, max_tokens=2500)

    return {
        "status": "success",
        "workload_type": workload_type,
        "capacity_plan_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


# =========================================================================
# 3. VERTICAL: PROGRAMAÇÃO E DESIGN DE SISTEMAS
# =========================================================================

async def execute_system_architecture_designer(
    system_goal: str,
    scale_requirements: str = "10.000 requisições por minuto com latência p99 < 150ms",
    tech_stack: Optional[str] = "Python, FastAPI, PostgreSQL, Redis, RabbitMQ",
    diagram_type: str = "sequence"
) -> Dict[str, Any]:
    """Desenha arquiteturas escaláveis e gera diagramas Mermaid prontos para renderização."""
    t0 = time.time()

    prompt = f"""Você é um Arquiteto Chefe de Software (Staff/Principal Architect).
Desenvolva a especificação técnica e o diagrama arquitetural para:
- OBJETIVO DO SISTEMA: {system_goal}
- REQUISITOS DE ESCALA & LATÊNCIA: {scale_requirements}
- STACK TECNOLÓGICA SUGERIDA: {tech_stack}
- TIPO PRINCIPAL DE DIAGRAMA: {diagram_type}

ESTRUTURA DA RESPOSTA:
1. 🏛️ **Visão Arquitetural & Padrões:** Escolha de padrão e justificativa.
2. 📊 **Diagrama Arquitetural em Mermaid (OBRIGATÓRIO):**
   Gere um bloco de código Mermaid completo:
   ```mermaid
   [diagrama aqui]
   ```
3. 💾 **Estratégia de Persistência e Cache:** Particionamento e cache.
4. 🛡️ **Padrões de Resiliência:** Circuit Breaker, Retries com Jitter, Rate Limiting."""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.1, max_tokens=3000)

    return {
        "status": "success",
        "system_goal": system_goal,
        "diagram_type": diagram_type,
        "architecture_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_code_security_audit(
    code_snippet: str,
    language: str = "Python",
    context: Optional[str] = "Endpoint público de API"
) -> Dict[str, Any]:
    """Audita códigos contra OWASP Top 10, injeções, vazamentos e falhas de segurança."""
    t0 = time.time()

    prompt = f"""Você é um Especialista em Segurança Ofensiva e Auditor de Código (AppSec / DevSecOps).
Realize uma auditoria estrita de segurança no seguinte trecho de código em {language}:
CONTEXTO DE EXECUÇÃO: {context}

CÓDIGO A SER AUDITADO:
```{language.lower()}
{code_snippet}
```

FORNEÇA:
1. 🚨 **Tabela de Vulnerabilidades Encontradas:** (Severidade, Trecho, Tipo CWE/OWASP, Impacto).
2. 🔍 **Explicação Detalhada do Vetor de Ataque.**
3. 🛡️ **Código Refatorado e Seguro.**"""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.0, max_tokens=3000)

    return {
        "status": "success",
        "language": language,
        "security_audit_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_test_suite_generator(
    code_snippet: str,
    test_framework: str = "pytest",
    language: str = "Python",
    mock_dependencies: bool = True
) -> Dict[str, Any]:
    """Gera suítes de testes unitários com fixtures, mocks e cobertura de edge cases."""
    t0 = time.time()

    prompt = f"""Você é um Engenheiro de Qualidade de Software (QA Automation & TDD Specialist).
Crie uma suíte de testes automatizados profissional para o código abaixo:
- LINGUAGEM: {language}
- FRAMEWORK DE TESTES: {test_framework}
- MOCKAR DEPENDÊNCIAS EXTERNAS: {'Sim' if mock_dependencies else 'Não'}

CÓDIGO ORIGINAL:
```{language.lower()}
{code_snippet}
```

DIRETRIZES:
1. Testar o "Caminho Feliz" (Happy Path).
2. Testar todos os cenários de erro e exceções (Edge Cases).
3. Usar Fixtures e mocks adequados. O código deve ser 100% executável."""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.1, max_tokens=3000)

    return {
        "status": "success",
        "language": language,
        "test_framework": test_framework,
        "test_suite_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


async def execute_code_refactor_clean_code(
    code_snippet: str,
    language: str = "Python",
    focus_areas: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Refatora código para Clean Code, SOLID, type hints e redução de complexidade."""
    t0 = time.time()
    focus_str = ", ".join(focus_areas) if focus_areas else "Legibilidade, SOLID, Type Hints e Performance"

    prompt = f"""Você é um Engenheiro de Software Sênior especialista em Clean Code e Arquitetura Limpa.
Refatore o seguinte trecho de código em {language}:
ÁREAS DE FOCO: {focus_str}

CÓDIGO ORIGINAL:
```{language.lower()}
{code_snippet}
```

FORNEÇA:
1. 💡 **Diagnóstico dos "Code Smells".**
2. ✨ **Código Totalmente Refatorado com Type Hints e SRP.**
3. 📈 **Quadro Comparativo Antes vs Depois.**"""

    output_text = await invoke_skill_llm(prompt, preferred_model=CODE_MODEL, temperature=0.1, max_tokens=3000)

    return {
        "status": "success",
        "language": language,
        "refactored_code_markdown": output_text,
        "execution_time_seconds": round(time.time() - t0, 3)
    }


# =========================================================================
# 4. MAPEAMENTO DE PERSONAS / PROMPTS DE SISTEMA PARA MODELOS DO LIBRECHAT
# =========================================================================

DOMAIN_MODEL_SYSTEM_PROMPTS = {
    "skill-social-media": """Você é um Diretor de Criação e Especialista Sênior em Social Media & Marketing Digital.
Suas respostas devem ser práticas, persuasivas e com alta retenção.
Ao criar conteúdo para redes sociais (Instagram, LinkedIn, X, TikTok), inclua:
1. Ganchos psicológicos (hooks) fortes para reter a atenção nos primeiros segundos.
2. Estrutura clara (carrossel slide a slide ou post direto).
3. Chamada para Ação (CTA) clara orientada a conversão.
4. Direção visual ou sugestão de imagem/prompt e hashtags estratégicas.""",

    "skill-devops": """Você é um Engenheiro DevOps Sênior e Especialista em Redes e Infraestrutura.
Suas respostas devem focar em automação, resiliência, boas práticas e segurança:
1. Em Docker e CI/CD: Forneça Dockerfiles multi-stage, usuários não-root e pipelines seguros.
2. Em Redes: Forneça diagnóstico cirúrgico (DNS, latência, MTU, roteamento) com comandos práticos.
3. Em Hardware: Avalie dimensionamento de CPU, memória, IOPS de storage e throughput de rede.""",

    "skill-coding": """Você é um Engenheiro de Software Principal especialista em Clean Code e Princípios SOLID.
Suas respostas devem focar em excelência técnica:
1. Código limpo, desacoplado, modular e com tipagem estrita.
2. Tratamento defensivo de erros e edge cases.
3. Sugestão ou implementação de testes unitários determinísticos.""",

    "skill-system-designer": """Você é um Arquiteto de Sistemas Distribuídos e Engenheiro de Software Principal.
Suas respostas devem analisar trade-offs arquiteturais profundos:
1. Padrões arquiteturais escaláveis (Microsserviços, EDA, CQRS, Saga).
2. Escolha de tecnologias de persistência e mensageria.
3. Sempre que relevante, inclua diagramas arquiteturais claros em sintaxe Mermaid.""",

    "skill-code-audit": """Você é um Engenheiro de Segurança de Aplicações (AppSec) e Auditor de Código Sênior.
Suas respostas devem analisar minuciosamente o código em busca de:
1. Vulnerabilidades OWASP Top 10 (Injection, Broken Auth, SSRF, IDOR, XSS, etc.).
2. Sanitização de inputs e vazamento de credenciais/segredos.
3. Código corrigido com explicações objetivas do vetor de ataque neutralizado."""
}

def get_skill_system_prompt(model_id: str) -> Optional[str]:
    """Retorna o prompt especializado caso o modelo selecionado seja uma skill de domínio."""
    if not model_id:
        return None
    m = model_id.lower()
    for key, prompt in DOMAIN_MODEL_SYSTEM_PROMPTS.items():
        if key in m:
            return prompt
    return None
