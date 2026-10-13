"""
Skills API Router - Endpoints Dedicados de Skills para o LibreChat
===================================================================
Expõe as Skills Especialistas (Social Media, DevOps, Engenharia de Software)
e as Ferramentas MCP como endpoints REST OpenAPI limpos e padronizados.

Gera também a rota dedicada:
    GET /skills/openapi.json
Permitindo que o usuário importe todas as skills no menu de "Actions / Skills"
do LibreChat com um único clique ou URL.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from domain_skills_service import (
    execute_social_campaign_planner,
    execute_carousel_copywriter,
    execute_visual_art_director,
    execute_hashtag_seo_optimizer,
    execute_docker_cicd_generator,
    execute_network_infra_troubleshooter,
    execute_hardware_capacity_planner,
    execute_system_architecture_designer,
    execute_code_security_audit,
    execute_test_suite_generator,
    execute_code_refactor_clean_code
)

from mcp_git_service import git_status_impl, git_diff_impl, git_log_impl, git_commit_impl
from mcp_database_service import sqlite_query_impl, sqlite_list_tables_impl, sqlite_describe_table_impl
from mcp_code_runner_service import run_python_code_impl, run_bash_command_impl
from mcp_brave_search_service import async_brave_search

router = APIRouter(prefix="/v1/skills", tags=["Skills Especialistas"])


# =========================================================================
# SCHEMAS PYDANTIC - SOCIAL MEDIA
# =========================================================================

class CampaignPlannerRequest(BaseModel):
    theme: str = Field(..., description="Tema ou produto foco da campanha de social media", example="Lançamento de curso online de Python com IA")
    target_audience: str = Field(..., description="Perfil do público-alvo ou persona", example="Desenvolvedores juniores e transição de carreira")
    platforms: List[str] = Field(default=["Instagram", "LinkedIn"], description="Redes sociais alvo", example=["Instagram", "LinkedIn", "YouTube"])
    goal: str = Field(default="Engajamento e Aquisição de Leads", description="Objetivo principal da campanha")
    duration_days: int = Field(default=7, description="Duração do cronograma em dias (1 a 30)", ge=1, le=30)


class CarouselCopywriterRequest(BaseModel):
    topic: str = Field(..., description="Tema principal do carrossel", example="5 erros clássicos em Docker que destroem sua performance")
    slides_count: int = Field(default=7, description="Quantidade de slides desejada (4 a 10)", ge=4, le=10)
    platform: str = Field(default="Instagram", description="Plataforma alvo (Instagram ou LinkedIn)")
    cta_action: str = Field(default="Salve este post e compartilhe com seu time", description="Chamada para ação no último slide")


class VisualArtDirectorRequest(BaseModel):
    subject: str = Field(..., description="Ideia ou cena a ser fotografada/gerada", example="Programador sênior trabalhando à noite em escritório minimalista com vista para cidade chuvosa")
    mood: str = Field(default="Cinematográfico e profissional", description="Clima ou atmosfera da cena")
    style: str = Field(default="Fotografia realista 8k", description="Estilo artístico ou formato fotográfico")
    aspect_ratio: str = Field(default="4:5", description="Proporção da imagem (1:1, 4:5, 16:9, 9:16)")
    lighting: str = Field(default="Iluminação suave de neon com rim light dramático", description="Tipo de iluminação")


class HashtagSeoRequest(BaseModel):
    niche: str = Field(..., description="Nicho do perfil ou mercado", example="Inteligência Artificial e Desenvolvimento de Software")
    content_summary: str = Field(..., description="Resumo do conteúdo publicado na postagem", example="Post explicando como conectar MCP no LibreChat")
    platform: str = Field(default="Instagram", description="Rede social foco")


# =========================================================================
# SCHEMAS PYDANTIC - DEVOPS & INFRAESTRUTURA
# =========================================================================

class DockerCicdRequest(BaseModel):
    app_type: str = Field(..., description="Tipo de aplicação ou framework", example="FastAPI com LangChain e Uvicorn")
    runtime_version: str = Field(default="Python 3.11", description="Linguagem e versão do runtime")
    port: int = Field(default=8000, description="Porta interna exposta pelo container")
    database: Optional[str] = Field(default="PostgreSQL", description="Banco de dados associado")
    ci_platform: str = Field(default="GitHub Actions", description="Plataforma de CI/CD (GitHub Actions ou GitLab CI)")


class NetworkTroubleshooterRequest(BaseModel):
    issue_description: str = Field(..., description="Descrição detalhada do sintoma ou falha", example="Container não consegue alcançar serviço no host em 10.100.10.30 porta 3080")
    service_type: str = Field(default="Nginx Proxy / Docker Network", description="Serviço ou protocolo afetado")
    os_environment: str = Field(default="Linux Ubuntu 22.04 LTS", description="Sistema operacional e versão")
    error_log: Optional[str] = Field(default=None, description="Logs de erro, saída de curl ou journalctl")


class HardwarePlannerRequest(BaseModel):
    workload_type: str = Field(..., description="Tipo de carga do servidor", example="Servidor de inferência local Ollama com modelo Qwen 32B e banco vetorial Qdrant")
    expected_users_or_concurrency: str = Field(default="50 usuários simultâneos", description="Concorrência ou volume de tráfego esperado")
    ai_model_specs: Optional[str] = Field(default="Modelos 7B a 32B quantizados Q4_K_M", description="Especificações de modelos de IA se houver")
    storage_profile: str = Field(default="Alta taxa de I/O aleatório com leituras vetoriais", description="Perfil de acesso ao disco")


# =========================================================================
# SCHEMAS PYDANTIC - PROGRAMAÇÃO & ARQUITETURA
# =========================================================================

class ArchitectureDesignerRequest(BaseModel):
    system_goal: str = Field(..., description="Objetivo e escopo do sistema a ser projetado", example="Plataforma de orquestração de múltiplos agentes de IA com streaming e cache vetorial")
    scale_requirements: str = Field(default="10.000 requisições/min com latência p99 < 150ms", description="Métricas de escalabilidade e SLA")
    tech_stack: Optional[str] = Field(default="Python, FastAPI, Redis, Qdrant, PostgreSQL", description="Tecnologias preferidas")
    diagram_type: str = Field(default="sequence", description="Tipo de diagrama Mermaid (sequence, class, flow, erDiagram)")


class CodeSecurityAuditRequest(BaseModel):
    code_snippet: str = Field(..., description="Trecho de código a ser auditado contra vulnerabilidades")
    language: str = Field(default="Python", description="Linguagem do código")
    context: Optional[str] = Field(default="Endpoint de API pública com entrada de usuário", description="Onde o código é executado")


class TestSuiteRequest(BaseModel):
    code_snippet: str = Field(..., description="Código-fonte para o qual os testes serão gerados")
    test_framework: str = Field(default="pytest", description="Framework de testes (pytest, unittest, jest)")
    language: str = Field(default="Python", description="Linguagem do código")
    mock_dependencies: bool = Field(default=True, description="Se deve mockar bancos de dados e APIs externas")


class CleanCodeRefactorRequest(BaseModel):
    code_snippet: str = Field(..., description="Código que necessita de refatoração e limpeza")
    language: str = Field(default="Python", description="Linguagem do código")
    focus_areas: Optional[List[str]] = Field(default=["SOLID", "Type Hints", "Legibilidade"], description="Áreas principais de foco da refatoração")


# =========================================================================
# ENDPOINTS REST DE SOCIAL MEDIA
# =========================================================================

@router.post("/social/campaign-planner", summary="Social Media: Planejador de Campanhas", description="Gera estratégia completa de campanha com ganchos virais e calendário diário.")
async def api_social_campaign_planner(req: CampaignPlannerRequest):
    return await execute_social_campaign_planner(
        theme=req.theme,
        target_audience=req.target_audience,
        platforms=req.platforms,
        goal=req.goal,
        duration_days=req.duration_days
    )


@router.post("/social/carousel-copywriter", summary="Social Media: Redator de Carrosséis", description="Cria roteiro e copy de carrossel slide a slide com direcionamento de arte.")
async def api_carousel_copywriter(req: CarouselCopywriterRequest):
    return await execute_carousel_copywriter(
        topic=req.topic,
        slides_count=req.slides_count,
        platform=req.platform,
        cta_action=req.cta_action
    )


@router.post("/social/visual-art-director", summary="Social Media: Diretor de Arte Visual (Prompts IA)", description="Gera prompts de imagem profissionais para FLUX, Midjourney e Grok.")
async def api_visual_art_director(req: VisualArtDirectorRequest):
    return await execute_visual_art_director(
        subject=req.subject,
        mood=req.mood,
        style=req.style,
        aspect_ratio=req.aspect_ratio,
        lighting=req.lighting
    )


@router.post("/social/hashtag-seo", summary="Social Media: Otimizador de SEO e Hashtags", description="Gera hashtags balanceadas e otimiza legendas para ranqueamento interno.")
async def api_hashtag_seo(req: HashtagSeoRequest):
    return await execute_hashtag_seo_optimizer(
        niche=req.niche,
        content_summary=req.content_summary,
        platform=req.platform
    )


# =========================================================================
# ENDPOINTS REST DE DEVOPS & INFRAESTRUTURA
# =========================================================================

@router.post("/devops/docker-cicd", summary="DevOps: Gerador de Dockerfile e CI/CD", description="Gera Dockerfile multi-stage otimizado, docker-compose e pipeline de CI/CD.")
async def api_docker_cicd(req: DockerCicdRequest):
    return await execute_docker_cicd_generator(
        app_type=req.app_type,
        runtime_version=req.runtime_version,
        port=req.port,
        database=req.database,
        ci_platform=req.ci_platform
    )


@router.post("/devops/network-troubleshooter", summary="DevOps: Diagnóstico de Redes e Infra", description="Diagnostica problemas de conectividade, Nginx, DNS e firewalls com comandos de correção.")
async def api_network_troubleshooter(req: NetworkTroubleshooterRequest):
    return await execute_network_infra_troubleshooter(
        issue_description=req.issue_description,
        service_type=req.service_type,
        os_environment=req.os_environment,
        error_log=req.error_log
    )


@router.post("/devops/hardware-planner", summary="DevOps: Planejador de Capacidade de Hardware", description="Dimensiona CPU, RAM, VRAM, Storage NVMe e IOPS para servidores e IA local.")
async def api_hardware_planner(req: HardwarePlannerRequest):
    return await execute_hardware_capacity_planner(
        workload_type=req.workload_type,
        expected_users_or_concurrency=req.expected_users_or_concurrency,
        ai_model_specs=req.ai_model_specs,
        storage_profile=req.storage_profile
    )


# =========================================================================
# ENDPOINTS REST DE PROGRAMAÇÃO & ARQUITETURA
# =========================================================================

@router.post("/coding/architecture-designer", summary="Programação: Designer de Arquitetura & Mermaid", description="Gera especificação de arquitetura de software e diagrama Mermaid renderizável.")
async def api_architecture_designer(req: ArchitectureDesignerRequest):
    return await execute_system_architecture_designer(
        system_goal=req.system_goal,
        scale_requirements=req.scale_requirements,
        tech_stack=req.tech_stack,
        diagram_type=req.diagram_type
    )


@router.post("/coding/security-audit", summary="Programação: Auditor de Segurança de Código (AppSec)", description="Audita código contra OWASP Top 10, injeções de SQL, XSS e vazamento de chaves.")
async def api_code_security_audit(req: CodeSecurityAuditRequest):
    return await execute_code_security_audit(
        code_snippet=req.code_snippet,
        language=req.language,
        context=req.context
    )


@router.post("/coding/test-suite", summary="Programação: Gerador de Testes Automatizados", description="Gera suíte completa de testes unitários com fixtures, mocks e cenários de borda.")
async def api_test_suite(req: TestSuiteRequest):
    return await execute_test_suite_generator(
        code_snippet=req.code_snippet,
        test_framework=req.test_framework,
        language=req.language,
        mock_dependencies=req.mock_dependencies
    )


@router.post("/coding/refactor-clean-code", summary="Programação: Refatorador Clean Code & SOLID", description="Refatora código para eliminar code smells, adicionando tipagem estrita e padrões SOLID.")
async def api_clean_code_refactor(req: CleanCodeRefactorRequest):
    return await execute_code_refactor_clean_code(
        code_snippet=req.code_snippet,
        language=req.language,
        focus_areas=req.focus_areas
    )
