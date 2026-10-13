"""
MCP Git Service - Operações Seguras de Controle de Versão de Código
====================================================================
Implementação da especificação Model Context Protocol (MCP) Git:
Permite aos agentes (CODING, agent-orchestrator e usuário) inspecionar,
controlar versões, visualizar diffs, histórico de commits e versionar
projetos de forma 100% segura e estritamente confinada dentro do
diretório seguro WORKSPACE_PATH (/app/workspace).

Mecanismos de Segurança (Sandbox Zero-Trust):
1. Resolução canônica contra WORKSPACE_PATH: impossível operar repositórios fora da sandbox.
2. Execução sem shell=True (argument list pura no subprocess) contra injeções de comando.
3. Limite seguro de tamanho de diff e log (max 30.000 chars) para evitar sobrecarga de tokens.
4. Exposição dupla: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
5. Suporte a git status, diff, log, commit, branches, checkout, init, show e add.
"""

import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.tools import tool

# Diretório raiz seguro do Sandbox
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/app/workspace")
os.makedirs(WORKSPACE_PATH, exist_ok=True)

MAX_OUTPUT_CHARS = 30000


# =========================================================================
# 1. FUNÇÕES DE SEGURANÇA E RESOLUÇÃO CANÔNICA
# =========================================================================

def get_workspace_root() -> Path:
    """Retorna o caminho canônico absoluto do diretório raiz do workspace."""
    return Path(WORKSPACE_PATH).resolve()


def resolve_safe_repo_path(rel_or_abs_path: str = "") -> Path:
    """
    Resolve e valida o caminho de um repositório git contra o diretório seguro do workspace.
    Garante que qualquer tentativa de Path Traversal fora da sandbox seja interceptada.
    """
    base_dir = get_workspace_root()
    clean_str = str(rel_or_abs_path or "").strip()

    if not clean_str or clean_str in [".", "./", "/"]:
        target = base_dir
    elif clean_str.startswith("/"):
        target = Path(clean_str).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            raise PermissionError(f"Acesso negado: O caminho absoluto '{clean_str}' está fora do workspace seguro ({base_dir}).")
    else:
        target = (base_dir / clean_str).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            raise PermissionError(f"Acesso negado: O caminho relativo '{clean_str}' tenta escapar da sandbox.")

    return target


def _run_git_command(args: List[str], cwd: Path, timeout: float = 15.0) -> Tuple[int, str, str]:
    """
    Executa comando git de forma segura via subprocess sem shell=True.
    """
    if not cwd.exists():
        return 1, "", f"Diretório não encontrado: {cwd}"

    # Garante configuração básica de identidade caso o repositório não tenha
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = env.get("GIT_AUTHOR_NAME", "Orchestrator Agent")
    env["GIT_AUTHOR_EMAIL"] = env.get("GIT_AUTHOR_EMAIL", "agent@orchestrator.local")
    env["GIT_COMMITTER_NAME"] = env.get("GIT_COMMITTER_NAME", "Orchestrator Agent")
    env["GIT_COMMITTER_EMAIL"] = env.get("GIT_COMMITTER_EMAIL", "agent@orchestrator.local")

    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Comando 'git {' '.join(args)}' excedeu o tempo limite de {timeout}s."
    except Exception as e:
        return -1, "", f"Erro ao executar git: {e}"


# =========================================================================
# 2. FERRAMENTAS DO MCP GIT (CORE IMPLEMENTATION)
# =========================================================================

def git_status_impl(repo_path: str = "") -> str:
    """Implementação direta de git status."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        code, stdout, stderr = _run_git_command(["status", "--porcelain", "-b"], cwd=safe_dir)
        if code != 0:
            if "not a git repository" in stderr.lower():
                return f"⚠️ O diretório '{safe_dir.name or 'workspace'}' não é um repositório Git inicializado. Utilize 'mcp_git_init' para inicializá-lo."
            return f"Erro ao consultar git status: {stderr}"

        # Também obtém status descritivo limpo
        _, desc_out, _ = _run_git_command(["status"], cwd=safe_dir)
        res = desc_out.strip() or stdout.strip() or "Nenhuma modificação detectada (working tree clean)."
        return res[:MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Erro ao executar git status: {e}"


def git_diff_impl(repo_path: str = "", target: str = "", staged: bool = False) -> str:
    """Implementação direta de git diff."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        args = ["diff"]
        if staged:
            args.append("--staged")
        if target and target.strip():
            args.append(target.strip())

        code, stdout, stderr = _run_git_command(args, cwd=safe_dir)
        if code != 0:
            return f"Erro ao consultar git diff: {stderr}"

        if not stdout.strip():
            return "Nenhuma alteração encontrada no diff."
        return stdout[:MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Erro ao executar git diff: {e}"


def git_log_impl(repo_path: str = "", max_count: int = 10) -> str:
    """Implementação direta de git log."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        count = min(max(1, int(max_count)), 50)
        code, stdout, stderr = _run_git_command(
            ["log", f"-n{count}", "--pretty=format:%h - %an (%cr): %s"],
            cwd=safe_dir
        )
        if code != 0:
            if "does not have any commits yet" in stderr.lower():
                return "O repositório ainda não possui nenhum commit."
            if "not a git repository" in stderr.lower():
                return f"⚠️ O diretório '{safe_dir.name or 'workspace'}' não é um repositório Git."
            return f"Erro ao consultar git log: {stderr}"

        if not stdout.strip():
            return "Nenhum commit encontrado no histórico."
        return stdout[:MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Erro ao executar git log: {e}"


def git_commit_impl(message: str, repo_path: str = "", files: Optional[List[str]] = None, add_all: bool = True) -> str:
    """Implementação direta de git add + git commit."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        clean_msg = (message or "").strip()
        if not clean_msg:
            return "Erro: Uma mensagem de commit não-vazia deve ser fornecida."

        # 1. Adicionar arquivos (git add)
        if files and len(files) > 0:
            for f in files:
                # Valida cada arquivo contra sandbox
                resolve_safe_repo_path(f)
            code_add, _, err_add = _run_git_command(["add"] + files, cwd=safe_dir)
        elif add_all:
            code_add, _, err_add = _run_git_command(["add", "-A"], cwd=safe_dir)
        else:
            code_add = 0
            err_add = ""

        if code_add != 0:
            return f"Erro no git add: {err_add}"

        # 2. Executar commit
        code_c, out_c, err_c = _run_git_command(["commit", "-m", clean_msg], cwd=safe_dir)
        if code_c != 0:
            if "nothing to commit" in (out_c + err_c).lower():
                return "Nada para commitar (árvore de trabalho limpa)."
            return f"Erro ao commitar: {err_c or out_c}"

        return f"Commit criado com sucesso!\n{out_c.strip()}"
    except Exception as e:
        return f"Erro ao criar commit: {e}"


def git_branches_impl(repo_path: str = "") -> str:
    """Implementação direta de listagem de branches."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        code, stdout, stderr = _run_git_command(["branch", "-v"], cwd=safe_dir)
        if code != 0:
            return f"Erro ao listar branches: {stderr}"
        return stdout.strip() or "Nenhuma branch listada."
    except Exception as e:
        return f"Erro ao listar branches: {e}"


def git_init_impl(repo_path: str = "") -> str:
    """Implementação direta de git init."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        safe_dir.mkdir(parents=True, exist_ok=True)
        code, stdout, stderr = _run_git_command(["init"], cwd=safe_dir)
        if code != 0:
            return f"Erro ao inicializar repositório: {stderr}"
        return f"Repositório Git inicializado com sucesso em '{safe_dir}'."
    except Exception as e:
        return f"Erro ao inicializar repositório: {e}"


def git_checkout_impl(target: str, repo_path: str = "", create_branch: bool = False) -> str:
    """Implementação direta de git checkout."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        clean_target = (target or "").strip()
        if not clean_target:
            return "Erro: Informe uma branch, commit ou arquivo de destino para checkout."

        args = ["checkout"]
        if create_branch:
            args.append("-b")
        args.append(clean_target)

        code, stdout, stderr = _run_git_command(args, cwd=safe_dir)
        if code != 0:
            return f"Erro no git checkout: {stderr or stdout}"
        return stdout.strip() or stderr.strip() or f"Checkout para '{clean_target}' concluído com sucesso."
    except Exception as e:
        return f"Erro no git checkout: {e}"


def git_show_impl(commit_hash: str, repo_path: str = "") -> str:
    """Implementação direta de git show."""
    try:
        safe_dir = resolve_safe_repo_path(repo_path)
        clean_hash = (commit_hash or "").strip()
        if not clean_hash:
            return "Erro: Informe o hash do commit para consulta."

        code, stdout, stderr = _run_git_command(["show", "--stat", clean_hash], cwd=safe_dir)
        if code != 0:
            return f"Erro no git show: {stderr}"
        return stdout[:MAX_OUTPUT_CHARS]
    except Exception as e:
        return f"Erro no git show: {e}"


# =========================================================================
# 3. EXPOSIÇÃO COMO LANGCHAIN TOOLS (@tool)
# =========================================================================

@tool
def mcp_git_status(repo_path: str = "") -> str:
    """
    Exibe o status atual do repositório Git no workspace seguro (arquivos modificados, novos ou deletados).
    Args:
        repo_path: Caminho relativo da pasta do repositório dentro do workspace (ou vazio para a raiz).
    """
    return git_status_impl(repo_path=repo_path)


@tool
def mcp_git_diff(repo_path: str = "", target: str = "", staged: bool = False) -> str:
    """
    Exibe as diferenças de código (diff) no repositório Git do workspace seguro.
    Args:
        repo_path: Caminho relativo do repositório dentro do workspace.
        target: Arquivo específico ou commit de comparação.
        staged: Se True, mostra alterações preparadas para commit (--staged).
    """
    return git_diff_impl(repo_path=repo_path, target=target, staged=staged)


@tool
def mcp_git_log(repo_path: str = "", max_count: int = 10) -> str:
    """
    Consulta o histórico recente de commits do repositório Git no workspace seguro.
    Args:
        repo_path: Caminho relativo do repositório dentro do workspace.
        max_count: Quantidade máxima de commits para exibir (padrão: 10).
    """
    return git_log_impl(repo_path=repo_path, max_count=max_count)


@tool
def mcp_git_commit(message: str, repo_path: str = "", files: Optional[List[str]] = None, add_all: bool = True) -> str:
    """
    Cria um novo commit no repositório Git seguro no workspace com mensagem estruturada.
    Args:
        message: Mensagem descritiva do commit.
        repo_path: Caminho relativo do repositório dentro do workspace.
        files: Lista opcional de arquivos específicos para commitar.
        add_all: Se True, adiciona todas as alterações não preparadas antes de commitar.
    """
    return git_commit_impl(message=message, repo_path=repo_path, files=files, add_all=add_all)


@tool
def mcp_git_branches(repo_path: str = "") -> str:
    """
    Lista as branches locais do repositório Git no workspace seguro.
    Args:
        repo_path: Caminho relativo do repositório dentro do workspace.
    """
    return git_branches_impl(repo_path=repo_path)


@tool
def mcp_git_init(repo_path: str = "") -> str:
    """
    Inicializa um novo repositório Git dentro da pasta indicada no workspace seguro.
    Args:
        repo_path: Caminho relativo onde inicializar o repositório.
    """
    return git_init_impl(repo_path=repo_path)


@tool
def mcp_git_checkout(target: str, repo_path: str = "", create_branch: bool = False) -> str:
    """
    Alterna entre branches ou restaura arquivos no repositório Git do workspace seguro.
    Args:
        target: Nome da branch ou arquivo para checkout.
        repo_path: Caminho relativo do repositório dentro do workspace.
        create_branch: Se True, cria a branch com a flag -b.
    """
    return git_checkout_impl(target=target, repo_path=repo_path, create_branch=create_branch)


@tool
def mcp_git_show(commit_hash: str, repo_path: str = "") -> str:
    """
    Exibe os detalhes e alterações de um commit específico pelo seu hash.
    Args:
        commit_hash: Hash identificador do commit.
        repo_path: Caminho relativo do repositório dentro do workspace.
    """
    return git_show_impl(commit_hash=commit_hash, repo_path=repo_path)


def get_mcp_git_langchain_tools() -> List[Any]:
    """Retorna todas as ferramentas MCP Git prontas para LangChain."""
    return [
        mcp_git_status,
        mcp_git_diff,
        mcp_git_log,
        mcp_git_commit,
        mcp_git_branches,
        mcp_git_init,
        mcp_git_checkout,
        mcp_git_show,
    ]


# =========================================================================
# 4. EXPOSIÇÃO COMO SCHEMAS OPENAI TOOL CALLING
# =========================================================================

def get_mcp_git_openai_tools() -> List[Dict[str, Any]]:
    """Retorna os schemas de chamada de função compatíveis com o padrão OpenAI Tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp_git_status",
                "description": "Exibe o status do repositório Git (arquivos modificados, untracked ou prontos para commit) dentro do workspace seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório dentro do workspace (deixe vazio para a raiz)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_diff",
                "description": "Mostra as alterações de código (diff) no repositório seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório dentro do workspace."
                        },
                        "target": {
                            "type": "string",
                            "description": "Arquivo específico ou commit para visualização do diff."
                        },
                        "staged": {
                            "type": "boolean",
                            "description": "Se verdadeiro, mostra as alterações prontas para commit (--staged)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_log",
                "description": "Exibe o histórico recente de commits de um repositório no workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório dentro do workspace."
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "Número máximo de commits para retornar (padrão: 10)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_commit",
                "description": "Prepara alterações e cria um commit Git com uma mensagem descritiva dentro do workspace seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Mensagem descritiva do commit."
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório dentro do workspace."
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Lista de arquivos específicos para commitar."
                        },
                        "add_all": {
                            "type": "boolean",
                            "description": "Se verdadeiro, adiciona todas as alterações antes de commitar (padrão: true)."
                        }
                    },
                    "required": ["message"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_branches",
                "description": "Lista as branches locais do repositório Git.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório dentro do workspace."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_init",
                "description": "Inicializa um novo repositório Git em uma pasta do workspace seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo onde inicializar o repositório."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_checkout",
                "description": "Alterna entre branches ou restaura arquivos no repositório seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Nome da branch ou arquivo para checkout."
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório."
                        },
                        "create_branch": {
                            "type": "boolean",
                            "description": "Se verdadeiro, cria uma nova branch com a flag -b."
                        }
                    },
                    "required": ["target"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_git_show",
                "description": "Exibe informações e estatísticas detalhadas de um commit específico.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "commit_hash": {
                            "type": "string",
                            "description": "Hash identificador do commit."
                        },
                        "repo_path": {
                            "type": "string",
                            "description": "Caminho relativo do repositório."
                        }
                    },
                    "required": ["commit_hash"]
                }
            }
        }
    ]


# =========================================================================
# 5. FLUXO EXECUTOR AUTÔNOMO (execute_mcp_git_flow)
# =========================================================================

async def execute_mcp_git_flow(
    user_prompt: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executa comandos Git com inteligência contextual e formatação clara para o usuário.
    Analisa o pedido do usuário, determina a ferramenta Git apropriada, executa e formata a resposta.
    """
    t0 = time.time()
    tools_called = []
    p_lower = user_prompt.lower()

    # Heurística rápida direta para comandos óbvios (0ms de latência de LLM)
    is_direct_status = any(k in p_lower for k in ["git status", "status do git", "status do repositório", "status do repositorio", "arquivos modificados no git"])
    is_direct_log = any(k in p_lower for k in ["git log", "histórico de commits", "historico de commits", "ultimos commits", "últimos commits"])
    is_direct_diff = any(k in p_lower for k in ["git diff", "diff do git", "o que mudou no git", "diferenças no git"])
    is_direct_branches = any(k in p_lower for k in ["git branch", "branches do git", "quais as branches"])
    is_direct_init = any(k in p_lower for k in ["git init", "inicialize o git", "inicie o git", "criar repositorio git", "criar repositório git"])

    if is_direct_status:
        res = git_status_impl()
        tools_called.append({"tool": "mcp_git_status", "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 📦 MCP Git: Status do Repositório\n\n```bash\n{res}\n```", tools_called

    if is_direct_log:
        res = git_log_impl(max_count=10)
        tools_called.append({"tool": "mcp_git_log", "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 📜 MCP Git: Histórico de Commits\n\n```bash\n{res}\n```", tools_called

    if is_direct_diff:
        res = git_diff_impl()
        tools_called.append({"tool": "mcp_git_diff", "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 🔍 MCP Git: Diff de Alterações\n\n```diff\n{res}\n```", tools_called

    if is_direct_branches:
        res = git_branches_impl()
        tools_called.append({"tool": "mcp_git_branches", "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 🌿 MCP Git: Branches\n\n```bash\n{res}\n```", tools_called

    if is_direct_init:
        res = git_init_impl()
        tools_called.append({"tool": "mcp_git_init", "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 🚀 MCP Git: Inicialização\n\n{res}", tools_called

    # Se envolver commit com mensagem explícita
    commit_match = re.search(r'(?:git\s+commit\s+(?:-m\s+)?|commite\s+(?:com\s+a\s+mensagem\s+)?|faça\s+um\s+commit\s+)(?:["\'])(.*?)(?:["\'])', user_prompt, re.IGNORECASE)
    if commit_match:
        msg = commit_match.group(1).strip()
        res = git_commit_impl(message=msg)
        tools_called.append({"tool": "mcp_git_commit", "arguments": {"message": msg}, "status": "success", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 💾 MCP Git: Novo Commit\n\n{res}", tools_called

    # Fluxo Inteligente com LLM Executor
    from config import LITELLM_BASE_URL, LITELLM_API_KEY
    from langchain_openai import ChatOpenAI
    import json

    system_prompt = """Você é o Assistente Especialista em MCP Git.
Sua missão é interpretar a intenção do usuário sobre controle de versão Git no workspace seguro.
Decida qual ferramenta Git executar entre:
- mcp_git_status: Para inspecionar status e modificações.
- mcp_git_diff: Para examinar diffs de arquivos.
- mcp_git_log: Para listar histórico de commits.
- mcp_git_commit: Para commitar alterações com mensagem semântica.
- mcp_git_branches: Para listar branches.
- mcp_git_init: Para inicializar o repositório.
- mcp_git_checkout: Para trocar de branch ou restaurar arquivo.
- mcp_git_show: Para detalhar um commit específico.

Responda OBRIGATORIAMENTE em JSON válido com:
{
  "tool": "nome_da_funcao",
  "arguments": {
    "repo_path": "",
    ...
  },
  "explanation": "resumo do que será feito"
}"""

    try:
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model="Classifier_Fast",
            temperature=0.0,
            timeout=12.0
        )
        resp = await llm.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        content = resp.content.strip()
        clean_json = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json, flags=re.MULTILINE).strip()
        parsed = json.loads(clean_json)

        tool_name = parsed.get("tool", "mcp_git_status")
        args = parsed.get("arguments", {})

        if tool_name == "mcp_git_diff":
            output = git_diff_impl(
                repo_path=args.get("repo_path", ""),
                target=args.get("target", ""),
                staged=bool(args.get("staged", False))
            )
            fmt = "diff"
        elif tool_name == "mcp_git_log":
            output = git_log_impl(
                repo_path=args.get("repo_path", ""),
                max_count=args.get("max_count", 10)
            )
            fmt = "bash"
        elif tool_name == "mcp_git_commit":
            output = git_commit_impl(
                message=args.get("message", "Atualização via MCP Git"),
                repo_path=args.get("repo_path", ""),
                files=args.get("files")
            )
            fmt = "markdown"
        elif tool_name == "mcp_git_branches":
            output = git_branches_impl(repo_path=args.get("repo_path", ""))
            fmt = "bash"
        elif tool_name == "mcp_git_init":
            output = git_init_impl(repo_path=args.get("repo_path", ""))
            fmt = "markdown"
        elif tool_name == "mcp_git_checkout":
            output = git_checkout_impl(
                target=args.get("target", "main"),
                repo_path=args.get("repo_path", ""),
                create_branch=bool(args.get("create_branch", False))
            )
            fmt = "bash"
        elif tool_name == "mcp_git_show":
            output = git_show_impl(
                commit_hash=args.get("commit_hash", "HEAD"),
                repo_path=args.get("repo_path", "")
            )
            fmt = "diff"
        else:
            output = git_status_impl(repo_path=args.get("repo_path", ""))
            fmt = "bash"

        tools_called.append({
            "tool": tool_name,
            "arguments": args,
            "status": "success",
            "duration_seconds": round(time.time() - t0, 3)
        })

        if fmt == "markdown":
            result_md = f"### 📦 MCP Git Operação Concluída\n\n{output}"
        else:
            result_md = f"### 📦 MCP Git: `{tool_name}`\n\n```{fmt}\n{output}\n```"

        return result_md, tools_called

    except Exception as err:
        print(f"[Aviso MCP Git LLM]: {err}. Executando status como fallback.")
        fallback_res = git_status_impl()
        tools_called.append({"tool": "mcp_git_status", "status": "fallback", "duration_seconds": round(time.time() - t0, 3)})
        return f"### 📦 MCP Git: Status do Workspace\n\n```bash\n{fallback_res}\n```", tools_called
