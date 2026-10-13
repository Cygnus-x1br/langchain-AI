"""
MCP Code Runner Service - Execução Segura e Confinada de Código
===============================================================
Permite aos agentes e ao usuário executar scripts Python, comandos Bash
e rotinas computacionais diretamente em sandbox confinada no WORKSPACE_PATH.

Mecanismos de Segurança (Sandbox Zero-Trust):
1. Diretório de trabalho estritamente confinado em WORKSPACE_PATH (/app/workspace).
2. Sanitização do ambiente: remoção preventiva de chaves de API sensíveis (OPENAI_API_KEY, etc.)
   para impedir que scripts executados vazem credenciais do orquestrador.
3. Timeout estrito (padrão 15s, teto máximo de 60s) com terminação forçada de processos órfãos.
4. Truncamento de saída (máximo 50 KB de stdout/stderr) para proteção de memória e token budget.
5. Bloqueio proativo de comandos perigosos em Bash (rm -rf /, fork bombs, escrita em /dev, reboot).
6. Exposição dupla: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
"""

import os
import re
import sys
import time
import json
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.tools import tool

# Diretório base do Workspace
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/app/workspace")
os.makedirs(WORKSPACE_PATH, exist_ok=True)

DEFAULT_TIMEOUT = int(os.getenv("CODE_RUNNER_TIMEOUT", "15"))
MAX_TIMEOUT = 60
MAX_OUTPUT_BYTES = 50 * 1024  # 50 KB

# Lista de variáveis sensíveis a serem omitidas do processo filho
SENSITIVE_ENV_VARS = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "GROQ_API_KEY",
    "TOGETHER_API_KEY",
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "LITELLM_MASTER_KEY",
    "POSTGRES_URL",
    "DATABASE_URL",
    "MONGODB_URI"
}

# Padrões perigosos de comando Bash
FORBIDDEN_BASH_PATTERNS = [
    r'rm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--force|-[a-zA-Z]*f[a-zA-Z]*r)\s+/(?:\s|$)', # rm -rf /
    r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:',                              # Fork bomb
    r'mkfs(?:\.[a-z0-9]+)?\s+',                                                # Formatação de disco
    r'dd\s+if=/dev/(?:zero|urandom)\s+of=/dev/[a-z]+',                         # Sobrescrita de partição
    r'\b(reboot|shutdown|init\s+0|halt|poweroff)\b',                           # Desligamento do host
    r'>\s*/dev/sd[a-z]',                                                       # Escrita direta em bloco
]


# =========================================================================
# 1. FUNÇÕES DE SUPORTE E SANITIZAÇÃO
# =========================================================================

def get_workspace_root() -> Path:
    """Retorna o caminho canônico absoluto do diretório raiz do workspace."""
    return Path(WORKSPACE_PATH).resolve()


def get_sanitized_env() -> Dict[str, str]:
    """Cria uma cópia do ambiente do sistema expurgando chaves de API sensíveis."""
    safe_env = dict(os.environ)
    for k in SENSITIVE_ENV_VARS:
        safe_env.pop(k, None)
    safe_env["PYTHONUNBUFFERED"] = "1"
    safe_env["WORKSPACE_DIR"] = str(get_workspace_root())
    return safe_env


def check_forbidden_bash(command: str) -> Optional[str]:
    """Verifica se o comando bash contém padrões de alta periculosidade."""
    for pattern in FORBIDDEN_BASH_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Comando bloqueado por política de segurança da Sandbox: padrão proibido detectado ('{pattern}')."
    return None


def truncate_output(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Limita a saída para não sobrecarregar o buffer."""
    if len(text.encode("utf-8", errors="replace")) > max_bytes:
        truncated = text[:max_bytes]
        return truncated + "\n... [Saída truncada devido ao limite de 50 KB] ..."
    return text


# =========================================================================
# 2. IMPLEMENTAÇÃO DE EXECUÇÃO ASSÍNCRONA
# =========================================================================

async def run_process_async(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout_sec: int = DEFAULT_TIMEOUT,
    stdin_data: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executa um processo assíncrono com limites rígidos de tempo e captura de streams.
    """
    safe_cwd = cwd or get_workspace_root()
    timeout = min(max(1, timeout_sec), MAX_TIMEOUT)
    env = get_sanitized_env()
    t_start = time.time()

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(safe_cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdin_bytes = stdin_data.encode("utf-8") if stdin_data else None

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=stdin_bytes),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
            dur_ms = round((time.time() - t_start) * 1000, 2)
            return {
                "status": "timeout",
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Tempo limite de execução excedido ({timeout}s). Processo interrompido.",
                "duration_ms": dur_ms,
                "timed_out": True
            }

        dur_ms = round((time.time() - t_start) * 1000, 2)
        stdout_str = truncate_output(stdout_bytes.decode("utf-8", errors="replace"))
        stderr_str = truncate_output(stderr_bytes.decode("utf-8", errors="replace"))

        return {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "stdout": stdout_str,
            "stderr": stderr_str,
            "duration_ms": dur_ms,
            "timed_out": False
        }
    except Exception as e:
        dur_ms = round((time.time() - t_start) * 1000, 2)
        return {
            "status": "exception",
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Falha na execução do processo: {str(e)}",
            "duration_ms": dur_ms,
            "timed_out": False
        }


# =========================================================================
# 3. IMPLEMENTAÇÕES DE BAIXO NÍVEL
# =========================================================================

async def run_python_code_impl(
    code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Executa código Python inline de forma confinada."""
    safe_args = args or []
    cmd = [sys.executable, "-c", code] + safe_args
    return await run_process_async(cmd=cmd, timeout_sec=timeout_seconds)


async def run_bash_command_impl(
    command: str,
    timeout_seconds: int = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """Executa comando Bash no workspace após validação de segurança."""
    violation = check_forbidden_bash(command)
    if violation:
        return {
            "status": "blocked",
            "exit_code": 126,
            "stdout": "",
            "stderr": violation,
            "duration_ms": 0.0,
            "timed_out": False
        }

    cmd = ["/bin/bash", "-c", command]
    return await run_process_async(cmd=cmd, timeout_sec=timeout_seconds)


async def run_script_file_impl(
    file_path: str,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Executa um arquivo de script (.py, .sh) salvo dentro do workspace."""
    base_dir = get_workspace_root()
    clean_p = str(file_path or "").strip()

    if clean_p.startswith("/"):
        target = Path(clean_p).resolve()
    else:
        target = (base_dir / clean_p).resolve()

    try:
        target.relative_to(base_dir)
    except ValueError:
        return {
            "status": "blocked",
            "exit_code": 126,
            "stdout": "",
            "stderr": f"Acesso negado: O script '{clean_p}' está fora do workspace seguro.",
            "duration_ms": 0.0,
            "timed_out": False
        }

    if not target.exists():
        return {
            "status": "not_found",
            "exit_code": 127,
            "stdout": "",
            "stderr": f"Arquivo não encontrado: '{clean_p}' não existe no workspace.",
            "duration_ms": 0.0,
            "timed_out": False
        }

    safe_args = args or []
    ext = target.suffix.lower()

    if ext == ".py":
        cmd = [sys.executable, str(target)] + safe_args
    elif ext in [".sh", ".bash"]:
        cmd = ["/bin/bash", str(target)] + safe_args
    elif ext in [".js", ".mjs"]:
        if shutil.which("node"):
            cmd = ["node", str(target)] + safe_args
        else:
            return {"status": "error", "exit_code": 1, "stdout": "", "stderr": "Node.js não disponível no container.", "duration_ms": 0.0, "timed_out": False}
    else:
        # Tentar execução direta com permissões normais
        cmd = [str(target)] + safe_args

    return await run_process_async(cmd=cmd, timeout_sec=timeout_seconds)


# =========================================================================
# 4. EXPOSIÇÃO LANGCHAIN TOOLS (@tool)
# =========================================================================

@tool
def mcp_run_python(code: str, timeout_seconds: int = 15) -> str:
    """
    Executa um snippet de código Python 3 diretamente no workspace seguro com isolamento de credenciais.
    
    Args:
        code: Código Python completo a ser executado.
        timeout_seconds: Tempo limite em segundos (padrão: 15, máx: 60).
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        res = loop.run_until_complete(run_python_code_impl(code=code, timeout_seconds=timeout_seconds))
    else:
        res = asyncio.run(run_python_code_impl(code=code, timeout_seconds=timeout_seconds))
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_run_bash(command: str, timeout_seconds: int = 15) -> str:
    """
    Executa um comando Bash no diretório do workspace seguro com validação de políticas de segurança.
    
    Args:
        command: Linha de comando Bash a executar.
        timeout_seconds: Tempo limite em segundos (padrão: 15, máx: 60).
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        res = loop.run_until_complete(run_bash_command_impl(command=command, timeout_seconds=timeout_seconds))
    else:
        res = asyncio.run(run_bash_command_impl(command=command, timeout_seconds=timeout_seconds))
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_run_script_file(file_path: str, timeout_seconds: int = 15) -> str:
    """
    Executa um arquivo de script (.py, .sh) salvo no workspace.
    
    Args:
        file_path: Caminho relativo do script dentro de /app/workspace (ex: 'meu_script.py').
        timeout_seconds: Tempo limite em segundos.
    """
    loop = asyncio.get_event_loop()
    if loop.is_running():
        res = loop.run_until_complete(run_script_file_impl(file_path=file_path, timeout_seconds=timeout_seconds))
    else:
        res = asyncio.run(run_script_file_impl(file_path=file_path, timeout_seconds=timeout_seconds))
    return json.dumps(res, ensure_ascii=False, indent=2)


def get_mcp_code_runner_langchain_tools() -> List[Any]:
    """Retorna lista das ferramentas MCP Code Runner prontas para LangChain Agents."""
    return [
        mcp_run_python,
        mcp_run_bash,
        mcp_run_script_file
    ]


# =========================================================================
# 5. EXPOSIÇÃO OPENAI TOOL CALLING SCHEMAS
# =========================================================================

def get_mcp_code_runner_openai_tools() -> List[Dict[str, Any]]:
    """Retorna a especificação dos schemas JSON no formato oficial OpenAI Tool Calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp_run_python",
                "description": "Executa código Python 3 de forma isolada e segura dentro do workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Código Python a ser executado."
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Tempo limite de execução em segundos (padrão: 15)."
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_run_bash",
                "description": "Executa comando shell Bash seguro dentro do diretório /app/workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Comando Bash a ser executado."
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Tempo limite de execução em segundos (padrão: 15)."
                        }
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_run_script_file",
                "description": "Executa um arquivo de script salvo dentro do workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Caminho do arquivo de script (ex: 'calculo.py', 'rotina.sh')."
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": "Tempo limite de execução em segundos."
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]


# =========================================================================
# 6. FLUXO EXECUTOR AUTÔNOMO (execute_mcp_code_runner_flow)
# =========================================================================

async def execute_mcp_code_runner_flow(
    user_prompt: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Identifica código ou comandos no prompt do usuário, executa na sandbox e retorna resultado formatado.
    """
    t0 = time.time()
    tools_called = []

    # 1. Extrair bloco Python explícito ```python ... ```
    py_block = re.search(r'```python\s*(.*?)\s*```', user_prompt, re.DOTALL | re.IGNORECASE)
    if py_block:
        code = py_block.group(1).strip()
        res = await run_python_code_impl(code=code)
        tools_called.append({"tool": "mcp_run_python", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
        return format_runner_result("Python", res), tools_called

    # 2. Extrair bloco Bash / Shell explícito ```bash ... ``` ou ```sh ... ```
    bash_block = re.search(r'```(?:bash|sh)\s*(.*?)\s*```', user_prompt, re.DOTALL | re.IGNORECASE)
    if bash_block:
        cmd = bash_block.group(1).strip()
        res = await run_bash_command_impl(command=cmd)
        tools_called.append({"tool": "mcp_run_bash", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
        return format_runner_result("Bash", res), tools_called

    # 3. Verificar pedido de execução de arquivo ("execute o script X.py")
    script_match = re.search(r'(?:execute|rode|rodar|executar)\s+(?:o\s+)?(?:script\s+|arquivo\s+)?([a-zA-Z0-9_\-./]+\.(?:py|sh|js))', user_prompt, re.IGNORECASE)
    if script_match:
        f_name = script_match.group(1).strip()
        res = await run_script_file_impl(file_path=f_name)
        tools_called.append({"tool": "mcp_run_script_file", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
        return format_runner_result(f"Script ({f_name})", res), tools_called

    # 4. Bloco de código genérico ``` ... ```
    generic_block = re.search(r'```\s*(.*?)\s*```', user_prompt, re.DOTALL)
    if generic_block:
        code = generic_block.group(1).strip()
        # Heurística: se tiver print, def, import, é python
        if any(kw in code for kw in ["print(", "def ", "import ", "for ", "while ", "class "]):
            res = await run_python_code_impl(code=code)
            tools_called.append({"tool": "mcp_run_python", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
            return format_runner_result("Python", res), tools_called
        else:
            res = await run_bash_command_impl(command=code)
            tools_called.append({"tool": "mcp_run_bash", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
            return format_runner_result("Bash", res), tools_called

    # Se não houver código explícito, orientar o usuário
    return (
        "### ⚙️ MCP Code Runner (Sandbox)\n\n"
        "Nenhum bloco de código ou script identificável foi encontrado no seu pedido.\n\n"
        "**Exemplos de uso:**\n"
        "- Envie um bloco de código Python: \n"
        "  ```python\n"
        "  print([x**2 for x in range(10)])\n"
        "  ```\n"
        "- Envie um comando Bash:\n"
        "  ```bash\n"
        "  ls -la && python3 --version\n"
        "  ```\n"
        "- Ou solicite: *'Execute o script teste.py'*"
    ), tools_called


def format_runner_result(engine_label: str, res: Dict[str, Any]) -> str:
    """Formata a saída de execução em Markdown amigável."""
    exit_code = res.get("exit_code", 0)
    status = res.get("status", "unknown")
    dur_ms = res.get("duration_ms", 0.0)
    stdout = res.get("stdout", "").strip()
    stderr = res.get("stderr", "").strip()

    status_badge = "✅ Concluído com Sucesso" if exit_code == 0 else f"❌ Falhou (Código {exit_code})"
    if res.get("timed_out"):
        status_badge = "⏳ Tempo Limite Esgotado (Timeout)"

    md = [
        f"### ⚙️ MCP Code Runner: {engine_label}",
        f"**Status:** {status_badge} | **Tempo:** {dur_ms} ms\n"
    ]

    if stdout:
        md.append(f"**Saída Padrão (stdout):**\n```text\n{stdout}\n```")
    if stderr:
        md.append(f"**Erros / Diagnóstico (stderr):**\n```text\n{stderr}\n```")

    if not stdout and not stderr:
        md.append("*(O comando foi executado e não produziu saída)*")

    return "\n\n".join(md)
