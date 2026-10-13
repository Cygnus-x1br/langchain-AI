"""
MCP Filesystem Service - Sandbox de Manipulação Segura de Arquivos
===================================================================
Permite aos agentes (CODING, agent-orchestrator e ferramentas especializadas)
ler, escrever, listar, buscar e manipular arquivos de forma 100% segura e
estritamente confinada dentro do diretório WORKSPACE_PATH (/app/workspace).

Mecanismos de Segurança (Sandbox Zero-Trust):
1. Resolução Canônica com 'Path.resolve()' e verificação rigorosa 'is_relative_to()'.
2. Bloqueio absoluto de Path Traversal ('../', caminhos absolutos para /etc, /app, etc).
3. Bloqueio de deleção da raiz do workspace.
4. Limites de segurança de leitura (max 5 MB) para evitar estouro de memória (OOM).
5. Exposição em dois formatos: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
"""

import os
import re
import glob
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.tools import tool

# Diretório raiz seguro do Sandbox
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/app/workspace")
os.makedirs(WORKSPACE_PATH, exist_ok=True)

MAX_READ_BYTES = 5 * 1024 * 1024  # 5 MB


# =========================================================================
# 1. FUNÇÕES DE SEGURANÇA E RESOLUÇÃO CANÔNICA
# =========================================================================

def get_workspace_root() -> Path:
    """Retorna o caminho canônico absoluto do diretório raiz do workspace."""
    return Path(WORKSPACE_PATH).resolve()


def resolve_safe_path(rel_or_abs_path: str, must_exist: bool = False) -> Path:
    """
    Resolve e valida o caminho solicitado contra a raiz do workspace seguro.
    Garante que qualquer tentativa de Path Traversal ('../' ou caminhos absolutos fora da sandbox)
    seja interceptada com PermissionError antes de qualquer operação de I/O.
    """
    base_dir = get_workspace_root()
    clean_str = str(rel_or_abs_path or "").strip()

    if not clean_str or clean_str in [".", "./"]:
        target = base_dir
    elif clean_str == "/":
        target = base_dir
    elif clean_str.startswith("/"):
        # Se for caminho absoluto, só é permitido se apontar explicitamente para dentro do workspace
        target = Path(clean_str).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            raise PermissionError(
                f"Acesso Negado (Violação de Sandbox): O caminho absoluto '{rel_or_abs_path}' "
                f"está fora do diretório seguro ({base_dir}). Use apenas caminhos dentro da sandbox."
            )
    else:
        # Caminho relativo: resolve a partir de base_dir e impede escape via '../'
        target = (base_dir / clean_str).resolve()
        try:
            target.relative_to(base_dir)
        except ValueError:
            raise PermissionError(
                f"Acesso Negado (Violação de Sandbox): O caminho '{rel_or_abs_path}' "
                f"tenta escapar da sandbox ({base_dir}) através de navegação relativa ('../')."
            )

    if must_exist and not target.exists():
        raise FileNotFoundError(f"Arquivo ou pasta não encontrado no workspace: '{rel_or_abs_path}'")

    return target


def get_relative_display_path(p: Path) -> str:
    """Converte um Path canônico em caminho relativo ao workspace para exibição limpa."""
    base_dir = get_workspace_root()
    try:
        rel = p.relative_to(base_dir)
        return str(rel) if str(rel) != "." else "/"
    except ValueError:
        return str(p)


# =========================================================================
# 2. OPERAÇÕES DE ARQUIVO PURAS (CORE SERVICE)
# =========================================================================

def core_read_file(path: str, offset: int = 0, limit: int = 50000) -> Dict[str, Any]:
    """Lê o conteúdo textual de um arquivo dentro do workspace seguro."""
    target = resolve_safe_path(path, must_exist=True)

    if target.is_dir():
        raise IsADirectoryError(f"'{get_relative_display_path(target)}' é um diretório, não um arquivo.")

    size = target.stat().st_size
    if size > MAX_READ_BYTES and limit > MAX_READ_BYTES:
        raise ValueError(f"Arquivo muito grande ({size} bytes). Limite máximo de leitura é de 5 MB.")

    with open(target, "r", encoding="utf-8", errors="replace") as f:
        if offset > 0:
            f.seek(offset)
        content = f.read(limit)

    return {
        "success": True,
        "path": get_relative_display_path(target),
        "size_bytes": size,
        "offset": offset,
        "content": content,
        "truncated": (offset + len(content)) < size
    }


def core_write_file(path: str, content: str) -> Dict[str, Any]:
    """Escreve ou sobrescreve um arquivo dentro do workspace seguro."""
    target = resolve_safe_path(path)
    base_dir = get_workspace_root()

    if target == base_dir:
        raise PermissionError("Não é permitido sobrescrever o diretório raiz do workspace.")

    # Cria diretórios pais automaticamente
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        bytes_written = f.write(content)

    return {
        "success": True,
        "path": get_relative_display_path(target),
        "bytes_written": bytes_written,
        "message": f"Arquivo '{get_relative_display_path(target)}' salvo com sucesso ({bytes_written} caracteres)."
    }


def core_append_file(path: str, content: str) -> Dict[str, Any]:
    """Adiciona texto ao final de um arquivo dentro do workspace seguro."""
    target = resolve_safe_path(path)
    base_dir = get_workspace_root()

    if target == base_dir:
        raise PermissionError("Não é permitido anexar ao diretório raiz do workspace.")

    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "a", encoding="utf-8") as f:
        bytes_written = f.write(content)

    return {
        "success": True,
        "path": get_relative_display_path(target),
        "bytes_appended": bytes_written,
        "message": f"Conteúdo anexado com sucesso ao arquivo '{get_relative_display_path(target)}'."
    }


def core_list_directory(path: str = "", recursive: bool = False, max_depth: int = 2) -> Dict[str, Any]:
    """Lista arquivos e subdiretórios dentro do workspace seguro."""
    target = resolve_safe_path(path, must_exist=True)

    if not target.is_dir():
        raise NotADirectoryError(f"'{get_relative_display_path(target)}' não é um diretório.")

    items = []
    base_dir = get_workspace_root()

    def scan_dir(current_dir: Path, current_depth: int):
        for entry in current_dir.iterdir():
            try:
                stat = entry.stat()
                is_directory = entry.is_dir()
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                rel_path = str(entry.relative_to(base_dir))

                item_info = {
                    "name": entry.name,
                    "path": rel_path,
                    "is_dir": is_directory,
                    "size_bytes": stat.st_size if not is_directory else 0,
                    "modified": mod_time
                }
                items.append(item_info)

                if recursive and is_directory and current_depth < max_depth:
                    scan_dir(entry, current_depth + 1)
            except Exception as item_err:
                items.append({"name": entry.name, "error": str(item_err)})

    scan_dir(target, 1)

    # Ordena diretórios primeiro, depois alfabético
    items.sort(key=lambda x: (not x.get("is_dir", False), x.get("name", "").lower()))

    return {
        "success": True,
        "base_path": get_relative_display_path(target),
        "total_items": len(items),
        "items": items
    }


def core_create_directory(path: str) -> Dict[str, Any]:
    """Cria uma nova pasta dentro do workspace seguro."""
    target = resolve_safe_path(path)
    base_dir = get_workspace_root()

    if target == base_dir:
        return {"success": True, "path": "/", "message": "Diretório raiz já existe."}

    target.mkdir(parents=True, exist_ok=True)
    return {
        "success": True,
        "path": get_relative_display_path(target),
        "message": f"Diretório '{get_relative_display_path(target)}' criado com sucesso."
    }


def core_delete_file(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Remove um arquivo ou pasta dentro do workspace seguro."""
    target = resolve_safe_path(path, must_exist=True)
    base_dir = get_workspace_root()

    if target == base_dir:
        raise PermissionError("Operação Proibida: Não é permitido remover o diretório raiz do workspace.")

    rel_name = get_relative_display_path(target)

    if target.is_dir():
        if recursive:
            import shutil
            shutil.rmtree(target)
            return {"success": True, "path": rel_name, "message": f"Diretório '{rel_name}' e seu conteúdo foram removidos."}
        else:
            target.rmdir()
            return {"success": True, "path": rel_name, "message": f"Diretório vazio '{rel_name}' removido com sucesso."}
    else:
        target.unlink()
        return {"success": True, "path": rel_name, "message": f"Arquivo '{rel_name}' removido com sucesso."}


def core_get_file_info(path: str) -> Dict[str, Any]:
    """Retorna metadados detalhados de um arquivo ou pasta no workspace seguro."""
    target = resolve_safe_path(path, must_exist=True)
    stat = target.stat()

    return {
        "success": True,
        "path": get_relative_display_path(target),
        "is_file": target.is_file(),
        "is_dir": target.is_dir(),
        "size_bytes": stat.st_size,
        "extension": target.suffix,
        "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }


def core_search_files(pattern: str = "*", query_text: Optional[str] = None, subpath: str = "") -> Dict[str, Any]:
    """Busca arquivos por padrão glob (ex: *.py) e opcionalmente por ocorrência de texto dentro do arquivo."""
    base_target = resolve_safe_path(subpath, must_exist=True)
    base_dir = get_workspace_root()

    matches = []
    # Busca recursiva com rglob
    for p in base_target.rglob(pattern):
        if p.is_file():
            rel_p = str(p.relative_to(base_dir))
            matched_query = False
            snippet = None

            if query_text:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if query_text.lower() in line.lower():
                                matched_query = True
                                snippet = f"L{line_num}: {line.strip()[:160]}"
                                break
                except Exception:
                    pass
            else:
                matched_query = True

            if matched_query:
                matches.append({
                    "path": rel_p,
                    "size_bytes": p.stat().st_size,
                    "matched_snippet": snippet
                })

    return {
        "success": True,
        "pattern": pattern,
        "query_text": query_text,
        "total_matches": len(matches),
        "matches": matches[:100]  # Limita para até 100 resultados
    }


# =========================================================================
# 3. FERRAMENTAS LANGCHAIN (@tool) PARA AGENTES E ORQUESTRADOR
# =========================================================================

@tool
def mcp_read_file(path: str) -> str:
    """Lê o conteúdo de um arquivo dentro do workspace seguro (/app/workspace). Informe o caminho relativo do arquivo."""
    try:
        res = core_read_file(path)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_write_file(path: str, content: str) -> str:
    """Cria ou sobrescreve um arquivo dentro do workspace seguro (/app/workspace). Cria pastas necessárias automaticamente."""
    try:
        res = core_write_file(path, content)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_append_file(path: str, content: str) -> str:
    """Adiciona texto ao final de um arquivo dentro do workspace seguro (/app/workspace)."""
    try:
        res = core_append_file(path, content)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_list_directory(path: str = "") -> str:
    """Lista arquivos e subpastas dentro do workspace seguro (/app/workspace). Deixe vazio para listar a raiz do workspace."""
    try:
        res = core_list_directory(path)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_create_directory(path: str) -> str:
    """Cria uma nova pasta dentro do workspace seguro (/app/workspace)."""
    try:
        res = core_create_directory(path)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_delete_file(path: str, recursive: bool = False) -> str:
    """Remove um arquivo ou pasta dentro do workspace seguro (/app/workspace). Use recursive=True para pastas não vazias."""
    try:
        res = core_delete_file(path, recursive=recursive)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_search_files(pattern: str = "*", query: str = "", path: str = "") -> str:
    """Pesquisa arquivos por padrão de nome (ex: *.py, *.json) ou por texto dentro dos arquivos no workspace seguro."""
    try:
        res = core_search_files(pattern=pattern, query_text=query if query else None, subpath=path)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


@tool
def mcp_get_file_info(path: str) -> str:
    """Obtém informações detalhadas e metadados (tamanho, data de modificação, extensão) de um arquivo ou pasta no workspace."""
    try:
        res = core_get_file_info(path)
        return json.dumps(res, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


def get_mcp_filesystem_langchain_tools() -> List[Any]:
    """Retorna a lista de todas as ferramentas de Filesystem no padrão LangChain Tool."""
    return [
        mcp_read_file,
        mcp_write_file,
        mcp_append_file,
        mcp_list_directory,
        mcp_create_directory,
        mcp_delete_file,
        mcp_search_files,
        mcp_get_file_info
    ]


# =========================================================================
# 4. DEFINIÇÕES NO PADRÃO OPENAI TOOL CALLING (JSON SCHEMA)
# =========================================================================

def get_mcp_filesystem_openai_tools() -> List[Dict[str, Any]]:
    """
    Retorna as especificações das ferramentas no formato oficial de Tools da OpenAI / LiteLLM,
    prontas para serem injetadas em payloads 'tools' de chat/completions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp_read_file",
                "description": "Lê o conteúdo textual de um arquivo dentro do workspace seguro (/app/workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho relativo do arquivo dentro do workspace (ex: 'app.py' ou 'src/utils.js')."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_write_file",
                "description": "Cria ou sobrescreve um arquivo no workspace seguro (/app/workspace). Cria pastas necessárias automaticamente.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho relativo do arquivo no workspace (ex: 'main.py' ou 'docs/setup.md')."
                        },
                        "content": {
                            "type": "string",
                            "description": "Conteúdo textual completo a ser gravado no arquivo."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_append_file",
                "description": "Adiciona texto ao final de um arquivo existente dentro do workspace seguro (/app/workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho relativo do arquivo no workspace."
                        },
                        "content": {
                            "type": "string",
                            "description": "Conteúdo textual a ser adicionado ao final do arquivo."
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_list_directory",
                "description": "Lista arquivos e pastas dentro do workspace seguro (/app/workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Subpasta relativa a listar (deixe vazio ou '.' para a raiz do workspace)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_create_directory",
                "description": "Cria um novo diretório dentro do workspace seguro (/app/workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho da pasta a ser criada dentro do workspace."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_delete_file",
                "description": "Remove um arquivo ou pasta dentro do workspace seguro (/app/workspace).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho relativo do arquivo ou pasta a ser removido."
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Se True, remove pastas não-vazias e todo o seu conteúdo recursivamente."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_search_files",
                "description": "Busca arquivos por padrão de nome (ex: *.py) ou por conteúdo de texto dentro do workspace seguro.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Padrão glob para filtrar arquivos (padrão: '*'). Ex: '*.py', '*.json'."
                        },
                        "query": {
                            "type": "string",
                            "description": "Texto opcional para buscar dentro dos arquivos correspondentes."
                        },
                        "path": {
                            "type": "string",
                            "description": "Subpasta onde a busca deve ser iniciada (padrão: raiz do workspace)."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_get_file_info",
                "description": "Obtém metadados (tamanho, data de modificação, extensão) de um arquivo ou pasta dentro do workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Caminho relativo do arquivo ou pasta a consultar."
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    ]


# =========================================================================
# 5. DESPACHANTE UNIFICADO DE EXECUÇÃO
# =========================================================================

async def execute_mcp_filesystem_action(tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Executa a ação correspondente do MCP Filesystem com isolamento de erros.
    Retorna: (sucesso: bool, texto_resposta: str, payload_log: dict)
    """
    t_start = time.time()
    name = tool_name.replace("functions.", "")

    try:
        if name in ["mcp_read_file", "read_file"]:
            res = core_read_file(args.get("path", ""))
        elif name in ["mcp_write_file", "write_file"]:
            res = core_write_file(args.get("path", ""), args.get("content", ""))
        elif name in ["mcp_append_file", "append_file"]:
            res = core_append_file(args.get("path", ""), args.get("content", ""))
        elif name in ["mcp_list_directory", "list_directory"]:
            res = core_list_directory(args.get("path", ""))
        elif name in ["mcp_create_directory", "create_directory"]:
            res = core_create_directory(args.get("path", ""))
        elif name in ["mcp_delete_file", "delete_file"]:
            res = core_delete_file(args.get("path", ""), recursive=args.get("recursive", False))
        elif name in ["mcp_search_files", "search_files"]:
            res = core_search_files(
                pattern=args.get("pattern", "*"),
                query_text=args.get("query"),
                subpath=args.get("path", "")
            )
        elif name in ["mcp_get_file_info", "get_file_info"]:
            res = core_get_file_info(args.get("path", ""))
        else:
            raise ValueError(f"Ferramenta desconhecida do MCP Filesystem: '{tool_name}'")

        dur = round(time.time() - t_start, 3)
        log_data = {
            "tool": name,
            "arguments": args,
            "status": "success",
            "duration_seconds": dur
        }
        return True, json.dumps(res, ensure_ascii=False), log_data

    except Exception as err:
        dur = round(time.time() - t_start, 3)
        err_msg = str(err)
        log_data = {
            "tool": name,
            "arguments": args,
            "status": f"error: {err_msg}",
            "duration_seconds": dur
        }
        return False, json.dumps({"success": False, "error": err_msg}, ensure_ascii=False), log_data


# =========================================================================
# 6. PIPELINE PRINCIPAL DO AGENTE MCP FILESYSTEM
# =========================================================================

async def execute_mcp_filesystem_flow(
    user_prompt: str,
    context_prompt: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pipeline mestre de orquestração do Agente com MCP Filesystem:
    1. Identifica a intenção (listar, ler, criar/escrever, deletar, buscar).
    2. Executa as ferramentas de sandbox com proteção anti-traversal.
    3. Se envolver criação de código ou conteúdo, utiliza o LLM do orquestrador
       para gerar o código completo e salvá-lo fisicamente em /app/workspace.
    4. Retorna resposta Markdown rica e o registro de auditoria das ferramentas.
    """
    flow_start = time.time()
    tools_called: List[Dict[str, Any]] = []
    # Isola o prompt real do usuário para evitar falso-positivo com histórico episódico injetado
    clean_user_prompt = user_prompt.split("\n\n[CONTEXTO")[0].split("\n[CONTEXTO")[0].strip()
    p_lower = clean_user_prompt.lower()

    from config import LITELLM_BASE_URL, LITELLM_API_KEY
    from langchain_openai import ChatOpenAI

    # -------------------------------------------------------------
    # CASO 1: Leitura de Arquivo Específico ("leia o arquivo X", "mostre X")
    # -------------------------------------------------------------
    is_read_intent = (
        any(k in p_lower for k in ["leia", "ler", "abra", "abrir", "mostre o arquivo", "conteúdo de", "conteudo de"])
        and not any(k in p_lower for k in ["crie", "criar", "salve", "salvar", "escreva", "escrever"])
    )
    if is_read_intent:
        t0 = time.time()
        file_match = re.search(r"['\"]?([\w\-_/]+\.[a-zA-Z0-9]+)['\"]?", clean_user_prompt)
        target_file = file_match.group(1) if file_match else ""

        if target_file:
            try:
                res = core_read_file(target_file)
                tools_called.append({
                    "tool": "mcp_read_file",
                    "arguments": {"path": target_file},
                    "status": "success",
                    "duration_seconds": round(time.time() - t0, 3)
                })

                content = res.get("content", "")
                ext = Path(target_file).suffix.lstrip(".").lower()
                lang_map = {
                    "py": "python", "js": "javascript", "ts": "typescript",
                    "json": "json", "md": "markdown", "html": "html",
                    "css": "css", "sh": "bash", "sql": "sql", "yaml": "yaml", "yml": "yaml"
                }
                lang = lang_map.get(ext, "")

                response_md = f"""### 📄 Conteúdo de `{res.get('path')}`
**Tamanho:** {res.get('size_bytes', 0):,} bytes | **Status:** Leitura Segura via MCP Filesystem

```{lang}
{content}
```
"""
                return response_md, tools_called

            except Exception as err:
                tools_called.append({
                    "tool": "mcp_read_file",
                    "arguments": {"path": target_file},
                    "status": f"error: {str(err)}",
                    "duration_seconds": round(time.time() - t0, 3)
                })
                return f"⚠️ **Não foi possível ler o arquivo `{target_file}`:** {str(err)}", tools_called

    # -------------------------------------------------------------
    # CASO 2: Listagem de Arquivos ("listar workspace", "quais arquivos", etc)
    # -------------------------------------------------------------
    is_list_intent = (
        any(k in p_lower for k in ["listar", "liste", "quais arquivos", "ver arquivos", "conteúdo do workspace", "conteudo do workspace"])
        and not any(k in p_lower for k in ["crie", "criar", "salve", "salvar", "escreva", "escrever"])
    )
    if is_list_intent or p_lower in ["workspace", "ls", "dir", "listar"]:
        t0 = time.time()
        # Identifica se o usuário especificou uma subpasta explícita (ex: 'na pasta src')
        subfolder_match = re.search(r"(?:na\s+pasta|no\s+diret[oó]rio|na\s+subpasta)\s+['\"]?([\w\-_/]+)['\"]?", clean_user_prompt, re.IGNORECASE)
        target_subfolder = subfolder_match.group(1).strip() if subfolder_match else ""
        if target_subfolder.lower() in ["workspace", "sandbox", ".", "./", "raiz", "root", "de"]:
            target_subfolder = ""
        t0 = time.time()
        # Identifica se o usuário especificou uma subpasta explícita (ex: 'na pasta src')
        subfolder_match = re.search(r"(?:na\s+pasta|no\s+diret[oó]rio|na\s+subpasta)\s+['\"]?([\w\-_/]+)['\"]?", user_prompt, re.IGNORECASE)
        target_subfolder = subfolder_match.group(1).strip() if subfolder_match else ""
        if target_subfolder.lower() in ["workspace", "sandbox", ".", "./", "raiz", "root", "de"]:
            target_subfolder = ""

        try:
            res = core_list_directory(target_subfolder, recursive=True)
            tools_called.append({
                "tool": "mcp_list_directory",
                "arguments": {"path": target_subfolder},
                "status": "success",
                "duration_seconds": round(time.time() - t0, 3)
            })

            items = res.get("items", [])
            if not items:
                msg = f"📂 **O diretório seguro do workspace (`/app/workspace/{target_subfolder}`) está atualmente vazio.**\n\nVocê pode me pedir para criar scripts, projetos ou arquivos aqui a qualquer momento!"
                return msg, tools_called

            rows = []
            for it in items:
                icon = "📁" if it.get("is_dir") else "📄"
                tipo = "Pasta" if it.get("is_dir") else "Arquivo"
                size_str = f"{it.get('size_bytes', 0):,} bytes" if not it.get("is_dir") else "-"
                rows.append(f"| {icon} `{it.get('path')}` | {tipo} | {size_str} | {it.get('modified')} |")

            table_md = "\n".join(rows)
            response_md = f"""### 📁 Arquivos no Workspace Seguro (`/app/workspace`)

| Item | Tipo | Tamanho | Última Modificação |
| :--- | :--- | :--- | :--- |
{table_md}

💡 *Total de itens encontrados: {len(items)}*
Para inspecionar o conteúdo de algum arquivo, basta pedir: `Leia o arquivo <nome>`."""
            return response_md, tools_called

        except Exception as err:
            tools_called.append({
                "tool": "mcp_list_directory",
                "arguments": {"path": target_subfolder},
                "status": f"error: {str(err)}",
                "duration_seconds": round(time.time() - t0, 3)
            })
            return f"⚠️ **Erro ao listar workspace:** {str(err)}", tools_called

    # -------------------------------------------------------------
    # CASO 3: Deleção de Arquivo ("delete X", "remova X")
    # -------------------------------------------------------------
    is_delete_intent = any(k in p_lower for k in ["delete", "deletar", "remova", "remover", "apague", "apagar", "exclua", "excluir"])
    if is_delete_intent:
        t0 = time.time()
        file_match = re.search(r"(?:arquivo|pasta|diret[oó]rio)?\s*['\"]?([\w\-_/]+(?:\.[a-zA-Z0-9]+)?)['\"]?", user_prompt, re.IGNORECASE)
        target_del = file_match.group(1) if file_match else ""

        if target_del:
            try:
                res = core_delete_file(target_del, recursive=True)
                tools_called.append({
                    "tool": "mcp_delete_file",
                    "arguments": {"path": target_del},
                    "status": "success",
                    "duration_seconds": round(time.time() - t0, 3)
                })
                return f"🗑️ **{res.get('message')}**", tools_called
            except Exception as err:
                tools_called.append({
                    "tool": "mcp_delete_file",
                    "arguments": {"path": target_del},
                    "status": f"error: {str(err)}",
                    "duration_seconds": round(time.time() - t0, 3)
                })
                return f"⚠️ **Erro ao excluir `{target_del}`:** {str(err)}", tools_called

    # -------------------------------------------------------------
    # CASO 4: Busca de Arquivos ("busque *.py", "pesquise X")
    # -------------------------------------------------------------
    is_search_intent = any(k in p_lower for k in ["busque", "buscar", "pesquise", "pesquisar", "procure", "procurar", "localize", "localizar"])
    if is_search_intent:
        t0 = time.time()
        pat_match = re.search(r"(\*\.[\w]+|[\w\-_\.]+\.[\w]+)", user_prompt)
        pattern = pat_match.group(1) if pat_match else "*"

        try:
            res = core_search_files(pattern=pattern, subpath="")
            tools_called.append({
                "tool": "mcp_search_files",
                "arguments": {"pattern": pattern},
                "status": "success",
                "duration_seconds": round(time.time() - t0, 3)
            })

            matches = res.get("matches", [])
            if not matches:
                return f"🔍 **Nenhum arquivo correspondente ao padrão `{pattern}` foi encontrado no workspace.**", tools_called

            items_str = "\n".join([f"- 📄 `{m.get('path')}` ({m.get('size_bytes', 0):,} bytes)" for m in matches[:30]])
            response_md = f"""### 🔍 Resultados da Busca no Workspace (`{pattern}`)
{items_str}

*Total de ocorrências: {len(matches)}*"""
            return response_md, tools_called
        except Exception as err:
            return f"⚠️ **Erro na busca:** {str(err)}", tools_called

    # -------------------------------------------------------------
    # CASO 5: Ajuda, Consulta de Ferramentas e Explicação do MCP Filesystem
    # -------------------------------------------------------------
    is_info_intent = any(k in p_lower for k in [
        "quais ferramentas", "quais recursos", "como funciona", "como usar",
        "ajuda", "comandos", "o que você pode fazer", "o que voce pode fazer",
        "quais operações", "quais operacoes", "ferramentas disponíveis",
        "ferramentas disponiveis", "ferramentas estao disponiveis", "ferramentas estão disponíveis",
        "manual", "documentação", "documentacao", "quais métodos", "quais metodos"
    ])
    is_explicit_creation = any(k in p_lower for k in ["crie", "criar", "salve", "salvar", "escreva", "escrever", "gere", "gerar", "faça", "desenvolva"])

    if is_info_intent and not is_explicit_creation:
        response_md = """### 🛠️ Ferramentas MCP Filesystem Ativas (`/app/workspace`)

O ambiente seguro de Sandbox isolado do orquestrador possui **8 ferramentas MCP oficiais** integradas diretamente ao diretório `/app/workspace`:

| Ferramenta | Descrição | Exemplo de Comando no Chat |
| :--- | :--- | :--- |
| 📖 `mcp_read_file` | Lê o conteúdo textual completo de um arquivo | *"Leia o arquivo app.py do workspace"* |
| 💾 `mcp_write_file` | Cria ou reescreve arquivos com código ou texto | *"Crie um script calculadora.py no workspace"* |
| ➕ `mcp_append_file` | Adiciona linhas ao final de um arquivo existente | *"Adicione 'NOVA_VAR=123' ao final do config.env"* |
| 📂 `mcp_list_directory` | Lista arquivos e subpastas detalhando tamanhos e datas | *"Liste os arquivos do workspace"* |
| 📁 `mcp_create_directory` | Cria subpastas estruturadas de forma segura | *"Crie a pasta src/services no workspace"* |
| 🗑️ `mcp_delete_file` | Remove arquivos ou diretórios com segurança | *"Remova o arquivo temp.log do workspace"* |
| 🔍 `mcp_search_files` | Busca por padrões de nome (*.py, *.json, etc.) | *"Busque arquivos *.py no workspace"* |
| ℹ️ `mcp_get_file_info` | Retorna metadados completos (tamanho, datas, tipo) | *"Verifique as informações do arquivo data.json"* |

🔒 **Segurança Sandbox:** Todas as operações são confinadas estritamente a `/app/workspace`. Qualquer tentativa de *Path Traversal* (`../` ou `/etc`) é bloqueada automaticamente com Zero-Trust."""
        return response_md, [{"tool": "mcp_help_info", "status": "success"}]

    # -------------------------------------------------------------
    # CASO 6: Criação / Escrita Inteligente de Código / Arquivo (Agentic Write)
    # -------------------------------------------------------------
    t0 = time.time()

    # Detecta ou infere o nome do arquivo desejado
    fname_match = re.search(r"['\"]?([\w\-_/]+\.[a-zA-Z0-9]+)['\"]?", user_prompt)
    suggested_filename = fname_match.group(1) if fname_match else None

    planner_prompt = f"""Você é um Agente Programador Sênior com acesso direto ao MCP Filesystem no diretório seguro '/app/workspace'.
O usuário solicitou a criação de um arquivo ou script.

PEDIDO DO USUÁRIO:
{user_prompt}

CONTEXTO ADICIONAL:
{context_prompt}

SUA TAREFA:
1. Determine o nome de arquivo mais adequado (ex: '{suggested_filename or "script.py"}').
2. Escreva o código ou conteúdo COMPLETO, robusto, funcional e sem omissões ('// TODO' é proibido).
3. Responda ESTRITAMENTE em formato JSON com o seguinte schema:
{{
  "filename": "{suggested_filename or "script.py"}",
  "content": "conteudo completo do arquivo aqui",
  "summary": "Breve explicação do que o script faz e como executar"
}}
Responda APENAS o JSON válido, sem crases adicionais."""

    try:
        try:
            generator_llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model="Combo_Free_Coding",
                temperature=0.1,
                max_tokens=4096,
                timeout=30.0
            )
            plan_res = await generator_llm.ainvoke(planner_prompt)
        except Exception as llm_err:
            print(f"[Aviso MCP LLM Fallback para ComboFree_Smart]: {llm_err}")
            fallback_llm = ChatOpenAI(
                base_url=LITELLM_BASE_URL,
                api_key=LITELLM_API_KEY,
                model="ComboFree_Smart",
                temperature=0.1,
                max_tokens=4096,
                timeout=30.0
            )
            plan_res = await fallback_llm.ainvoke(planner_prompt)

        raw_json = plan_res.content if hasattr(plan_res, "content") else str(plan_res)
        raw_json = raw_json.strip()

        # Remove eventuais blocos de ```json ou ```
        if "```json" in raw_json:
            raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
        elif raw_json.startswith("```"):
            raw_json = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", raw_json)
            raw_json = re.sub(r"\s*```$", "", raw_json).strip()

        try:
            data = json.loads(raw_json)
            filename = data.get("filename", suggested_filename or "script.py")
            content = data.get("content", "")
            summary = data.get("summary", "Arquivo gerado e salvo com sucesso no workspace.")
        except Exception:
            # Fallback resiliente: se o modelo devolveu código puro
            filename = suggested_filename or "script.py"
            content = raw_json
            summary = f"Arquivo {filename} gerado com sucesso conforme solicitação."

        # Escreve o arquivo fisicamente na sandbox
        w_res = core_write_file(filename, content)
        tools_called.append({
            "tool": "mcp_write_file",
            "arguments": {"path": filename, "bytes": len(content)},
            "status": "success",
            "duration_seconds": round(time.time() - t0, 3)
        })

        ext = Path(filename).suffix.lstrip(".").lower()
        lang = "python" if ext == "py" else ("javascript" if ext == "js" else ext)

        final_response = f"""### 💾 Arquivo Criado e Salvo no Workspace!

- **Arquivo:** `{w_res.get('path')}` (`/app/workspace/{w_res.get('path')}`)
- **Tamanho:** {w_res.get('bytes_written', 0):,} caracteres
- **Status:** Gravado com sucesso no Sandbox Seguro

#### 📝 **Código Implementado:**
```{lang}
{content}
```

#### 🚀 **Descrição & Uso:**
{summary}
"""
        return final_response, tools_called

    except Exception as err:
        tools_called.append({
            "tool": "mcp_agentic_write",
            "arguments": {"prompt": user_prompt[:80]},
            "status": f"error: {str(err)}",
            "duration_seconds": round(time.time() - t0, 3)
        })
        return f"⚠️ **Ocorreu um erro ao processar a criação do arquivo no workspace:** {str(err)}", tools_called

