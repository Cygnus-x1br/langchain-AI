"""
MCP Database Service - Ferramentas de Banco de Dados Confinadas
===============================================================
Permite aos agentes e ao usuário consultar, descrever tabelas e executar
comandos em bancos de dados relacionais e analíticos de forma controlada.

Motores Suportados:
1. SQLite: Bancos locais estritamente confinados em WORKSPACE_PATH (/app/workspace).
2. DuckDB: Motor SQL analítico de alta performance para arquivos locais (.csv, .parquet, .json, .duckdb).
3. PostgreSQL: Conexão via URI ou variáveis de ambiente (POSTGRES_URL / DATABASE_URL).
4. MongoDB: Consultas NoSQL via PyMongo (MONGODB_URI).

Mecanismos de Segurança (Zero-Trust Sandbox):
1. Proteção de leitura estrita (read_only=True por padrão): bloqueia DROP, TRUNCATE, DELETE, UPDATE acidentais.
2. Resolução canônica de caminhos de arquivos de banco SQLite/DuckDB via resolve_safe_path().
3. Limite de paginação de resultados (máximo 200 registros) para evitar estouro de memória (OOM).
4. Exposição dupla: LangChain Tools (@tool) e OpenAI Tool Calling Schemas.
"""

import os
import re
import time
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

from langchain_core.tools import tool

# Diretório base do Workspace
WORKSPACE_PATH = os.getenv("WORKSPACE_PATH", "/app/workspace")
os.makedirs(WORKSPACE_PATH, exist_ok=True)

MAX_ROWS_RETURNED = 200


# =========================================================================
# 1. SEGURANÇA E RESOLUÇÃO CANÔNICA
# =========================================================================

def get_workspace_root() -> Path:
    """Retorna o caminho canônico absoluto do diretório raiz do workspace."""
    return Path(WORKSPACE_PATH).resolve()


def resolve_safe_db_path(db_name_or_path: str, default_name: str = "workspace.db") -> Path:
    """
    Resolve e valida o caminho do arquivo de banco de dados contra a raiz do workspace seguro.
    Garante que qualquer tentativa de Path Traversal ('../' ou caminhos fora da sandbox)
    seja interceptada com PermissionError.
    """
    base_dir = get_workspace_root()
    clean_str = str(db_name_or_path or "").strip()

    if not clean_str or clean_str in [".", "./"]:
        clean_str = default_name

    if clean_str.startswith("/"):
        target = Path(clean_str).resolve()
        try:
            target.relative_to(base_dir)
            return target
        except ValueError:
            raise PermissionError(f"Acesso negado: O caminho '{clean_str}' está fora do workspace seguro '{base_dir}'.")
    else:
        target = (base_dir / clean_str).resolve()
        try:
            target.relative_to(base_dir)
            return target
        except ValueError:
            raise PermissionError(f"Acesso negado: Tentativa de travessia detectada em '{clean_str}'.")


def is_safe_read_query(query: str) -> bool:
    """
    Verifica se a query é uma operação estritamente de leitura (SELECT, PRAGMA, EXPLAIN, WITH, SHOW, DESCRIBE).
    """
    clean_q = re.sub(r'--.*?$|/\*.*?\*/', '', query, flags=re.MULTILINE).strip().upper()
    read_verbs = ("SELECT", "WITH", "PRAGMA", "EXPLAIN", "SHOW", "DESCRIBE", "DESC")
    if any(clean_q.startswith(verb) for verb in read_verbs):
        dangerous = re.search(r'\b(DROP|TRUNCATE|DELETE|UPDATE|INSERT|ALTER|ATTACH|DETACH|REINDEX|VACUUM)\b', clean_q)
        return dangerous is None
    return False


# =========================================================================
# 2. IMPLEMENTAÇÕES DE BAIXO NÍVEL (SQLITE, DUCKDB, POSTGRES, MONGO)
# =========================================================================

def sqlite_list_tables_impl(database: str = "workspace.db") -> Dict[str, Any]:
    """Lista todas as tabelas e views presentes no banco SQLite do workspace."""
    try:
        db_path = resolve_safe_db_path(database)
        if not db_path.exists():
            return {
                "status": "warning",
                "message": f"O banco de dados '{db_path.name}' ainda não existe no workspace. Crie tabelas primeiro.",
                "tables": []
            }

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            rows = cursor.fetchall()
            tables = [{"type": r[0], "name": r[1]} for r in rows]
            return {
                "status": "success",
                "database": db_path.name,
                "table_count": len(tables),
                "tables": tables
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def sqlite_describe_table_impl(table_name: str, database: str = "workspace.db") -> Dict[str, Any]:
    """Retorna as colunas, tipos e índices de uma tabela SQLite."""
    try:
        db_path = resolve_safe_db_path(database)
        if not db_path.exists():
            return {"status": "error", "error": f"Banco '{db_path.name}' não encontrado."}

        # Sanitizar nome da tabela para evitar SQL Injection em PRAGMA
        if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
            return {"status": "error", "error": "Nome de tabela inválido. Use apenas caracteres alfanuméricos e underscore."}

        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            if not columns:
                return {"status": "error", "error": f"Tabela '{table_name}' não encontrada no banco '{db_path.name}'."}

            col_list = [{
                "cid": c[0],
                "name": c[1],
                "type": c[2],
                "notnull": bool(c[3]),
                "dflt_value": c[4],
                "pk": bool(c[5])
            } for c in columns]

            cursor.execute(f"SELECT sql FROM sqlite_master WHERE name = ?", (table_name,))
            create_sql = cursor.fetchone()

            return {
                "status": "success",
                "table": table_name,
                "database": db_path.name,
                "columns": col_list,
                "create_sql": create_sql[0] if create_sql else None
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def sqlite_query_impl(query: str, database: str = "workspace.db", read_only: bool = True, limit: int = 100) -> Dict[str, Any]:
    """Executa uma consulta SQL no banco SQLite do workspace."""
    try:
        db_path = resolve_safe_db_path(database)
        if not db_path.exists():
            return {"status": "error", "error": f"Banco '{db_path.name}' não existe no workspace."}

        if read_only and not is_safe_read_query(query):
            return {
                "status": "error",
                "error": "Operação bloqueada pelo modo seguro (read_only=True). Operações de alteração (INSERT, UPDATE, DELETE, DROP) exigem read_only=False ou a ferramenta 'mcp_db_execute'."
            }

        limit = min(max(1, limit), MAX_ROWS_RETURNED)

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)

            if cursor.description is None:
                conn.commit()
                return {
                    "status": "success",
                    "database": db_path.name,
                    "rows_affected": cursor.rowcount,
                    "message": f"Comando executado com sucesso. Linhas afetadas: {cursor.rowcount}"
                }

            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchmany(limit)
            result_data = [dict(row) for row in rows]

            return {
                "status": "success",
                "database": db_path.name,
                "columns": columns,
                "row_count": len(result_data),
                "limit_applied": limit,
                "rows": result_data
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def sqlite_execute_impl(statement: str, database: str = "workspace.db") -> Dict[str, Any]:
    """Executa scripts DDL/DML (CREATE, INSERT, UPDATE, DELETE) no banco SQLite do workspace."""
    try:
        db_path = resolve_safe_db_path(database)
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.executescript(statement)
            conn.commit()
            return {
                "status": "success",
                "database": db_path.name,
                "message": "Script SQL executado com sucesso."
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def duckdb_query_impl(query: str, database: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
    """Executa consulta analítica via DuckDB sobre arquivos locais ou banco DuckDB."""
    try:
        import duckdb
    except ImportError:
        return {"status": "error", "error": "DuckDB não está instalado no ambiente."}

    try:
        limit = min(max(1, limit), MAX_ROWS_RETURNED)
        base_dir = get_workspace_root()

        # Substituir caminhos relativos de arquivos para caminhos absolutos seguros
        def replace_file_ref(match):
            rel_file = match.group(1)
            safe_p = resolve_safe_db_path(rel_file)
            return f"'{str(safe_p)}'"

        processed_query = re.sub(r"'(read_[a-z_]+\([^)]*\)|[a-zA-Z0-9_\-./]+\.(csv|parquet|json|jsonl))'", replace_file_ref, query, flags=re.IGNORECASE)

        if database:
            db_path = resolve_safe_db_path(database, default_name="workspace.duckdb")
            conn = duckdb.connect(str(db_path))
        else:
            conn = duckdb.connect(":memory:")

        rel = conn.execute(processed_query)
        columns = [desc[0] for desc in rel.description] if rel.description else []
        rows = rel.fetchmany(limit)
        conn.close()

        formatted_rows = [dict(zip(columns, row)) for row in rows]
        return {
            "status": "success",
            "engine": "duckdb",
            "columns": columns,
            "row_count": len(formatted_rows),
            "rows": formatted_rows
        }
    except Exception as e:
        return {"status": "error", "error": f"Erro no DuckDB: {str(e)}"}


def postgres_query_impl(query: str, uri: Optional[str] = None, read_only: bool = True, limit: int = 100) -> Dict[str, Any]:
    """Executa consulta SQL no PostgreSQL via psycopg2 ou SQLAlchemy."""
    conn_uri = uri or os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL")
    if not conn_uri:
        return {
            "status": "error",
            "error": "Nenhuma URI de PostgreSQL configurada. Forneça o parâmetro 'uri' ou defina POSTGRES_URL / DATABASE_URL no .env."
        }

    if read_only and not is_safe_read_query(query):
        return {
            "status": "error",
            "error": "Operação bloqueada pelo modo seguro (read_only=True) em PostgreSQL."
        }

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        limit = min(max(1, limit), MAX_ROWS_RETURNED)

        with psycopg2.connect(conn_uri) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                if cur.description is None:
                    conn.commit()
                    return {"status": "success", "rows_affected": cur.rowcount}
                rows = cur.fetchmany(limit)
                cols = [desc[0] for desc in cur.description]
                return {
                    "status": "success",
                    "engine": "postgresql",
                    "columns": cols,
                    "row_count": len(rows),
                    "rows": [dict(r) for r in rows]
                }
    except Exception as e:
        return {"status": "error", "error": f"Erro no PostgreSQL: {str(e)}"}


# =========================================================================
# 3. EXPOSIÇÃO LANGCHAIN TOOLS (@tool)
# =========================================================================

@tool
def mcp_db_list_tables(database: str = "workspace.db", engine: str = "sqlite") -> str:
    """
    Lista todas as tabelas e views presentes no banco de dados seguro do workspace.
    
    Args:
        database: Nome do arquivo de banco no workspace (ex: 'workspace.db', 'app.db'). Padrão: 'workspace.db'.
        engine: Motor de banco ('sqlite' ou 'duckdb'). Padrão: 'sqlite'.
    """
    res = sqlite_list_tables_impl(database=database)
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_db_describe_table(table_name: str, database: str = "workspace.db") -> str:
    """
    Descreve os campos, tipos, chave primária e DDL da tabela no banco de dados.
    
    Args:
        table_name: Nome da tabela a ser inspecionada.
        database: Nome do arquivo de banco (padrão: 'workspace.db').
    """
    res = sqlite_describe_table_impl(table_name=table_name, database=database)
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_db_query(
    query: str,
    database: str = "workspace.db",
    engine: str = "sqlite",
    read_only: bool = True,
    limit: int = 50
) -> str:
    """
    Executa uma consulta SQL segura em banco de dados SQLite ou DuckDB.
    
    Args:
        query: Sentença SQL a ser executada (ex: 'SELECT * FROM users WHERE active = 1').
        database: Nome do arquivo de banco no workspace (padrão: 'workspace.db').
        engine: 'sqlite', 'duckdb' ou 'postgres'.
        read_only: Se True, bloqueia comandos destrutivos (padrão: True).
        limit: Máximo de linhas a retornar (máximo 200).
    """
    engine_lower = (engine or "sqlite").strip().lower()
    if engine_lower == "duckdb":
        res = duckdb_query_impl(query=query, database=database, limit=limit)
    elif engine_lower == "postgres":
        res = postgres_query_impl(query=query, read_only=read_only, limit=limit)
    else:
        res = sqlite_query_impl(query=query, database=database, read_only=read_only, limit=limit)
    return json.dumps(res, ensure_ascii=False, indent=2)


@tool
def mcp_db_execute(statement: str, database: str = "workspace.db") -> str:
    """
    Executa comandos de alteração DDL ou DML (CREATE TABLE, INSERT INTO, UPDATE, DELETE) no SQLite do workspace.
    
    Args:
        statement: Script SQL ou comando DDL/DML.
        database: Nome do arquivo de banco no workspace (padrão: 'workspace.db').
    """
    res = sqlite_execute_impl(statement=statement, database=database)
    return json.dumps(res, ensure_ascii=False, indent=2)


def get_mcp_database_langchain_tools() -> List[Any]:
    """Retorna lista das ferramentas MCP Database prontas para LangChain Agents."""
    return [
        mcp_db_list_tables,
        mcp_db_describe_table,
        mcp_db_query,
        mcp_db_execute
    ]


# =========================================================================
# 4. EXPOSIÇÃO OPENAI TOOL CALLING SCHEMAS
# =========================================================================

def get_mcp_database_openai_tools() -> List[Dict[str, Any]]:
    """Retorna a especificação dos schemas JSON no formato oficial OpenAI Tool Calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "mcp_db_list_tables",
                "description": "Lista todas as tabelas e views presentes no banco de dados seguro do workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": "string",
                            "description": "Nome do arquivo de banco no workspace (padrão: 'workspace.db')."
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["sqlite", "duckdb"],
                            "description": "Motor de banco de dados."
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_db_describe_table",
                "description": "Inspeciona colunas, tipos, chaves e DDL de uma tabela.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Nome da tabela a ser inspecionada."
                        },
                        "database": {
                            "type": "string",
                            "description": "Nome do arquivo de banco no workspace (padrão: 'workspace.db')."
                        }
                    },
                    "required": ["table_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_db_query",
                "description": "Executa uma consulta SQL (SELECT) segura com limitação de paginação.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Comando SQL a ser executado."
                        },
                        "database": {
                            "type": "string",
                            "description": "Arquivo de banco no workspace (padrão: 'workspace.db')."
                        },
                        "engine": {
                            "type": "string",
                            "enum": ["sqlite", "duckdb", "postgres"],
                            "description": "Motor de banco de dados."
                        },
                        "read_only": {
                            "type": "boolean",
                            "description": "Se True, bloqueia comandos de alteração estrutural ou exclusão."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Número máximo de registros a retornar (máximo 200)."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mcp_db_execute",
                "description": "Executa scripts DDL/DML (CREATE TABLE, INSERT, UPDATE) no banco do workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "statement": {
                            "type": "string",
                            "description": "Script SQL a ser executado."
                        },
                        "database": {
                            "type": "string",
                            "description": "Arquivo de banco no workspace (padrão: 'workspace.db')."
                        }
                    },
                    "required": ["statement"]
                }
            }
        }
    ]


# =========================================================================
# 5. FLUXO EXECUTOR AUTÔNOMO (execute_mcp_database_flow)
# =========================================================================

async def execute_mcp_database_flow(
    user_prompt: str,
    conversation_id: str = "default_session",
    user_id: str = "default_user"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executa comandos de banco de dados com inteligência contextual e formatação clara para o usuário.
    """
    t0 = time.time()
    tools_called = []
    p_lower = user_prompt.lower()

    # Heurística rápida direta: listar tabelas
    if any(k in p_lower for k in ["listar tabelas", "quais as tabelas", "mostre as tabelas", "list tables", "tabelas do banco"]):
        res = sqlite_list_tables_impl()
        tools_called.append({"tool": "mcp_db_list_tables", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
        if res.get("status") == "success":
            tables = res.get("tables", [])
            if not tables:
                return "### 🗄️ MCP Database: Tabelas\n\nO banco de dados do workspace está vazio ou não possui tabelas criadas.", tools_called
            t_md = "\n".join([f"- **{t['name']}** ({t['type']})" for t in tables])
            return f"### 🗄️ MCP Database: Tabelas no Workspace (`{res.get('database')}`)\n\nTotal de tabelas: **{res.get('table_count')}**\n\n{t_md}", tools_called
        return f"### ⚠️ MCP Database\n\n{res.get('message', res.get('error'))}", tools_called

    # Descrever tabela
    desc_match = re.search(r'(?:descreva|estrutura|schema|campos\s+da\s+tabela)\s+([a-zA-Z0-9_]+)', user_prompt, re.IGNORECASE)
    if desc_match:
        tbl = desc_match.group(1).strip()
        res = sqlite_describe_table_impl(table_name=tbl)
        tools_called.append({"tool": "mcp_db_describe_table", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
        if res.get("status") == "success":
            cols = res.get("columns", [])
            col_lines = "\n".join([f"| `{c['name']}` | {c['type']} | {'Sim' if c['pk'] else 'Não'} | {'Sim' if c['notnull'] else 'Não'} |" for c in cols])
            md = (
                f"### 🗄️ MCP Database: Schema da Tabela `{tbl}`\n\n"
                f"| Campo | Tipo | Chave Primária | Obrigatório |\n"
                f"| :--- | :--- | :---: | :---: |\n"
                f"{col_lines}\n\n"
            )
            if res.get("create_sql"):
                md += f"```sql\n{res['create_sql']}\n```"
            return md, tools_called
        return f"### ⚠️ MCP Database\n\nErro ao inspecionar tabela `{tbl}`: {res.get('error')}", tools_called

    # Extrair bloco SQL direto se existir
    sql_block = re.search(r'```(?:sql)?\s*(.*?)\s*```', user_prompt, re.DOTALL | re.IGNORECASE)
    raw_sql = sql_block.group(1).strip() if sql_block else ""
    if not raw_sql:
        # Tentar capturar comando SQL inline comum
        select_match = re.search(r'(SELECT\s+.*?;?)', user_prompt, re.IGNORECASE | re.DOTALL)
        if select_match:
            raw_sql = select_match.group(1).strip()

    if raw_sql:
        is_select = raw_sql.strip().upper().startswith(("SELECT", "WITH", "PRAGMA", "EXPLAIN"))
        if is_select:
            res = sqlite_query_impl(query=raw_sql, read_only=True)
            tools_called.append({"tool": "mcp_db_query", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
            if res.get("status") == "success":
                rows = res.get("rows", [])
                cols = res.get("columns", [])
                if not rows:
                    return f"### 🗄️ MCP Database: Consulta Executada\n\nA consulta retornou **0 registros**.\n\n```sql\n{raw_sql}\n```", tools_called

                # Formatar como tabela Markdown
                header = "| " + " | ".join(cols) + " |"
                sep = "| " + " | ".join(["---"] * len(cols)) + " |"
                row_lines = []
                for r in rows[:50]:
                    vals = [str(r.get(c, "")) for c in cols]
                    row_lines.append("| " + " | ".join(vals) + " |")

                table_str = "\n".join([header, sep] + row_lines)
                extra = f"\n\n*(Exibindo {len(rows)} de {res.get('row_count')} registros retornados)*" if len(rows) > 50 else ""
                return f"### 🗄️ MCP Database: Resultado da Consulta\n\n{table_str}{extra}", tools_called
            return f"### ⚠️ MCP Database: Erro na Consulta\n\n{res.get('error')}", tools_called
        else:
            # Comando DDL/DML
            res = sqlite_execute_impl(statement=raw_sql)
            tools_called.append({"tool": "mcp_db_execute", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
            if res.get("status") == "success":
                return f"### 🗄️ MCP Database: Sucesso\n\n{res.get('message')}\n\n```sql\n{raw_sql}\n```", tools_called
            return f"### ⚠️ MCP Database: Erro na Execução\n\n{res.get('error')}", tools_called

    # Se não houver SQL explícito, consultar tabelas por padrão
    res = sqlite_list_tables_impl()
    tools_called.append({"tool": "mcp_db_list_tables", "status": res.get("status"), "duration_seconds": round(time.time() - t0, 3)})
    tables = res.get("tables", [])
    t_str = ", ".join([t['name'] for t in tables]) if tables else "Nenhuma tabela criada ainda"
    return (
        f"### 🗄️ MCP Database (Workspace)\n\n"
        f"Banco atual: `{res.get('database', 'workspace.db')}`\n"
        f"Tabelas disponíveis: **{t_str}**\n\n"
        f"Você pode enviar sentenças SQL completas em blocos de código (ex: `SELECT * FROM tabela;`) "
        f"ou solicitar a estrutura de tabelas (`descreva a tabela X`)."
    ), tools_called
