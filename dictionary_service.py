import os
import re
import json
import time
from typing import Optional, Tuple, Dict, Any, Set

UNIVERSAL_DICT_PATH = os.getenv("UNIVERSAL_DICT_PATH", "/app/data/universal_dictionary.json")
if not os.path.exists(UNIVERSAL_DICT_PATH):
    UNIVERSAL_DICT_PATH = os.path.join(os.path.dirname(__file__), "data", "universal_dictionary.json")

_cached_dict: Dict[str, Any] = {}
_cached_dict_mtime: float = 0.0


def load_universal_dictionary() -> Dict[str, Any]:
    """Carrega o Dicionário Universal de Termos em memória com auto-recarga quando o arquivo for modificado."""
    global _cached_dict, _cached_dict_mtime
    try:
        if os.path.exists(UNIVERSAL_DICT_PATH):
            mtime = os.path.getmtime(UNIVERSAL_DICT_PATH)
            if mtime != _cached_dict_mtime or not _cached_dict:
                with open(UNIVERSAL_DICT_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _cached_dict = data.get("categories", {})
                _cached_dict_mtime = mtime
                print(f"[Dicionário Universal] Carregado com sucesso de '{UNIVERSAL_DICT_PATH}' ({len(_cached_dict)} categorias).")
    except Exception as e:
        print(f"[Aviso Dicionário Universal] Falha ao carregar {UNIVERSAL_DICT_PATH}: {e}")
    return _cached_dict


def match_universal_dictionary(
    prompt: str,
    has_image_now: bool = False,
    has_image_in_history: bool = False
) -> Optional[Tuple[str, str]]:
    """
    Classifica a intenção do usuário a partir do Dicionário Universal de Termos e Expressões.
    Retorna uma tupla (categoria, routing_method) se houver correspondência, ou None caso contrário.
    """
    if not prompt:
        return None

    p_lower = prompt.lower().strip()
    # Extrai tokens de palavras para verificação ultrarrápida via sets
    tokens: Set[str] = set(re.findall(r"\b\w+\b", p_lower))
    cat_dict = load_universal_dictionary()
    if not cat_dict:
        return None

    # -------------------------------------------------------------
    # 1. Regra de Edição de Imagem (com anexo presente ou recente)
    # -------------------------------------------------------------
    if has_image_now or has_image_in_history:
        edit_cfg = cat_dict.get("IMAGE_EDIT", {})
        # Checagem por frases completas ("coloca um fundo azul", "trocar o fundo", etc.)
        if any(phrase in p_lower for phrase in edit_cfg.get("phrases", [])):
            return "IMAGE_EDIT", "universal_dict_image_edit_phrase"
        
        # Checagem por verbos de edição
        edit_verbs = set(edit_cfg.get("verbs", []))
        if tokens & edit_verbs:
            return "IMAGE_EDIT", "universal_dict_image_edit_verb"

    # -------------------------------------------------------------
    # 2. Vídeos Curtos / Reels / TikTok (precede post social e imagem)
    # -------------------------------------------------------------
    vid_cfg = cat_dict.get("SHORT_VIDEO", {})
    if any(phrase in p_lower for phrase in vid_cfg.get("phrases", [])):
        return "SHORT_VIDEO", "universal_dict_video_phrase"
    
    vid_formats = set(vid_cfg.get("formats", []))
    if any(fmt in p_lower for fmt in vid_formats):
        action_verbs = {"crie", "criar", "gere", "gerar", "faça", "fazer", "monte", "produza", "produzir", "roteiro", "sobre", "para"}
        if tokens & action_verbs:
            return "SHORT_VIDEO", "universal_dict_video_format"

    # -------------------------------------------------------------
    # 3. Post para Redes Sociais e Leads (precede imagem genérica)
    # -------------------------------------------------------------
    soc_cfg = cat_dict.get("SOCIAL_MEDIA_POST", {})
    if any(phrase in p_lower for phrase in soc_cfg.get("phrases", [])):
        return "SOCIAL_MEDIA_POST", "universal_dict_social_phrase"
    
    networks = set(soc_cfg.get("networks", []))
    content_types = set(soc_cfg.get("content_types", []))
    creation_verbs = {"crie", "criar", "gere", "gerar", "faça", "fazer", "elabore", "elaborar", "monte", "montar", "escreva", "escrever", "produza", "produzir"}
    
    # Tolerância a variações e erros de digitação (ex: postgem, postagens, posts, publicacao)
    has_social_token = bool(tokens & content_types) or any(t.startswith("post") or t.startswith("publica") for t in tokens)
    has_network_mention = bool(tokens & networks) or any(net in p_lower for net in networks)
    
    # 1. Menção explícita a qualquer rede social (Instagram, LinkedIn, etc.) com pedido de criação ou imagem
    if has_network_mention and (has_social_token or (tokens & creation_verbs) or has_image_now or has_image_in_history or "imagem" in tokens or "foto" in tokens):
        return "SOCIAL_MEDIA_POST", "universal_dict_social_network_detected"

    # 2. Post com imagem anexa ou referenciada
    if (has_image_now or has_image_in_history or "imagem" in tokens or "foto" in tokens) and has_social_token:
        return "SOCIAL_MEDIA_POST", "universal_dict_social_with_image"

    # 3. Criação genérica de postagem/post ("crie uma postagem...", "monte um post...")
    if (tokens & creation_verbs) and has_social_token:
        return "SOCIAL_MEDIA_POST", "universal_dict_social_verb_content"

    # -------------------------------------------------------------
    # 4. Geração de Imagem Nova
    # -------------------------------------------------------------
    # Proteção estrita: se citou rede social ou post, NUNCA é geração de imagem pura
    if has_network_mention:
        return "SOCIAL_MEDIA_POST", "universal_dict_social_network_fallback"
    if has_social_token:
        return "SOCIAL_MEDIA_POST", "universal_dict_social_token_fallback"

    gen_cfg = cat_dict.get("IMAGE_GENERATION", {})
    if any(phrase in p_lower for phrase in gen_cfg.get("phrases", [])):
        return "IMAGE_GENERATION", "universal_dict_image_gen_phrase"
    
    gen_verbs = set(gen_cfg.get("verbs", []))
    gen_targets = set(gen_cfg.get("targets", []))
    if (tokens & gen_verbs) and (tokens & gen_targets):
        return "IMAGE_GENERATION", "universal_dict_image_gen_verb_target"

    # -------------------------------------------------------------
    # 5. Exportação de Planilhas e Gráficos
    # -------------------------------------------------------------
    chart_cfg = cat_dict.get("DATA_EXPORT_CHART", {})
    if any(phrase in p_lower for phrase in chart_cfg.get("phrases", [])):
        return "DATA_EXPORT_CHART", "universal_dict_chart_phrase"
    
    chart_targets = set(chart_cfg.get("targets", []))
    chart_verbs = set(chart_cfg.get("verbs", []))
    if (tokens & chart_targets) and (tokens & chart_verbs):
        return "DATA_EXPORT_CHART", "universal_dict_chart_verb_target"

    # -------------------------------------------------------------
    # 6. Clima e Meteorologia
    # -------------------------------------------------------------
    wea_cfg = cat_dict.get("WEATHER_INFO", {})
    if any(phrase in p_lower for phrase in wea_cfg.get("phrases", [])):
        return "WEATHER_INFO", "universal_dict_weather_phrase"

    # -------------------------------------------------------------
    # 7. Programação, Infraestrutura, Servidores e Código
    # -------------------------------------------------------------
    cod_cfg = cat_dict.get("CODING", {})
    if any(phrase in p_lower for phrase in cod_cfg.get("phrases", [])):
        return "CODING", "universal_dict_coding_phrase"
    
    infra_terms = set(cod_cfg.get("infrastructure_and_systems", []))
    if infra_terms:
        # Match direto por tokens ou substrings completas (ex: 'servidor de arquivos', 'set-smbclientconfiguration')
        if tokens & infra_terms or any(term in p_lower for term in infra_terms):
            return "CODING", "universal_dict_coding_infra_term"

    # -------------------------------------------------------------
    # 8. Raciocínio Pesado / Matemática Avançada
    # -------------------------------------------------------------
    rea_cfg = cat_dict.get("HEAVY_REASONING", {})
    if any(phrase in p_lower for phrase in rea_cfg.get("phrases", [])):
        return "HEAVY_REASONING", "universal_dict_reasoning_phrase"
    if any(topic in p_lower for topic in rea_cfg.get("topics", [])):
        return "HEAVY_REASONING", "universal_dict_reasoning_topic"

    # -------------------------------------------------------------
    # 9. Busca Web em Tempo Real / Notícias
    # -------------------------------------------------------------
    sea_cfg = cat_dict.get("REALTIME_SEARCH", {})
    if any(phrase in p_lower for phrase in sea_cfg.get("phrases", [])):
        return "REALTIME_SEARCH", "universal_dict_search_phrase"

    # -------------------------------------------------------------
    # 11. MCP Filesystem / Sandbox de Arquivos
    # -------------------------------------------------------------
    fs_cfg = cat_dict.get("MCP_FILESYSTEM", {})
    if any(phrase in p_lower for phrase in fs_cfg.get("phrases", [])):
        return "MCP_FILESYSTEM", "universal_dict_filesystem_phrase"

    fs_keywords = set(fs_cfg.get("keywords", []))
    fs_verbs = set(fs_cfg.get("verbs", []))
    if (tokens & fs_keywords) and (tokens & fs_verbs):
        return "MCP_FILESYSTEM", "universal_dict_filesystem_keyword_verb"
    if "workspace" in tokens:
        return "MCP_FILESYSTEM", "universal_dict_filesystem_workspace_token"

    # -------------------------------------------------------------
    # 12. MCP Fetch / Web Scrape Seguro
    # -------------------------------------------------------------
    fetch_cfg = cat_dict.get("MCP_FETCH", {})
    if any(phrase in p_lower for phrase in fetch_cfg.get("phrases", [])):
        return "MCP_FETCH", "universal_dict_fetch_phrase"

    fetch_keywords = set(fetch_cfg.get("keywords", []))
    fetch_verbs = set(fetch_cfg.get("verbs", []))
    if (tokens & fetch_keywords) and (tokens & fetch_verbs):
        return "MCP_FETCH", "universal_dict_fetch_keyword_verb"
    if "mcp_fetch" in tokens or "fetch" in tokens:
        return "MCP_FETCH", "universal_dict_fetch_token"

    # -------------------------------------------------------------
    # 13. MCP Git / Controle de Versão Seguro
    # -------------------------------------------------------------
    git_cfg = cat_dict.get("MCP_GIT", {})
    if any(phrase in p_lower for phrase in git_cfg.get("phrases", [])):
        return "MCP_GIT", "universal_dict_git_phrase"

    git_keywords = set(git_cfg.get("keywords", []))
    git_verbs = set(git_cfg.get("verbs", []))
    if (tokens & git_keywords) and (tokens & git_verbs):
        return "MCP_GIT", "universal_dict_git_keyword_verb"
    if any(k in tokens for k in ["mcp_git", "git_status", "git_commit", "git_diff", "git_log"]):
        return "MCP_GIT", "universal_dict_git_token"

    # -------------------------------------------------------------
    # 14. MCP Brave Search / Busca Web Independente
    # -------------------------------------------------------------
    brave_cfg = cat_dict.get("MCP_BRAVE_SEARCH", {})
    if any(phrase in p_lower for phrase in brave_cfg.get("phrases", [])):
        return "MCP_BRAVE_SEARCH", "universal_dict_brave_phrase"

    brave_keywords = set(brave_cfg.get("keywords", []))
    if (tokens & brave_keywords):
        return "MCP_BRAVE_SEARCH", "universal_dict_brave_keyword"

    # -------------------------------------------------------------
    # 15. MCP Database / Consultas SQL e Bancos
    # -------------------------------------------------------------
    db_cfg = cat_dict.get("MCP_DATABASE", {})
    if any(phrase in p_lower for phrase in db_cfg.get("phrases", [])):
        return "MCP_DATABASE", "universal_dict_db_phrase"

    db_keywords = set(db_cfg.get("keywords", []))
    db_verbs = set(db_cfg.get("verbs", []))
    if (tokens & db_keywords) and (tokens & db_verbs):
        return "MCP_DATABASE", "universal_dict_db_keyword_verb"
    if any(k in tokens for k in ["mcp_database", "mcp_db", "duckdb", "sqlite3"]):
        return "MCP_DATABASE", "universal_dict_db_token"

    # -------------------------------------------------------------
    # 16. MCP Code Runner / Execução Segura de Scripts
    # -------------------------------------------------------------
    cr_cfg = cat_dict.get("MCP_CODE_RUNNER", {})
    if any(phrase in p_lower for phrase in cr_cfg.get("phrases", [])):
        return "MCP_CODE_RUNNER", "universal_dict_code_runner_phrase"

    cr_keywords = set(cr_cfg.get("keywords", []))
    cr_verbs = set(cr_cfg.get("verbs", []))
    if (tokens & cr_keywords) and (tokens & cr_verbs):
        return "MCP_CODE_RUNNER", "universal_dict_code_runner_keyword_verb"
    if any(k in tokens for k in ["mcp_code_runner", "mcp_runner", "code_runner"]):
        return "MCP_CODE_RUNNER", "universal_dict_code_runner_token"

    return None


def get_full_dictionary() -> Dict[str, Any]:
    """Retorna o dicionário completo categorizado."""
    return load_universal_dictionary()


def save_universal_dictionary(categories_dict: Dict[str, Any]) -> bool:
    """Persiste o dicionário atualizado em disco e atualiza o cache em memória."""
    global _cached_dict, _cached_dict_mtime
    try:
        os.makedirs(os.path.dirname(UNIVERSAL_DICT_PATH), exist_ok=True)
        payload = {
            "version": "1.0",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Dicionário Universal Padrão de Termos e Intenções para Roteamento Semântico",
            "categories": categories_dict
        }
        with open(UNIVERSAL_DICT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        _cached_dict = categories_dict
        _cached_dict_mtime = os.path.getmtime(UNIVERSAL_DICT_PATH)
        print(f"[Dicionário Universal] Salvo com sucesso em {UNIVERSAL_DICT_PATH}")
        return True
    except Exception as e:
        print(f"[Erro Salvar Dicionário]: {e}")
        return False


def add_term_to_dictionary(category: str, field_type: str, term: str) -> bool:
    """Adiciona um novo termo/frase a uma categoria e lista específica."""
    cat_dict = load_universal_dictionary()
    if category not in cat_dict:
        cat_dict[category] = {}
    
    clean_term = term.strip().lower()
    if not clean_term:
        return False

    if field_type not in cat_dict[category]:
        cat_dict[category][field_type] = []

    if clean_term not in cat_dict[category][field_type]:
        cat_dict[category][field_type].append(clean_term)
        return save_universal_dictionary(cat_dict)
    return True


def remove_term_from_dictionary(category: str, field_type: str, term: str) -> bool:
    """Remove um termo/frase de uma categoria e lista específica."""
    cat_dict = load_universal_dictionary()
    clean_term = term.strip().lower()

    if category in cat_dict and field_type in cat_dict[category]:
        if clean_term in cat_dict[category][field_type]:
            cat_dict[category][field_type].remove(clean_term)
            return save_universal_dictionary(cat_dict)
    return False
