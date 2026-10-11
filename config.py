import os
import re
import json
import time
import datetime
from zoneinfo import ZoneInfo
import uuid
import base64
import httpx

# Configuração de Fuso Horário Local (São Paulo / Brasil - UTC-3)
LOCAL_TZ = ZoneInfo("America/Sao_Paulo")

def get_now_local():
    """Retorna o timestamp atual no fuso horário oficial de São Paulo (UTC-3)."""
    return datetime.datetime.now(LOCAL_TZ)

# Endpoints e Chaves
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "sk-anything")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://10.100.10.31:8080")
LIBRECHAT_MONGO_URI = os.getenv("LIBRECHAT_MONGO_URI", "mongodb://10.100.10.30:27017/LibreChat")

# Feature Flags
ORCHESTRATOR_DEBUG = os.getenv("ORCHESTRATOR_DEBUG", "false").lower() in ("true", "1", "yes")
ADMIN_EMAILS = [e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()]

def is_debug_allowed_for_user(user_email: str = "", user_id: str = "") -> bool:
    """Verifica se o usuário tem permissão para visualizar blocos de debug do orquestrador."""
    if not ORCHESTRATOR_DEBUG:
        return False
    if not ADMIN_EMAILS:
        return True
    u_email = (user_email or "").strip().lower()
    u_id = (user_id or "").strip()
    return (u_email in ADMIN_EMAILS) or (u_id in ADMIN_EMAILS)

# Diretório de Logs
LOGS_DIR = os.getenv("LOGS_DIR", "/app/logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Diretório de Armazenamento e Exibição de Imagens Geradas
IMAGE_STORAGE_DIR = os.getenv("IMAGE_STORAGE_DIR", "/app/generated_images")
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
SERVER_PUBLIC_URL = os.getenv("SERVER_PUBLIC_URL", "https://ai.jmarc.com.br/images/generated").rstrip("/")

def sync_image_to_librechat(local_filepath: str):
    """
    Sincroniza a imagem gerada localmente com a pasta pública do LibreChat na VM1,
    garantindo acesso via https://ai.jmarc.com.br/images/generated/<filename>.
    """
    try:
        import shutil
        import subprocess
        fname = os.path.basename(local_filepath)

        # 1. Se montado diretamente no container (ex: via volume compartilhado)
        target_direct = f"/librechat_images/generated/{fname}"
        if os.path.exists("/librechat_images/generated"):
            shutil.copy2(local_filepath, target_direct)
            print(f"[Sync LibreChat] Imagem copiada diretamente para {target_direct}")
            return

        # 2. Envio via SSH/SCP usando chave privada antigravity (executa com timeout para garantir que o arquivo chegou antes do response)
        ssh_key = "/root/.ssh/antigravity" if os.path.exists("/root/.ssh/antigravity") else "/home/jean/.ssh/antigravity"
        cmd = f"scp -F /dev/null -o StrictHostKeyChecking=no -i {ssh_key} '{local_filepath}' jean@10.100.10.30:~/LibreChat/images/generated/{fname}"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            print(f"[Sync LibreChat] Sincronização concluída com sucesso para 10.100.10.30: {fname}")
        else:
            print(f"[Sync LibreChat Erro SCP]: {res.stderr}")
    except Exception as e:
        print(f"[Aviso Sincronização Imagem LibreChat]: {e}")


def write_log(entry: dict):
    """Grava um registro de log em formato JSON Lines (.jsonl) particionado por data (São Paulo / BR)."""
    try:
        today_str = get_now_local().strftime("%Y-%m-%d")
        log_file_path = os.path.join(LOGS_DIR, f"requests_{today_str}.jsonl")
        
        log_line = json.dumps(entry, ensure_ascii=False)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
            f.flush()
        print(f"[Log Registrado] Salvo em {log_file_path}")
    except Exception as e:
        print(f"[Falha ao gravar log]: {e}")

# ---------------------------------------------------------
# Gestão de Estado de Sessão e Continuidade Conversacional
# ---------------------------------------------------------
CONVERSATION_SESSIONS: dict = {}

def get_conversation_state(conversation_id: str, messages: list = None) -> dict:
    """
    Retorna o estado da conversa (categoria anterior, modelo, prompt anterior).
    Caso a sessão não esteja em memória, tenta inferir contextualmente a partir do histórico de mensagens.
    """
    if conversation_id and conversation_id in CONVERSATION_SESSIONS:
        return CONVERSATION_SESSIONS[conversation_id]

    state = {
        "last_category": None,
        "last_model": None,
        "last_prompt": None,
        "last_assistant_snippet": None,
        "updated_at": 0
    }

    if messages and len(messages) >= 2:
        # Tenta localizar a penúltima mensagem do usuário e a resposta do assistente
        user_msgs = [m for m in messages[:-1] if m.get("role") == "user"]
        asst_msgs = [m for m in messages[:-1] if m.get("role") == "assistant"]
        if user_msgs:
            u_content = user_msgs[-1].get("content", "")
            if isinstance(u_content, str):
                state["last_prompt"] = u_content
            elif isinstance(u_content, list):
                state["last_prompt"] = " ".join([p.get("text", "") for p in u_content if isinstance(p, dict) and p.get("type") == "text"])
        if asst_msgs:
            a_content = asst_msgs[-1].get("content", "")
            if isinstance(a_content, str):
                state["last_assistant_snippet"] = a_content[:300]
            elif isinstance(a_content, list):
                state["last_assistant_snippet"] = " ".join([p.get("text", "") for p in a_content if isinstance(p, dict) and p.get("type") == "text"])[:300]

    return state

def update_conversation_state(conversation_id: str, category: str, target_model: str, prompt: str, assistant_snippet: str = ""):
    """Atualiza o estado da sessão com a última categoria, modelo e prompt utilizados."""
    if not conversation_id:
        return
    CONVERSATION_SESSIONS[conversation_id] = {
        "last_category": category,
        "last_model": target_model,
        "last_prompt": prompt,
        "last_assistant_snippet": assistant_snippet[:300] if assistant_snippet else "",
        "updated_at": time.time()
    }

def message_has_image(msg: dict) -> bool:
    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part):
                return True
    return False

def history_has_images(messages: list) -> bool:
    return any(message_has_image(m) for m in messages)

def sanitize_messages_for_ollama(messages: list, keep_images: bool = False) -> list:
    sanitized = []
    for msg in messages:
        role = msg.get("role", "user")
        raw_content = msg.get("content", "")

        if isinstance(raw_content, list):
            if keep_images:
                clean_blocks = [item for item in raw_content if isinstance(item, dict) and item.get("type") in ["text", "image_url"]]
                if clean_blocks:
                    sanitized.append({"role": role, "content": clean_blocks})
            else:
                text_blocks = [item.get("text", "") for item in raw_content if isinstance(item, dict) and item.get("type") == "text"]
                content_str = "\n".join(text_blocks).strip()
                if content_str:
                    sanitized.append({"role": role, "content": content_str})
        elif raw_content is not None and str(raw_content).strip():
            sanitized.append({"role": role, "content": str(raw_content).strip()})

    return sanitized

def ensure_alternating_roles(messages: list) -> list:
    if not messages:
        return []

    merged = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if not merged:
            merged.append({"role": role, "content": content})
            continue

        prev_role = merged[-1]["role"]

        if role == prev_role and role != "system":
            prev_content = merged[-1]["content"]

            if isinstance(prev_content, str) and isinstance(content, str):
                merged[-1]["content"] = f"{prev_content}\n\n{content}"
            else:
                p_list = prev_content if isinstance(prev_content, list) else [{"type": "text", "text": str(prev_content)}]
                c_list = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
                merged[-1]["content"] = p_list + c_list
        else:
            merged.append({"role": role, "content": content})

    return merged

async def resolve_image_to_base64(image_source: dict) -> dict:
    """
    Converte qualquer imagem de entrada (URL local da VM1/VM2, caminho relativo ou URL externa)
    em base64 inline ('data:image/...;base64,...') para que modelos na nuvem (Google Gemini) possam
    processá-la sem sofrer restrições de rede de IPs privados (RFC1918).
    """
    if not image_source:
        return image_source

    img_val = image_source.get("image_url", {})
    if isinstance(img_val, dict):
        url_str = img_val.get("url", "")
    elif isinstance(img_val, str):
        url_str = img_val
    else:
        url_str = image_source.get("url", "")

    if not url_str:
        return image_source

    if str(url_str).startswith("data:image/"):
        return {"type": "image_url", "image_url": {"url": url_str}}

    raw_bytes = None
    content_type = "image/png"

    # Caso A: Imagem armazenada localmente na pasta de imagens geradas
    clean_name = os.path.basename(str(url_str).split("?")[0])
    local_file = os.path.join(IMAGE_STORAGE_DIR, clean_name)
    if os.path.exists(local_file):
        try:
            with open(local_file, "rb") as f:
                raw_bytes = f.read()
            print(f"[Imagem Base64] Lida do armazenamento local: {local_file} ({len(raw_bytes)} bytes)")
        except Exception as read_err:
            print(f"[Aviso Leitura Local Imagem]: {read_err}")

    # Caso B: URL HTTP/HTTPS ou caminho relativo do LibreChat
    if not raw_bytes:
        target_url = str(url_str)
        if target_url.startswith("/images/") or target_url.startswith("/api/files/"):
            target_url = f"http://10.100.10.30:3080{target_url}"

        try:
            print(f"[Imagem Base64] Baixando imagem para conversão base64 de: '{target_url}'...")
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(target_url)
                if r.status_code == 200:
                    raw_bytes = r.content
                    content_type = r.headers.get("content-type", "image/png").split(";")[0]
                    print(f"[Imagem Base64] Download concluído com sucesso ({len(raw_bytes)} bytes, tipo: {content_type})")
                else:
                    print(f"[Aviso Download Imagem] HTTP {r.status_code} ao buscar '{target_url}'")
        except Exception as dl_err:
            print(f"[Falha Download Imagem]: {dl_err}")

    if raw_bytes:
        b64_encoded = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:{content_type};base64,{b64_encoded}"
        return {"type": "image_url", "image_url": {"url": data_url}}

    return image_source

def is_context_dependent_prompt(prompt: str, messages: list) -> bool:
    """
    Identifica se uma pergunta/instrução tem relação com a anterior (é anafórica ou de continuação).
    Evita falsos cache hits e preserva a linhagem da conversa para o classificador e para o cache.
    """
    if not messages or len(messages) <= 1:
        return False

    has_assistant_history = any(m.get("role") == "assistant" for m in messages[:-1])
    if not has_assistant_history:
        return False

    p = prompt.strip().lower()

    # 1. Pronomes demonstrativos / anafóricos em português e referências a temas
    relative_pronouns = (
        r"\b(disso|disto|daquilo|dele|dela|deles|delas|desse|deste|daquele|desses|destes|daqueles|"
        r"dessas|destas|daquelas|nisso|nisto|naquilo|nesse|neste|naquele|nesses|nestes|naqueles|"
        r"nessas|nestas|naquelas|acima|anterior|citado|citada|mencionado|mencionada|referido|referida|"
        r"o mesmo|a mesma|os mesmos|as mesmas|isso|isto|aquilo|esse|este|aquele|essa|esta|aquela|"
        r"sobre o tema|sobre o assunto|sobre isso|sobre ele|sobre ela|a respeito disso|a respeito|deste tema|desse tema)\b"
    )
    if re.search(relative_pronouns, p):
        return True

    # 2. Comandos de continuação / manipulação relativa (sem sujeito autônomo)
    continuation_commands = (
        r"^(classifique|ordene|organize|resuma|explique|detalhe|continue|liste|traduza|coloque em|"
        r"converta|compare|qual deles|quais são eles|e sobre|e o|e a|e se|por que|porque|qual o próximo|"
        r"e agora|qual o melhor|qual o pior|mais um|outro exemplo|mostre em tabela|em formato de tabela|"
        r"ordem crescente|ordem decrescente|do maior|do menor|em ordem|faça uma dissertação|uma dissertação|"
        r"faça uma redação|escreva uma redação|faça um resumo|desenvolva o tema|aprofunde|prossiga|estava me referindo)\b"
    )
    if re.search(continuation_commands, p):
        return True

    # 3. Frases curtas (até 10 palavras) em conversas ativas tendem a ser relativas/elípticas
    words = p.split()
    if len(words) <= 10:
        return True

    return False

def clean_location_query(prompt: str) -> str:
    """Extrai a provável cidade ou localidade de uma pergunta sobre clima."""
    cleaned = prompt
    patterns_to_remove = [
        r"(?i)como\s+est[aá]\s+o\s+tempo\s+(em|de|para|na|no)?",
        r"(?i)qual\s+(é(\s+a)?|a)?\s*previs[aã]o\s+d[oe]\s+tempo\s+(para|em|de|na|no)?",
        r"(?i)qual\s+(é(\s+a)?|a)?\s*(temperatura|clima|previs[aã]o)\s+(para|em|de|na|no)?",
        r"(?i)vai\s+chover\s+(hoje\s+)?(em|de|para|na|no)?",
        r"(?i)previs[aã]o\s+d[oe]\s+tempo\s+(para|em|de|na|no)?",
        r"(?i)\b(previs[aã]o|temperatura|clima|tempo)\s+(para|em|de|na|no)?",
        r"(?i)\b(hoje|agora|amanh[aã]|para hoje|de hoje|nesta semana)\b"
    ]
    for pat in patterns_to_remove:
        cleaned = re.sub(pat, " ", cleaned)

    cleaned = cleaned.replace("?", "").replace("!", "").strip()
    cleaned = re.sub(r"^(em|de|para|na|no)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "/" in cleaned:
        city = cleaned.split("/")[0].strip()
        return f"{city}, BR"
    elif " - " in cleaned:
        city = cleaned.split(" - ")[0].strip()
        return f"{city}, BR"

    if cleaned and "," not in cleaned:
        return f"{cleaned}, BR"

    return cleaned if cleaned else prompt
