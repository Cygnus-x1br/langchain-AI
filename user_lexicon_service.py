import os
import re
import sqlite3
import time
from typing import Optional, Tuple, Dict, Any, List

LEXICON_DB_PATH = os.getenv("LEXICON_DB_PATH", "/app/data/user_lexicons.db")
if not os.path.exists(os.path.dirname(LEXICON_DB_PATH)):
    LEXICON_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "user_lexicons.db")


def _get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(LEXICON_DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_user_lexicon_db():
    """Inicializa a tabela de vocabulário e padrões aprendidos por usuário."""
    try:
        os.makedirs(os.path.dirname(LEXICON_DB_PATH), exist_ok=True)
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    category TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    sample_prompt TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, pattern, category)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_pattern ON user_learned_patterns(user_id, pattern)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_cat ON user_learned_patterns(user_id, category)")
            conn.commit()
    except Exception as e:
        print(f"[Aviso Inicialização Lexicon DB]: {e}")


# Inicializa o banco no carregamento do módulo
init_user_lexicon_db()


def extract_candidate_patterns(prompt: str) -> List[str]:
    """
    Extrai frases candidatas (n-grams de 2 a 4 palavras) e expressões verbais significativas do prompt.
    Filtra preposições soltas e foca em expressões de intenção.
    """
    if not prompt:
        return []
    
    clean = re.sub(r"[^\w\s]", " ", prompt.lower()).strip()
    words = clean.split()
    if len(words) < 2:
        return []

    candidates = []
    # N-grams de 2, 3 e 4 palavras
    for n in (2, 3, 4):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            # Evita n-grams puramente de stopwords ou verbos imperativos genéricos combinados com artigos
            stopwords = {"de", "do", "da", "em", "no", "na", "para", "pro", "pra", "um", "uma", "os", "as", "o", "a", "e", "que", "se", "com", "por"}
            generic_starts = {"crie um", "crie uma", "faça um", "faça uma", "gere um", "gere uma", "monte um", "monte uma", "com a", "com o", "sobre a", "sobre o"}
            if ngram in generic_starts:
                continue
            ngram_words = set(words[i:i+n])
            if len(ngram_words - stopwords) >= 1:
                candidates.append(ngram)

    return candidates[:8] # Limita aos primeiros candidatos mais relevantes


def match_user_lexicon(user_id: str, prompt: str) -> Optional[Tuple[str, str]]:
    """
    Verifica se o prompt do usuário contém algum padrão previamente aprendido para este usuário específico.
    Requer que o padrão tenha sido reforçado (frequência >= 2 ou alta confiança).
    Retorna (categoria, routing_method) ou None.
    """
    if not user_id or not prompt:
        return None

    p_lower = prompt.lower().strip()
    social_signals = ["instagram", "insta", "linkedin", "facebook", "post", "postagem", "postgem", "publica", "stories", "feed", "tiktok"]
    has_social = any(s in p_lower for s in social_signals)

    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT pattern, category, frequency, confidence
                FROM user_learned_patterns
                WHERE user_id = ? AND frequency >= 2
                ORDER BY frequency DESC, LENGTH(pattern) DESC
            """, (user_id,))
            rows = cursor.fetchall()

            for row in rows:
                pattern = row["pattern"]
                # Proteção: padrão aprendido não pode desviar pedidos de redes sociais para IMAGE_GENERATION
                if row["category"] == "IMAGE_GENERATION" and has_social:
                    continue

                # Se o padrão aprendido estiver contido no prompt
                pattern_re = r"\b" + re.escape(pattern) + r"\b"
                if re.search(pattern_re, p_lower):
                    cat = row["category"]
                    method = f"user_lexicon_learned_{pattern.replace(' ', '_')}"
                    print(f"[Lexicon Match Usuário] Padrão aprendido '{pattern}' (Freq: {row['frequency']}) reconhecido para {user_id} -> {cat}")
                    return cat, method
                if f" {pattern} " in f" {p_lower} " or p_lower.startswith(f"{pattern} ") or p_lower.endswith(f" {pattern}") or p_lower == pattern:
                    cat = row["category"]
                    freq = row["frequency"]
                    # Atualiza o timestamp de último uso
                    cursor.execute("""
                        UPDATE user_learned_patterns
                        SET last_used_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND pattern = ? AND category = ?
                    """, (user_id, pattern, cat))
                    conn.commit()
                    print(f"[Lexicon Match Usuário] Padrão aprendido '{pattern}' (Freq: {freq}) reconhecido para {user_id} -> {cat}")
                    return cat, f"user_lexicon_learned_{pattern.replace(' ', '_')}"
    except Exception as e:
        print(f"[Aviso Consulta Lexicon Usuário]: {e}")

    return None


def learn_user_pattern(user_id: str, prompt: str, category: str):
    """
    Analisa o prompt de uma requisição bem-sucedida e registra/reforça padrões linguísticos associados ao usuário.
    """
    if not user_id or not prompt or not category:
        return
    if category in ["CASUAL_CHAT", "UNKNOWN"]:
        return

    # Extrai n-grams candidatos
    patterns = extract_candidate_patterns(prompt)
    if not patterns:
        return

    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            for pat in patterns:
                # Upsert padrão aprendido para o usuário
                cursor.execute("""
                    INSERT INTO user_learned_patterns (user_id, pattern, category, frequency, sample_prompt, last_used_at)
                    VALUES (?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, pattern, category) DO UPDATE SET
                        frequency = frequency + 1,
                        last_used_at = CURRENT_TIMESTAMP,
                        sample_prompt = excluded.sample_prompt
                """, (user_id, pat, category, prompt[:200]))
            conn.commit()
    except Exception as e:
        print(f"[Aviso Aprendizado Lexicon]: {e}")


def remove_user_learned_patterns_for_prompt(user_id: str, prompt: str) -> int:
    """
    Remove padrões aprendidos no SQLite para o usuário a partir de um prompt rejeitado por feedback negativo (thumbsDown).
    Evita que associações errôneas ou induzidas por alucinação continuem influenciando rotas futuras.
    """
    if not user_id or not prompt:
        return 0

    patterns = extract_candidate_patterns(prompt)
    if not patterns:
        return 0

    removed = 0
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join("?" for _ in patterns)
            cursor.execute(f"""
                DELETE FROM user_learned_patterns
                WHERE user_id = ? AND pattern IN ({placeholders})
            """, [user_id] + patterns)
            removed = cursor.rowcount
            conn.commit()
            if removed > 0:
                print(f"[Feedback Purge Lexicon] Removidos {removed} padrões aprendidos para usuário '{user_id}' devido a feedback negativo.")
    except Exception as e:
        print(f"[Aviso Remoção Lexicon por Feedback]: {e}")
    return removed


def get_all_learned_patterns(limit: int = 100) -> List[Dict[str, Any]]:
    """Retorna lista de padrões aprendidos de todos os usuários para visualização no Dashboard."""
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, pattern, category, frequency, confidence, sample_prompt, last_used_at
                FROM user_learned_patterns
                ORDER BY frequency DESC, last_used_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[Aviso Listagem Lexicon]: {e}")
        return []


def delete_learned_pattern(pattern_id: int) -> bool:
    """Remove um padrão aprendido manualmente pelo administrador."""
    try:
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_learned_patterns WHERE id = ?", (pattern_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[Aviso Exclusão Lexicon]: {e}")
        return False
