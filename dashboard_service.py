import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, List

LOGS_DIR = "/app/logs"
DOCS_DIR = "/app/docs_view"
ORCHESTRATOR_DOC = "/app/docs_view/langchain_orchestrator_architecture_v2.md"
README_DOC = "/app/docs_view/README.md"

def get_available_log_dates() -> List[str]:
    """Retorna lista de datas com logs disponíveis (formato YYYY-MM-DD)."""
    files = glob.glob(os.path.join(LOGS_DIR, "requests_*.jsonl"))
    dates = []
    for f in sorted(files, reverse=True):
        basename = os.path.basename(f)
        # requests_YYYY-MM-DD.jsonl
        d = basename.replace("requests_", "").replace(".jsonl", "")
        dates.append(d)
    return dates

def load_logs_for_date(target_date: str = None) -> List[Dict[str, Any]]:
    """Carrega os logs do dia especificado ou do mais recente."""
    dates = get_available_log_dates()
    if not dates:
        return []
    
    if not target_date or target_date not in dates:
        target_date = dates[0]
        
    log_file = os.path.join(LOGS_DIR, f"requests_{target_date}.jsonl")
    entries = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    # Retorna em ordem decrescente cronológica (mais recentes primeiro)
    return list(reversed(entries))

def calculate_dashboard_metrics(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula estatísticas de desempenho, rotas, modelos e latência a partir dos logs."""
    total_requests = len(entries)
    if total_requests == 0:
        return {
            "total_requests": 0,
            "success_rate": 0.0,
            "avg_latency": 0.0,
            "categories": {},
            "models": {},
            "status_counts": {},
            "hourly_distribution": {},
            "cache_hits": 0,
            "cache_hit_rate": 0.0
        }
    
    success_count = sum(1 for e in entries if e.get("status") == "success")
    cache_hits = sum(1 for e in entries if e.get("cache_hit") is True)
    latencies = [float(e.get("duration_seconds", 0)) for e in entries if e.get("duration_seconds") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    
    categories = {}
    models = {}
    status_counts = {}
    hourly = {f"{h:02d}:00": 0 for h in range(24)}
    
    for e in entries:
        cat = e.get("category", "UNKNOWN")
        categories[cat] = categories.get(cat, 0) + 1
        
        mod = e.get("target_model", "Direct / Outro")
        models[mod] = models.get(mod, 0) + 1
        
        st = "success" if e.get("status") == "success" else "error"
        status_counts[st] = status_counts.get(st, 0) + 1
        
        # Hora
        ts = e.get("timestamp", "")
        if ts and "T" in ts:
            try:
                hour = ts.split("T")[1][:2]
                k = f"{hour}:00"
                if k in hourly:
                    hourly[k] += 1
            except Exception:
                pass
                
    return {
        "total_requests": total_requests,
        "success_rate": round((success_count / total_requests) * 100, 1),
        "avg_latency": avg_latency,
        "cache_hits": cache_hits,
        "cache_hit_rate": round((cache_hits / total_requests) * 100, 1),
        "categories": categories,
        "models": models,
        "status_counts": status_counts,
        "hourly_distribution": hourly
    }

def read_markdown_document(doc_name: str) -> str:
    """Lê o conteúdo do arquivo de documentação solicitado."""
    if doc_name == "architecture":
        path = ORCHESTRATOR_DOC
    elif doc_name == "readme":
        path = README_DOC
    else:
        path = ORCHESTRATOR_DOC
        
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"Arquivo de documentação não encontrado em: {path}"
