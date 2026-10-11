import httpx
import time
import datetime
from typing import Optional, Dict, Any, List

async def fetch_currency_rates(query: str) -> Optional[Dict[str, Any]]:
    """
    Consulta cotações e séries temporais de moedas via AwesomeAPI (USD, EUR, BTC, GBP).
    Retorna dicionário estruturado com dados reais e oficiais.
    """
    q = query.lower()
    pair = None
    import re
    if "dolar" in q or "dólar" in q or re.search(r"\busd\b", q):
        pair = "USD-BRL"
    elif "euro" in q or re.search(r"\beur\b", q):
        pair = "EUR-BRL"
    elif "bitcoin" in q or re.search(r"\bbtc\b", q):
        pair = "BTC-BRL"
    elif "libra esterlina" in q or re.search(r"\bgbp\b", q) or re.search(r"\bcotação da libra\b|\bmoeda libra\b", q):
        pair = "GBP-BRL"

    if not pair:
        return None

    # Verifica se o usuário pediu série histórica / período / dias / gráfico de evolução
    is_historical = any(k in q for k in [
        "período", "periodo", "últimos", "ultimos", "dias", "semana", "mês", "mes",
        "evolução", "evolucao", "histórico", "historico", "gráfico", "grafico", "planilha"
    ])

    num_days = 15
    if "30" in q or "mês" in q or "mes" in q:
        num_days = 30
    elif "60" in q or "2 meses" in q or "dois meses" in q:
        num_days = 60
    elif "90" in q or "trimestre" in q or "3 meses" in q:
        num_days = 90
    elif "7" in q or "semana" in q:
        num_days = 7

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            if is_historical:
                url = f"https://economia.awesomeapi.com.br/json/daily/{pair}/{num_days}"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    # A API retorna em ordem decrescente (mais recente primeiro). Invertemos para ordem cronológica!
                    data_chronological = list(reversed(data))
                    series = []
                    for item in data_chronological:
                        ts = int(item.get("timestamp", 0))
                        dt_str = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else item.get("create_date", "")[:10]
                        bid_val = float(item.get("bid", 0.0))
                        high_val = float(item.get("high", 0.0))
                        low_val = float(item.get("low", 0.0))
                        series.append({
                            "data": dt_str,
                            "cotacao_fechamento": round(bid_val, 4),
                            "maxima": round(high_val, 4),
                            "minima": round(low_val, 4)
                        })
                    
                    latest = data[0] if data else {}
                    return {
                        "source": "AwesomeAPI (Mercado Financeiro / Câmbio Oficial)",
                        "pair": pair,
                        "tipo": "serie_historica",
                        "dias_solicitados": len(series),
                        "cotacao_atual": float(latest.get("bid", 0.0)),
                        "maxima_hoje": float(latest.get("high", 0.0)),
                        "minima_hoje": float(latest.get("low", 0.0)),
                        "serie": series
                    }
            else:
                # Consulta cotação pontual em tempo real
                url = f"https://economia.awesomeapi.com.br/last/{pair}"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    key = pair.replace("-", "")
                    rate_info = data.get(key, {})
                    return {
                        "source": "AwesomeAPI (Mercado Financeiro / Câmbio Oficial)",
                        "pair": pair,
                        "tipo": "cotacao_tempo_real",
                        "nome": rate_info.get("name"),
                        "cotacao_atual": float(rate_info.get("bid", 0.0)),
                        "maxima": float(rate_info.get("high", 0.0)),
                        "minima": float(rate_info.get("low", 0.0)),
                        "variacao_pct": f"{float(rate_info.get('pctChange', 0.0)):.2f}%",
                        "atualizado_em": rate_info.get("create_date")
                    }
    except Exception as e:
        print(f"[Aviso AwesomeAPI Câmbio]: {e}")

    return None


async def fetch_bcb_indicators(query: str) -> Optional[Dict[str, Any]]:
    """
    Consulta indicadores econômicos oficiais do Banco Central do Brasil (SGS).
    - Selic Meta: Série 432
    - IPCA Mensal: Série 433
    - CDI Diário: Série 12
    - Poupança: Série 196
    """
    q = query.lower()
    serie_id = None
    indicator_name = None

    if "selic" in q:
        serie_id = "432"
        indicator_name = "Taxa Selic Meta (% a.a.)"
    elif "ipca" in q or "inflação" in q or "inflacao" in q:
        serie_id = "433"
        indicator_name = "IPCA Mensal (%)"
    elif "cdi" in q:
        serie_id = "12"
        indicator_name = "Taxa CDI (% a.d.)"
    elif "poupanca" in q or "poupança" in q:
        serie_id = "196"
        indicator_name = "Rendimento da Poupança (% a.m.)"

    if not serie_id:
        return None

    try:
        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie_id}/dados/ultimos/12?formato=json"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                latest = data[-1] if data else {}
                return {
                    "source": "Banco Central do Brasil (BCB - Sistema Gerenciador de Séries Temporais)",
                    "indicador": indicator_name,
                    "serie_id": serie_id,
                    "valor_atual": float(latest.get("valor", 0.0)) if latest.get("valor") else latest.get("valor"),
                    "data_referencia": latest.get("data"),
                    "historico_recente": data
                }
    except Exception as e:
        print(f"[Aviso BCB SGS]: {e}")

    return None
