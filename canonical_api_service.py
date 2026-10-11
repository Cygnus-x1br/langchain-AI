import re
import httpx
import time
from typing import Optional, Dict, Any, List

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ---------------------------------------------------------
# 1. API DE CEP & LOCALIZAÇÃO GEOGRÁFICA (BrasilAPI + ViaCEP)
# ---------------------------------------------------------
async def fetch_cep_location(query: str) -> Optional[Dict[str, Any]]:
    """
    Identifica padrões de CEP (8 dígitos) e consulta BrasilAPI / ViaCEP.
    Retorna endereço completo, bairro, cidade, UF, coordenadas e contexto geográfico.
    """
    # Procura CEP no formato 00000-000 ou 00000000
    cep_match = re.search(r"\b(\d{5})-?(\d{3})\b", query)
    if not cep_match:
        return None

    raw_cep = f"{cep_match.group(1)}{cep_match.group(2)}"
    t0 = time.time()

    async with httpx.AsyncClient(timeout=6.0, headers=HEADERS) as client:
        # 1. Tenta BrasilAPI v2 (traz coordenadas geográficas lat/lon)
        try:
            resp = await client.get(f"https://brasilapi.com.br/api/cep/v2/{raw_cep}")
            if resp.status_code == 200:
                data = resp.json()
                loc = data.get("location", {}).get("coordinates", {})
                lat = loc.get("latitude")
                lon = loc.get("longitude")
                return {
                    "source": "BrasilAPI (v2)",
                    "cep": f"{cep_match.group(1)}-{cep_match.group(2)}",
                    "logradouro": data.get("street") or "",
                    "bairro": data.get("neighborhood") or "",
                    "cidade": data.get("city") or "",
                    "estado": data.get("state") or "",
                    "latitude": lat,
                    "longitude": lon,
                    "ibge": data.get("ibge", {}).get("city", ""),
                    "duration_seconds": round(time.time() - t0, 3)
                }
        except Exception as e:
            print(f"[Aviso BrasilAPI CEP]: {e}")

        # 2. Fallback ViaCEP
        try:
            resp_viacep = await client.get(f"https://viacep.com.br/ws/{raw_cep}/json/")
            if resp_viacep.status_code == 200:
                data = resp_viacep.json()
                if "erro" not in data:
                    return {
                        "source": "ViaCEP",
                        "cep": data.get("cep") or f"{cep_match.group(1)}-{cep_match.group(2)}",
                        "logradouro": data.get("logradouro") or "",
                        "bairro": data.get("bairro") or "",
                        "cidade": data.get("localidade") or "",
                        "estado": data.get("uf") or "",
                        "ddd": data.get("ddd") or "",
                        "ibge": data.get("ibge") or "",
                        "duration_seconds": round(time.time() - t0, 3)
                    }
        except Exception as e:
            print(f"[Aviso ViaCEP]: {e}")

    return None


# ---------------------------------------------------------
# 2. RESOLUÇÃO GEOGRÁFICA DE MUNICÍPIOS / BAIRROS (OpenStreetMap Nominatim)
# ---------------------------------------------------------
async def resolve_geographic_context(location_name: str) -> Optional[Dict[str, Any]]:
    """
    Resolve coordenadas e contexto territorial (cidade, estado, microrregião) de cidades/bairros
    citados pelo usuário (ex: 'Santo André', 'Moema, SP', 'Campinas').
    Permite enriquecer buscas locais de restaurantes, serviços e comércios.
    """
    if not location_name or len(location_name.strip()) < 3:
        return None

    clean_loc = location_name.strip()
    # Adiciona 'Brasil' caso não contenha país para precisão
    search_q = f"{clean_loc}, Brasil" if "brasil" not in clean_loc.lower() else clean_loc

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=6.0, headers=HEADERS) as client:
            url = f"https://nominatim.openstreetmap.org/search?q={search_q}&format=json&limit=1&addressdetails=1"
            resp = await client.get(url)
            if resp.status_code == 200:
                results = resp.json()
                if results and len(results) > 0:
                    top = results[0]
                    addr = top.get("address", {})
                    cidade = addr.get("city") or addr.get("town") or addr.get("municipality") or clean_loc
                    estado = addr.get("state") or ""
                    bairro = addr.get("suburb") or addr.get("neighbourhood") or ""
                    lat = top.get("lat")
                    lon = top.get("lon")
                    return {
                        "source": "OpenStreetMap Nominatim",
                        "display_name": top.get("display_name", ""),
                        "cidade": cidade,
                        "bairro": bairro,
                        "estado": estado,
                        "latitude": lat,
                        "longitude": lon,
                        "duration_seconds": round(time.time() - t0, 3)
                    }
    except Exception as e:
        print(f"[Aviso Geocoding Nominatim]: {e}")

    return None


# ---------------------------------------------------------
# 3. CONSULTA OFICIAL DE CNPJ / EMPRESAS (BrasilAPI)
# ---------------------------------------------------------
async def fetch_cnpj_data(query: str) -> Optional[Dict[str, Any]]:
    """
    Identifica CNPJ (14 dígitos, formatado ou puro) e consulta os dados oficiais na Receita Federal via BrasilAPI.
    Retorna Razão Social, Nome Fantasia, Situação, CNAE, Endereço e Sócios.
    """
    # Procura formato 00.000.000/0000-00 ou 14 dígitos consecutivos
    cnpj_match = re.search(r"\b(\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})\b|\b(\d{14})\b", query)
    if not cnpj_match:
        return None

    matched_str = cnpj_match.group(1) or cnpj_match.group(2)
    raw_cnpj = re.sub(r"\D", "", matched_str)
    if len(raw_cnpj) != 14:
        return None

    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=8.0, headers=HEADERS) as client:
            resp = await client.get(f"https://brasilapi.com.br/api/cnpj/v1/{raw_cnpj}")
            if resp.status_code == 200:
                data = resp.json()
                # Monta resumo estruturado dos dados cadastrais
                socios = [s.get("nome_socio") for s in data.get("qsa", []) if s.get("nome_socio")]
                return {
                    "source": "BrasilAPI / Receita Federal",
                    "cnpj": f"{raw_cnpj[:2]}.{raw_cnpj[2:5]}.{raw_cnpj[5:8]}/{raw_cnpj[8:12]}-{raw_cnpj[12:]}",
                    "razao_social": data.get("razao_social", ""),
                    "nome_fantasia": data.get("nome_fantasia", "") or "Não informado",
                    "situacao_cadastral": data.get("descricao_situacao_cadastral", ""),
                    "data_inicio_atividade": data.get("data_inicio_atividade", ""),
                    "cnae_fiscal_descricao": data.get("cnae_fiscal_descricao", ""),
                    "natureza_juridica": data.get("natureza_juridica", ""),
                    "capital_social": data.get("capital_social", 0),
                    "logradouro": f"{data.get('descricao_tipo_de_logradouro', '')} {data.get('logradouro', '')}".strip(),
                    "numero": data.get("numero", ""),
                    "bairro": data.get("bairro", ""),
                    "municipio": data.get("municipio", ""),
                    "uf": data.get("uf", ""),
                    "cep": data.get("cep", ""),
                    "socios": socios[:5],
                    "duration_seconds": round(time.time() - t0, 3)
                }
    except Exception as e:
        print(f"[Aviso BrasilAPI CNPJ]: {e}")

    return None


# ---------------------------------------------------------
# 4. CONSULTA DE PREÇO OFICIAL DE VEÍCULOS / TABELA FIPE (BrasilAPI)
# ---------------------------------------------------------
async def fetch_fipe_vehicle_price(query: str) -> Optional[Dict[str, Any]]:
    """
    Identifica menções explícitas à Tabela FIPE e código FIPE (ex: 004123-8) ou marca/modelo.
    """
    q = query.lower()
    if "fipe" not in q:
        return None

    # Verifica se há código FIPE direto (ex: 004123-8 ou 0041238)
    code_match = re.search(r"\b(\d{6}-?\d{1})\b", query)
    if code_match:
        fipe_code = code_match.group(1).replace("-", "")
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=8.0, headers=HEADERS) as client:
                resp = await client.get(f"https://brasilapi.com.br/api/fipe/preco/v1/{fipe_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 0:
                        top = data[0]
                        return {
                            "source": "BrasilAPI / Tabela FIPE",
                            "codigo_fipe": top.get("codigoFipe", ""),
                            "modelo": top.get("modelo", ""),
                            "marca": top.get("marca", ""),
                            "ano_modelo": top.get("anoModelo", ""),
                            "combustivel": top.get("combustivel", ""),
                            "valor": top.get("valor", ""),
                            "mes_referencia": top.get("mesReferencia", ""),
                            "duration_seconds": round(time.time() - t0, 3)
                        }
        except Exception as e:
            print(f"[Aviso BrasilAPI FIPE Código]: {e}")

    return None
