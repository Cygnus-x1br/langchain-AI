import os
import re
import time
import uuid
import json
from typing import Tuple, List, Dict, Any, Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend não interativo para servidor
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    IMAGE_STORAGE_DIR,
    SERVER_PUBLIC_URL,
    get_now_local,
    sync_image_to_librechat,
    LITELLM_BASE_URL,
    LITELLM_API_KEY
)
from langchain_openai import ChatOpenAI

# Estilo visual moderno para os gráficos (Seaborn + Matplotlib)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--"
})

# Modelo Ultrarrápido e Gratuito para Estruturação de Dados e Validação de Eixos (Cloudflare Llama 3.1 8B Fast)
data_llm_fast = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model="Classifier_Fast",
    temperature=0.0,
    max_tokens=4096,
    timeout=12.0
)

# Modelo Fallback
data_llm_fallback = ChatOpenAI(
    base_url=LITELLM_BASE_URL,
    api_key=LITELLM_API_KEY,
    model="ComboFree_Smart",
    temperature=0.1,
    max_tokens=4096,
    timeout=25.0
)


def export_dataframe_to_files(
    df: pd.DataFrame,
    base_name: str = "export"
) -> Tuple[str, str, str, str]:
    """
    Exporta um DataFrame para CSV e Excel (.xlsx) na pasta pública montada e sincroniza com a VM1.
    Retorna (csv_url, xlsx_url, csv_path, xlsx_path).
    """
    timestamp = get_now_local().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    clean_name = re.sub(r"[^\w\-]", "_", base_name).strip("_") or "tabela"

    csv_filename = f"{clean_name}_{timestamp}_{uid}.csv"
    xlsx_filename = f"{clean_name}_{timestamp}_{uid}.xlsx"

    csv_path = os.path.join(IMAGE_STORAGE_DIR, csv_filename)
    xlsx_path = os.path.join(IMAGE_STORAGE_DIR, xlsx_filename)

    # Exporta CSV com codificação UTF-8 com BOM para Excel brasileiro abrir sem desconfigurar acentos
    df.to_csv(csv_path, index=False, encoding="utf-8-sig", sep=";")

    # Exporta Excel (.xlsx) profissionalmente estilizado
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
        workbook = writer.book
        worksheet = writer.sheets["Dados"]

        # Estilização profissional: Cabeçalho com fundo azul escuro, texto branco e bordas
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
        header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        center_align = Alignment(horizontal="center", vertical="center")
        thin_border = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9")
        )

        for col_idx, cell in enumerate(worksheet[1], 1):
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = center_align

        # Ajuste dinâmico da largura das colunas e alinhamento
        for col in worksheet.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                cell.border = thin_border
                val_str = str(cell.value or "")
                if len(val_str) > max_len:
                    max_len = len(val_str)
            worksheet.column_dimensions[col_letter].width = max(max_len + 4, 12)

    # Sincroniza ambos para a pasta pública do LibreChat na VM1
    sync_image_to_librechat(csv_path)
    sync_image_to_librechat(xlsx_path)

    t_now = int(time.time())
    csv_url = f"{SERVER_PUBLIC_URL}/{csv_filename}?t={t_now}"
    xlsx_url = f"{SERVER_PUBLIC_URL}/{xlsx_filename}?t={t_now}"

    print(f"[Data Export] Arquivos gerados: {csv_filename} e {xlsx_filename}")
    return csv_url, xlsx_url, csv_path, xlsx_path


def generate_chart_image(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_col: str,
    title: str = "Gráfico",
    hue_col: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Gera uma visualização de dados profissional via Seaborn/Matplotlib e salva em PNG.
    Retorna (chart_url, chart_path).
    """
    try:
        timestamp = get_now_local().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:6]
        chart_filename = f"chart_{timestamp}_{uid}.png"
        chart_path = os.path.join(IMAGE_STORAGE_DIR, chart_filename)

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        chart_type_lower = (chart_type or "bar").lower()

        # Garante que colunas numéricas estejam com tipo correto
        if y_col in df.columns:
            df[y_col] = pd.to_numeric(df[y_col].astype(str).str.replace(",", "."), errors="coerce").fillna(0)

        if chart_type_lower in ["bar", "colunas", "barras"]:
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax, palette="Blues_d", legend=False)
            # Adiciona rótulos de valores no topo das barras
            for p in ax.patches:
                height = p.get_height()
                if height and not pd.isna(height) and height > 0:
                    val_str = f"{height:,.4f}".rstrip("0").rstrip(".") if height < 10 else f"{height:,.2f}"
                    ax.annotate(
                        val_str.replace(",", "X").replace(".", ",").replace("X", "."),
                        (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="semibold", color="#222222",
                        xytext=(0, 4), textcoords="offset points"
                    )
        elif chart_type_lower in ["line", "linha", "linhas"]:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col, marker="o", markersize=7, linewidth=2.5, ax=ax, color="#1f77b4")
            # Adiciona anotação de valores em cima dos pontos de linha
            y_vals = df[y_col].values
            x_vals = df[x_col].values
            for xv, yv in zip(x_vals, y_vals):
                if pd.notna(yv) and yv > 0:
                    val_str = f"{yv:,.4f}".rstrip("0").rstrip(".") if yv < 10 else f"{yv:,.2f}"
                    ax.annotate(
                        val_str.replace(",", "X").replace(".", ",").replace("X", "."),
                        (xv, yv),
                        ha="center", va="bottom",
                        fontsize=9, fontweight="bold", color="#1a237e",
                        xytext=(0, 6), textcoords="offset points"
                    )
            # Ajuste de margem vertical para que os pontos não fiquem colados no topo/fundo
            if len(y_vals) > 0 and min(y_vals) > 0:
                y_min, y_max = min(y_vals), max(y_vals)
                pad = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else y_min * 0.05
                ax.set_ylim(max(0, y_min - pad), y_max + pad)
        elif chart_type_lower in ["pie", "pizza", "rosca"]:
            values = df[y_col].values
            labels = df[x_col].values
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("pastel"))
            ax.axis("equal")
        elif chart_type_lower in ["scatter", "dispersao", "dispersão"]:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=100, ax=ax, palette="viridis")
        elif chart_type_lower in ["box", "boxplot"]:
            sns.boxplot(data=df, x=x_col, y=y_col, ax=ax, palette="Set2")
        else:
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col if hue_col else x_col, ax=ax, palette="Blues_d", legend=False)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="#1a1a1a")
        if chart_type_lower not in ["pie", "pizza", "rosca"]:
            ax.set_xlabel(str(x_col).capitalize(), fontsize=11, labelpad=10)
            ax.set_ylabel(str(y_col).capitalize(), fontsize=11, labelpad=10)
            plt.xticks(rotation=25, ha="right", fontsize=9)

        plt.tight_layout()
        fig.savefig(chart_path, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

        sync_image_to_librechat(chart_path)
        t_now = int(time.time())
        chart_url = f"{SERVER_PUBLIC_URL}/{chart_filename}?t={t_now}"
        print(f"[Chart Generator] Gráfico gerado com sucesso: {chart_path} ({chart_url})")
        return chart_url, chart_path

    except Exception as e:
        print(f"[Erro Gerador de Gráfico]: {e}")
        try:
            plt.close("all")
        except Exception:
            pass
        return None, None


EXTRACT_DATA_PROMPT = """Você é um Engenheiro de Dados e Analista de BI especializado.
Sua missão é extrair dados estruturados em tabela a partir da solicitação do usuário e do contexto fornecido, e determinar a melhor visualização gráfica (Seaborn/Matplotlib).

DIRETRIZES FUNDAMENTAIS:
1. Monte uma tabela completa, rica e lógica atendendo estritamente ao pedido do usuário e fatos das fontes fornecidas.
2. NUNCA invente números se houver valores reais no texto/fontes. Extraia as datas e valores reais fielmente.
3. Se for uma série temporal (cotações, histórico, evolução por dia/mês/ano), ORDENE OBRIGATORIAMENTE em ordem cronológica crescente (do mais antigo para o mais recente).
4. Para séries temporais ou evolução de preços/cotações, use SEMPRE "chart_type": "line".
5. Os valores numéricos das linhas DEVEM ser floats ou ints puros (ex: 5.2274, e NUNCA strings formatadas com "R$").
6. Responda ESTRITAMENTE em formato JSON no esquema abaixo.

SCHEMA JSON OBRIGATÓRIO:
{{
  "title": "Título conciso da planilha / relatório",
  "columns": ["Coluna1", "Coluna2", "Coluna3"],
  "rows": [
    ["Valor A1", 100, 20.5],
    ["Valor A2", 150, 35.0]
  ],
  "chart_config": {{
    "generate_chart": true,
    "chart_type": "bar | line | pie | scatter | box",
    "x_column": "Coluna1",
    "y_column": "Coluna2",
    "chart_title": "Título explicativo para o gráfico"
  }},
  "analysis_summary": "Resumo executivo de 2 ou 3 parágrafos explicando os dados, destaques e insights principais."
}}

SOLICITAÇÃO DO USUÁRIO:
{user_prompt}
"""


async def execute_data_export_flow(
    user_prompt: str,
    context_text: str = ""
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Pipeline multi-etapa de Inteligência de Dados:
      1. Extração / Estruturação tabular via LLM (JSON).
      2. Conversão para Pandas DataFrame.
      3. Exportação para CSV e Excel (.xlsx).
      4. Renderização do gráfico temático com Seaborn/Matplotlib.
      5. Geração de resposta elegante no LibreChat com tabela Markdown, links de download e imagem do gráfico.
    """
    start_time = time.time()
    tools_called: List[Dict[str, Any]] = []

    full_prompt = EXTRACT_DATA_PROMPT.format(
        user_prompt=f"{user_prompt}\n\nContexto Adicional:\n{context_text}" if context_text else user_prompt
    )

    try:
        t0 = time.time()
        used_model = "Classifier_Fast"
        try:
            res = await data_llm_fast.ainvoke(full_prompt)
        except Exception as fast_err:
            print(f"[Data Model Fallback] Falha no modelo rápido ({fast_err}). Usando ComboFree_Smart...")
            used_model = "ComboFree_Smart"
            res = await data_llm_fallback.ainvoke(full_prompt)

        raw_text = res.content if hasattr(res, "content") else str(res)
        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>", 1)[-1].strip()

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("LLM não retornou JSON de dados válido.")

        parsed_data = json.loads(match.group(0))
        tools_called.append({
            "action": "data_structure_extraction",
            "model": used_model,
            "status": "success",
            "duration_seconds": round(time.time() - t0, 2)
        })

    except Exception as extract_err:
        print(f"[Erro Extração Dados]: {extract_err}")
        return (
            f"Não foi possível estruturar os dados para exportação: {extract_err}",
            [{"action": "data_structure_extraction", "status": f"error: {extract_err}"}]
        )

    title = parsed_data.get("title", "Relatório de Dados")
    columns = parsed_data.get("columns", [])
    rows = parsed_data.get("rows", [])
    chart_config = parsed_data.get("chart_config", {})
    summary = parsed_data.get("analysis_summary", "")

    if not columns or not rows:
        return "Nenhum dado tabular pôde ser extraído da solicitação.", tools_called

    # Criação do Pandas DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # 1. Exportação para CSV e Excel
    t_exp = time.time()
    csv_url, xlsx_url, csv_path, xlsx_path = export_dataframe_to_files(df, base_name=title)
    tools_called.append({
        "action": "spreadsheet_generation",
        "format": ["csv", "xlsx"],
        "status": "success",
        "rows_count": len(df),
        "columns_count": len(df.columns),
        "duration_seconds": round(time.time() - t_exp, 2)
    })

    # 2. Renderização do Gráfico (Seaborn/Matplotlib)
    chart_url = None
    if chart_config.get("generate_chart", True):
        t_chart = time.time()
        chart_type = chart_config.get("chart_type", "bar")
        x_col = chart_config.get("x_column") or columns[0]
        y_col = chart_config.get("y_column") or (columns[1] if len(columns) > 1 else columns[0])
        chart_title = chart_config.get("chart_title", title)

        chart_url, chart_path = generate_chart_image(
            df=df,
            chart_type=chart_type,
            x_col=x_col,
            y_col=y_col,
            title=chart_title
        )
        if chart_url:
            tools_called.append({
                "action": "chart_generation",
                "library": "seaborn+matplotlib",
                "chart_type": chart_type,
                "status": "success",
                "chart_url": chart_url,
                "duration_seconds": round(time.time() - t_chart, 2)
            })

    # 3. Montagem da Resposta Markdown Rica
    try:
        md_table = df.to_markdown(index=False)
    except Exception:
        # Fallback nativo simples e infalível se tabulate não estiver presente
        headers = [str(c) for c in df.columns]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = []
        for _, row in df.iterrows():
            data_rows.append("| " + " | ".join([str(val) for val in row.values]) + " |")
        md_table = "\n".join([header_row, sep_row] + data_rows)

    out = []
    out.append(f"### 📊 {title}\n")

    if summary:
        out.append(f"{summary}\n")

    if chart_url:
        out.append(f"![Gráfico Gerado]({chart_url})\n")

    out.append("#### 📋 Dados Estruturados:")
    out.append(md_table + "\n")

    out.append("#### 📥 Arquivos Disponíveis para Download:")
    out.append(f"- 📗 **[Baixar Planilha Excel (.xlsx)]({xlsx_url})**")
    out.append(f"- 📄 **[Baixar Arquivo Separado por Ponto e Vírgula (.csv)]({csv_url})**\n")
    out.append(f"_Processamento de BI e análise visual concluído em {round(time.time() - start_time, 2)}s._")

    return "\n".join(out), tools_called
