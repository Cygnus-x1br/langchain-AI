import os
import re
import time
import uuid
import datetime
from typing import Tuple, Dict, Any, Optional

try:
    import markdown
except ImportError:
    markdown = None

try:
    from weasyprint import HTML
except ImportError:
    HTML = None

from config import (
    IMAGE_STORAGE_DIR,
    SERVER_PUBLIC_URL,
    get_now_local,
    sync_image_to_librechat
)

def extract_document_title(markdown_text: str, fallback_title: str = "Documento") -> str:
    """Extrai um título limpo a partir do primeiro cabeçalho (# Título) ou início do markdown."""
    if not markdown_text:
        return fallback_title
    
    # 1. Procura primeiro H1 ou H2 markdown
    h_match = re.search(r'^(?:#|##)\s+([^\n]+)', markdown_text, flags=re.MULTILINE)
    if h_match:
        raw_t = h_match.group(1).strip()
        cleaned = re.sub(r'[*_`#\[\]]', '', raw_t).strip()
        if cleaned:
            return cleaned[:80]
            
    # 2. Procura primeira linha em negrito
    b_match = re.search(r'^\*\*([^\n*]+)\*\*', markdown_text, flags=re.MULTILINE)
    if b_match:
        cleaned = b_match.group(1).strip()
        if len(cleaned) >= 3:
            return cleaned[:80]

    # 3. Fallback: primeiras palavras da primeira linha com texto
    for line in markdown_text.splitlines():
        line_clean = re.sub(r'[*_`#\[\]\->]', '', line).strip()
        if len(line_clean) >= 5 and not any(k in line_clean.lower() for k in ["provedor:", "modelo:", "debug:"]):
            return line_clean[:60]

    return fallback_title


def sanitize_markdown_for_pdf(markdown_text: str) -> str:
    """Remove blocos de metadados internos ou debug antes de renderizar o PDF."""
    if not markdown_text:
        return ""
    text = markdown_text
    # Remove blocos [DEBUG] ou dados técnicos internos
    text = re.sub(r'```(?:debug|json|yaml)\s*\{.*?\}\s*```', '', text, flags=re.DOTALL)
    text = re.sub(r'\[DEBUG\].*?\[/DEBUG\]', '', text, flags=re.DOTALL)
    # Remove eventuais links de imagens quebradas ou marcadores de debug do orquestrador
    text = re.sub(r'\[Aviso.*?\]', '', text)
    return text.strip()


def build_styled_html(markdown_text: str, title: str, author: str = "LangChain Orchestrator") -> str:
    """Converte markdown em HTML estruturado com CSS de qualidade de publicação executiva."""
    clean_md = sanitize_markdown_for_pdf(markdown_text)
    
    # Renderiza Markdown para HTML com extensões ricas
    html_body = markdown.markdown(
        clean_md,
        extensions=[
            'fenced_code',
            'tables',
            'nl2br',
            'toc',
            'sane_lists'
        ]
    )

    current_date_str = get_now_local().strftime("%d/%m/%Y %H:%M")
    safe_title = title.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')

    html_document = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>{safe_title}</title>
    <style>
        @page {{
            size: A4;
            margin: 22mm 18mm 22mm 18mm;
            @top-left {{
                content: "{safe_title}";
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 8pt;
                font-weight: 500;
                color: #718096;
                border-bottom: 0.5pt solid #E2E8F0;
                padding-bottom: 5px;
            }}
            @top-right {{
                content: "{current_date_str}";
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 8pt;
                color: #A0AEC0;
                border-bottom: 0.5pt solid #E2E8F0;
                padding-bottom: 5px;
            }}
            @bottom-left {{
                content: "Documento Gerado via IA Orchestrator • {author}";
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 7.5pt;
                color: #A0AEC0;
                border-top: 0.5pt solid #E2E8F0;
                padding-top: 5px;
            }}
            @bottom-right {{
                content: "Página " counter(page) " de " counter(pages);
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 8pt;
                font-weight: 600;
                color: #718096;
                border-top: 0.5pt solid #E2E8F0;
                padding-top: 5px;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.65;
            color: #2D3748;
            margin: 0;
            padding: 0;
        }}

        /* Cabeçalho Visual da Primeira Página */
        .doc-header {{
            border-bottom: 2.5pt solid #2B6CB0;
            padding-bottom: 12px;
            margin-bottom: 22px;
        }}
        .doc-title {{
            font-size: 20pt;
            font-weight: 700;
            color: #1A365D;
            margin: 0 0 6px 0;
            line-height: 1.25;
        }}
        .doc-subtitle {{
            font-size: 9.5pt;
            color: #718096;
            margin: 0;
        }}

        /* Títulos e Hierarquia */
        h1, h2, h3, h4, h5, h6 {{
            color: #1A365D;
            font-weight: 700;
            page-break-after: avoid;
            break-after: avoid;
        }}

        h1 {{
            font-size: 16pt;
            margin-top: 24px;
            margin-bottom: 10px;
            border-bottom: 1.5pt solid #CBD5E0;
            padding-bottom: 4px;
        }}

        h2 {{
            font-size: 13pt;
            color: #2B6CB0;
            margin-top: 20px;
            margin-bottom: 8px;
        }}

        h3 {{
            font-size: 11pt;
            color: #2D3748;
            margin-top: 16px;
            margin-bottom: 6px;
        }}

        p {{
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }}

        /* Listas e Tópicos */
        ul, ol {{
            margin-top: 4px;
            margin-bottom: 12px;
            padding-left: 22px;
        }}

        li {{
            margin-bottom: 4px;
            line-height: 1.55;
        }}

        /* Citações e Destaques */
        blockquote {{
            border-left: 3.5pt solid #3182CE;
            background-color: #F7FAFC;
            padding: 10px 14px;
            margin: 12px 0;
            font-style: italic;
            color: #4A5568;
            border-radius: 0 4px 4px 0;
            page-break-inside: avoid;
        }}

        /* Código Inline e Blocos */
        code {{
            font-family: "DejaVu Sans Mono", "Courier New", Courier, monospace;
            font-size: 8.5pt;
            background-color: #EDF2F7;
            padding: 1.5px 4px;
            border-radius: 3px;
            color: #C53030;
        }}

        pre {{
            background-color: #1A202C;
            color: #EDF2F7;
            padding: 12px 14px;
            border-radius: 6px;
            font-family: "DejaVu Sans Mono", "Courier New", Courier, monospace;
            font-size: 8pt;
            line-height: 1.45;
            overflow-x: auto;
            margin: 14px 0;
            page-break-inside: avoid;
            border: 1px solid #2D3748;
        }}

        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}

        /* Tabelas Executivas */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 8.5pt;
            page-break-inside: avoid;
        }}

        th {{
            background-color: #2B6CB0;
            color: #FFFFFF;
            font-weight: 600;
            text-align: left;
            padding: 7px 10px;
            border: 1px solid #2B6CB0;
        }}

        td {{
            padding: 6px 10px;
            border: 1px solid #E2E8F0;
            vertical-align: top;
        }}

        tr:nth-child(even) td {{
            background-color: #F7FAFC;
        }}

        /* Linhas Horizontais */
        hr {{
            border: none;
            border-top: 1pt solid #E2E8F0;
            margin: 18px 0;
        }}

        /* Links */
        a {{
            color: #3182CE;
            text-decoration: none;
        }}

        /* Elementos Fortes */
        strong, b {{
            font-weight: 600;
            color: #1A202C;
        }}
    </style>
</head>
<body>
    <div class="doc-header">
        <div class="doc-title">{safe_title}</div>
        <div class="doc-subtitle">Relatório Formatado &bullet; {current_date_str}</div>
    </div>

    {html_body}
</body>
</html>
"""
    return html_document


def convert_markdown_to_pdf(
    markdown_text: str,
    title: Optional[str] = None,
    author: str = "LangChain Orchestrator",
    prefix: str = "doc"
) -> Dict[str, Any]:
    """
    Converte um texto em Markdown para um documento PDF profissional (A4) usando WeasyPrint.
    Salva o arquivo em IMAGE_STORAGE_DIR e disponibiliza para download via SERVER_PUBLIC_URL.
    Retorna dicionário com status, pdf_url, filename, size_bytes e title.
    """
    start_time = time.time()
    
    # 1. Determina título do documento
    doc_title = title.strip() if title and title.strip() else extract_document_title(markdown_text)
    
    # 2. Gera nome limpo de arquivo
    timestamp = get_now_local().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    clean_slug = re.sub(r'[^\w\-]', '_', doc_title).strip('_').lower()
    clean_slug = re.sub(r'_+', '_', clean_slug)[:30] or "documento"
    
    filename = f"{prefix}_{clean_slug}_{timestamp}_{uid}.pdf"
    target_path = os.path.join(IMAGE_STORAGE_DIR, filename)
    public_url = f"{SERVER_PUBLIC_URL}/{filename}?t={int(time.time())}"

    # 3. Compila HTML estilizado
    html_content = build_styled_html(markdown_text, title=doc_title, author=author)

    # 4. Renderiza PDF via WeasyPrint
    try:
        html_obj = HTML(string=html_content)
        html_obj.write_pdf(target_path)
        
        file_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        sync_image_to_librechat(target_path)
        duration = round(time.time() - start_time, 2)
        
        print(f"[PDF Export] Documento gerado com sucesso em {duration}s: {target_path} ({file_size} bytes)")
        return {
            "status": "success",
            "success": True,
            "title": doc_title,
            "filename": filename,
            "pdf_url": public_url,
            "local_path": target_path,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1),
            "duration_seconds": duration
        }
    except Exception as e:
        print(f"[Erro PDF Export]: {e}")
        return {
            "status": "error",
            "success": False,
            "error": str(e),
            "title": doc_title,
            "filename": None,
            "pdf_url": None,
            "size_bytes": 0
        }


def is_pdf_export_request(user_prompt: str) -> bool:
    """
    Detecta se o usuário está solicitando a conversão ou exportação da mensagem/resposta anterior em PDF.
    Exemplos:
      - "Gere um PDF desse resumo"
      - "Converta o texto anterior em PDF"
      - "Gere um pdf com as formatações do markdown"
      - "Baixar o texto em PDF"
      - "Exportar para PDF"
    """
    if not user_prompt:
        return False
    p_low = user_prompt.lower().strip()
    if "pdf" not in p_low:
        return False

    # Evita falsos positivos como dúvidas sobre leitores ou formato
    if re.search(r"o\s+que\s+[ée]\s+(?:um\s+)?pdf", p_low):
        return False
    if re.search(r"como\s+(?:abrir|ler|visualizar)\s+(?:um\s+)?pdf", p_low):
        return False

    # Padrões explícitos de referência ou conversão
    export_patterns = [
        r"(?:gerar?|gere|gera)\s+(?:um\s+)?pdf\s+(?:desse|deste|do|da|com)",
        r"(?:converter?|converta|converte)\s+(?:o\s+)?(?:texto|resumo|conte[uú]do|artigo|post|resposta)?\s*(?:anterior\s+)?(?:em|para)?\s*pdf",
        r"(?:exportar?|exporte|exporta)\s+(?:em|para|o)?\s*pdf",
        r"(?:baixar?|baixe|baixa|download)\s+(?:o\s+)?(?:texto|resumo|arquivo|documento)?\s*(?:em|no)?\s*pdf",
        r"(?:salvar?|salve|salva)\s+(?:em|como|no)?\s*pdf",
        r"pdf\s+(?:com\s+as\s+formata[cç][oõ]es|formatado|do\s+resumo|do\s+texto|da\s+resposta)",
        r"formato\s+pdf"
    ]
    if any(re.search(pat, p_low) for pat in export_patterns):
        return True

    # Comandos curtos com verbos de ação (< 120 caracteres)
    if len(p_low) <= 120:
        verbs = ["gerar", "gere", "converter", "converta", "exportar", "exporte", "baixar", "baixe", "salvar", "salve", "criar", "crie"]
        if any(v in p_low for v in verbs):
            return True

    return False


def prompt_wants_pdf_attachment(user_prompt: str) -> bool:
    """
    Detecta se um pedido novo/geral pede para acompanhar um arquivo PDF gerado.
    Exemplo: "Faça um resumo sobre a Companhia de Jesus e gere em PDF"
    """
    if not user_prompt:
        return False
    p_low = user_prompt.lower().strip()
    if "pdf" not in p_low:
        return False

    return any(re.search(pat, p_low) for pat in [
        r"(?:e\s+)?(?:gere|gerar|salve|salvar|exporte|exportar|entregue|forne[çc]a|disponibilize)\s+(?:tamb[ée]m\s+)?(?:em|como|um)?\s*pdf",
        r"(?:em|no)\s+formato\s+pdf",
        r"pdf\s+(?:para\s+download|formatado|para\s+apresenta[cç][aã]o|anexo)"
    ])


def format_pdf_markdown_card(pdf_result: Dict[str, Any], intro_text: Optional[str] = None) -> str:
    """Gera um card em Markdown profissional e elegante com link direto para download do PDF."""
    if not pdf_result or pdf_result.get("status") != "success":
        err_msg = pdf_result.get("error", "Erro desconhecido") if pdf_result else "Falha no processador"
        return f"> ⚠️ *Não foi possível gerar o PDF: {err_msg}*"

    title = pdf_result.get("title", "Documento Formatado")
    size_kb = pdf_result.get("size_kb", 0)
    pdf_url = pdf_result.get("pdf_url", "")
    filename = pdf_result.get("filename", "documento.pdf")
    duration = pdf_result.get("duration_seconds", 0)
    current_time = get_now_local().strftime("%d/%m/%Y às %H:%M")

    lines = []
    if intro_text:
        lines.append(f"{intro_text}\n")
    lines.append("### 📄 Documento PDF Gerado com Sucesso!")
    lines.append("O texto e as formatações em Markdown foram convertidos em um documento PDF profissional com tipografia executiva, cabeçalhos, rodapés e paginação.\n")
    lines.append(f"- **Título do Documento:** `{title}`")
    lines.append(f"- **Tamanho do Arquivo:** `{size_kb} KB`")
    lines.append(f"- **Tempo de Processamento:** `{duration}s`")
    lines.append(f"- **Data de Emissão:** `{current_time}`\n")
    lines.append(f"[📥 **Baixar Documento em PDF ({filename})**]({pdf_url})\n")
    lines.append("_Documento diagramado e pronto para apresentação, impressão ou compartilhamento._")

    return "\n".join(lines)

