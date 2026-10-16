import re
from typing import Tuple, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from config import (
    LITELLM_BASE_URL,
    LITELLM_API_KEY,
    OLLAMA_BASE_URL,
    message_has_image,
    history_has_images,
    is_context_dependent_prompt,
    get_conversation_state
)
from prompts import CLASSIFICATION_PROMPT_TEMPLATE
from dictionary_service import match_universal_dictionary
from user_lexicon_service import match_user_lexicon

# Instancia o LLM de classificação primário e fallbacks dinâmicos com cache
_classifier_chains: Dict[str, Any] = {}

def get_classifier_chain(model_name: str):
    """Retorna chain de classificação para o modelo especificado (com cache)."""
    if model_name not in _classifier_chains:
        llm = ChatOpenAI(
            base_url=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            model=model_name,
            temperature=0.0,
            max_tokens=350,
            timeout=8.0
        )
        _classifier_chains[model_name] = CLASSIFICATION_PROMPT_TEMPLATE | llm | StrOutputParser()
    return _classifier_chains[model_name]

classifier_chain = get_classifier_chain("Classifier_Fast")
classifier_chain_fallback = get_classifier_chain("Combo_Free_Chat")


def quick_rule_classifier(prompt: str) -> Optional[str]:
    """Verificação ultrarrápida por regras com Regex (0ms)."""
    p = prompt.lower()
    
    # 0. Conversão e Exportação para PDF (Prioridade sobre RAG)
    try:
        from pdf_export_service import is_pdf_export_request
        if is_pdf_export_request(prompt):
            return "PDF_EXPORT"
    except Exception:
        pass

    # RAG / Documentos Internos / Memória Temporária de Livros/Textos
    rag_pattern = (
        r"(segundo\s+(meus|os|esse|este)\s+documentos?"
        r"|no\s+(manual|documento|arquivo|pdf|relat[oó]rio|livro|romance|texto(\s+(enviado|anexo|anterior|acima))?)"
        r"|(nesse|neste|do|deste|desse)\s+(texto|documento|arquivo|livro|cap[ií]tulo)"
        r"|de\s+acordo\s+com\s+(o\s+manual|o\s+arquivo|os\s+documentos|o\s+livro|o\s+texto)"
        r"|na\s+base\s+de\s+conhecimento|nos\s+meus\s+arquivos|na\s+mem[oó]ria"
        r"|personagens\s+do\s+livro|resumo\s+d[oe]\s+(livro|texto|documento|arquivo|cap[ií]tulo)"
        r"|analise\s+(esse|este|o)\s+(texto|documento|arquivo)"
        r"|sobre\s+o\s+texto|no\s+texto"
        r"|que\s+(te\s+)?(mandei|enviei|anexei|subi)"
        r"|attached\s+document\(s\)"
        r"|texto\s+(enviado|anexo|acima)"
        r"|\b(o\s+que\s+(eu\s+)?posso\s+fazer\s+n(esse|este)\s+sistema"
        r"|o\s+que\s+posso\s+fazer\s+aqui"
        r"|quais\s+(as\s+|s[aã]o\s+as\s+)?funcionalidades\s+(d[oe]|deste|desse|neste|nesse)\s+sistema"
        r"|funcionalidades\s+dispon[ií]veis\s+(no|neste|nesse)\s+(sistema|chat|librechat)"
        r"|o\s+que\s+(o\s+)?(sistema|chat|librechat|orquestrador)\s+(faz|sabe\s+fazer|pode\s+fazer|oferece)"
        r"|como\s+(funciona|usar)\s+(o\s+)?(sistema|librechat|orquestrador)"
        r"|qual\s+(o\s+)?endpoint\s+.*(opencode|cline|continue|cursor|vscode)"
        r"|conectar\s+(o\s+)?opencode\s+ao\s+sistema"
        r"|configura[cç][aã]o\s+d[oe]\s+(opencode|cline|continue|cursor)\s+no\s+sistema"
        r"|como\s+(fa[çc]o\s+a\s+|fazer\s+a\s+)?configura[cç][aã]o\s+d[oe]\s+opencode"
        r"|como\s+configurar\s+o\s+opencode)\b)"
    )
    if re.search(rag_pattern, p):
        return "DOCUMENT_RAG"

    # Clima / Tempo
    weather_pattern = r"(previs[aã]o\s+d[oe]\s+tempo|previs[aã]o\s+(para|de)|clima\s+em|temperatura\s+em|vai\s+chover|tempo\s+hoje|tempo\s+para\s+hoje|quantos\s+graus)"
    if re.search(weather_pattern, p):
        return "WEATHER_INFO"

    # 1. Indicadores Financeiros / Cotações / Séries Estatísticas / CNPJ / CEP / FIPE -> Obrigatoriamente REALTIME_SEARCH
    realtime_data_pattern = (
        r"\b(cota[cç][aã]o|pre[cç]o|valor)\s+(d[oe]|d[oa]|dos?|das?)?\s*(d[oó]lar|euro|bitcoin|btc|moeda|a[cç][oõ]es|petr4|vale3|bolsa|ibovespa)\b"
        r"|\b(taxa\s+selic|selic|ipca|infla[cç][aã]o|cdi|igp-?m|rendimento\s+da\s+poupan[cç]a)\b"
        r"|\b(d[oó]lar\s+(comercial|hoje|em\s+reais)|euro\s+hoje|bitcoin\s+hoje)\b"
        r"|\b(cnpj|consult(e|ar)?\s+(o\s+)?cnpj|razao\s+social|raz[aã]o\s+social)\b"
        r"|\b(\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2})\b"
        r"|\b(cep\s+\d{5}-?\d{3}|consult(e|ar)?\s+(o\s+)?cep|qual\s+(é|e)\s+o\s+endereço\s+do\s+cep)\b"
        r"|\b(tabela\s+fipe|pre[cç]o\s+fipe|valor\s+pela\s+fipe|c[oó]digo\s+fipe)\b"
    )
    if re.search(realtime_data_pattern, p):
        return "REALTIME_SEARCH"

    # 2. Busca em tempo real / Notícias / Eleições / Apuração / Esportes / Lugares e Restaurantes (Fast-Pass 0ms)
    realtime_pattern = (
        r"(not[ií]cias\s+(sobre|de|do|da|hoje)|[uú]ltimas\s+not[ií]cias|cota[cç][aã]o\s+d[oe]"
        r"|resultado\s+d(o\s+jogo|as?\s+elei[cç][oõ]es?|as?\s+apura[cç][oõ]es?)"
        r"|elei[cç][aã]o|elei[cç][oõ]es|apuração\s+d[aeo]|quem\s+ganhou\s+(a\s+elei[cç][aã]o|o\s+jogo)"
        r"|quem\s+foi\s+eleito|governador\s+eleito|senador\s+eleito|prefeito\s+eleito"
        r"|hoje\s+no\s+brasil|pre[cç]o\s+do\s+d[oó]lar|d[oó]lar\s+hoje"
        r"|\b(restaurantes?|pizzarias?|hamburguerias?|bares|onde\s+comer|onde\s+fica|o\s+que\s+fazer\s+em)\s+(em|no|na|de|do|da)\b"
        r"|\b(em\s+2026|de\s+2026|at[eé]\s+o\s+momento|atualmente|neste\s+ano|neste\s+m[eê]s|este\s+ano|ano\s+atual|ranking\s+atual|campeonato\s+de\s+2026|ranking\s+de\s+2026|melhores\s+nadadores|tabela\s+do\s+brasileir[aã]o|jogos\s+de\s+hoje)\b)"
    )
    if re.search(realtime_pattern, p):
        return "REALTIME_SEARCH"

    # 3. Exportador de Tabelas / Planilhas / CSV / Excel / Gráficos para dados sintéticos/conceituais (Fast-Pass 0ms)
    data_export_pattern = (
        r"\b(export(ar|e|ador)?\s+(para|em|pro|o|a)?\s*(excel|csv|planilha|tabela))"
        r"|\b(ger(ar|e|ador)?\s+(um[a]?\s+)?(planilha|csv|excel|\.xlsx|\.csv|gr[aá]fico))"
        r"|\b(baix(ar|e)?\s+(em|como|o|a)?\s*(excel|csv|planilha|\.xlsx))"
        r"|\b(monte|crie|fa[çc]a|construa)\s+(um[a]?\s+)?(planilha|tabela)\s+.*(gr[aá]fico|excel|csv)"
        r"|\b(plote|plotar|gere|gerar|crie|criar|fa[çc]a)\s+(um[a]?\s+)?gr[aá]fico\s+(de\s+barras?|de\s+linhas?|de\s+pizza|de\s+dispers[aã]o|estat[ií]stico|dos?\s+dados?|comparativo)"
        r"|\b(gr[aá]fico\s+e\s+(tabela|planilha|excel|csv))\b"
        r"|\b(tabela\s+e\s+(gr[aá]fico|excel|csv))\b"
        r"|\b(tabela\s+de\s+.*(gr[aá]fico|excel|csv))\b"
        r"|\b(dados\s+em\s+(excel|csv))\b"
    )
    if re.search(data_export_pattern, p):
        return "DATA_EXPORT_CHART"

    # Programação / Código / Scripts / Infraestrutura / Servidores / Linux (Fast-Pass 0ms)
    coding_pattern = (
        r"(escreva\s+um(a)?\s+(função|script|código|classe|programa|query|sql|regex)|crie\s+um(a)?\s+(script|função|código|api)|em\s+(python|javascript|typescript|bash|sql|html|css|php|java|c\+\+|rust|golang)|como\s+fazer\s+em\s+python|como\s+programar|depurar|debug|dockerfile|docker-compose|código\s+para)"
        r"|\b(servidor\s+smb|servidor\s+de\s+arquivos|configurar\s+smb|compartilhamento\s+smb|smbclient|samba|powershell|set-smbclientconfiguration|systemctl|iptables|chmod|chown)\b"
        r"|\b(como\s+configurar\s+acesso\s+a\s+servidor|configurar\s+acesso\s+ao?\s+servidor|conectar\s+ao?\s+servidor\s+smb)\b"
    )
    if re.search(coding_pattern, p):
        # Se também envolver workspace de forma explícita, prioriza MCP_FILESYSTEM
        if "workspace" in p:
            return "MCP_FILESYSTEM"
        return "CODING"

    # MCP Filesystem / Workspace Sandbox (Fast-Pass 0ms)
    filesystem_pattern = (
        r"\b(workspace|no\s+workspace|do\s+workspace|na\s+sandbox|sandbox\s+de\s+arquivos)\b"
        r"|\b(listar|ver|quais)\s+(os\s+)?(arquivos|pastas|diret[oó]rios)\s+(do|no|na)\s+(workspace|sandbox)\b"
        r"|\b(crie|criar|gere|gerar|salve|salvar|grave|gravar|escreva|escrever)\s+(um\s+)?(arquivo|script|c[oó]digo|pasta)\s+.*(no|para\s+o)\s+workspace\b"
        r"|\b(leia|ler|abra|abrir|mostre|mostrar|conte[uú]do\s+d[oe])\s+(o\s+)?(arquivo|script)\s+.*(do|no)\s+workspace\b"
        r"|\b(delete|deletar|remova|remover|apague|apagar)\s+(o\s+)?(arquivo|pasta)\s+.*(do|no)\s+workspace\b"
        r"|\b(busque|buscar|pesquise|pesquisar|procure|procurar)\s+.*(no|do)\s+workspace\b"
    )
    if re.search(filesystem_pattern, p):
        return "MCP_FILESYSTEM"

    # MCP Fetch / Extração e Leitura de Páginas Web via URL (Fast-Pass 0ms)
    fetch_pattern = (
        r"\b(mcp\s*fetch|fetch)\b"
        r"|https?://[^\s]+\s+.*(leia|ler|acesse|acessar|extraia|extrair|resuma|resumir|o que diz|conte[uú]do|analise|analisar)"
        r"|(leia|ler|acesse|acessar|extraia|extrair|resuma|resumir|mostre o conte[uú]do de|o que diz no?|analise|analisar|visite|visitar)\s+.*https?://[^\s]+"
        r"|^(acesse|leia|abra|veja|extraia|resuma)\s+https?://[^\s]+"
        r"|^https?://[^\s]+$"
    )
    if re.search(fetch_pattern, p):
        return "MCP_FETCH"

    # MCP Git / Controle de Versão de Código (Fast-Pass 0ms)
    git_pattern = (
        r"\b(mcp\s*git|git\s+status|git\s+diff|git\s+log|git\s+commit|git\s+branch|git\s+branches|git\s+checkout|git\s+init|git\s+show|git\s+add)\b"
        r"|\b(status|diff|hist[oó]rico|commits?|branches?)\s+(do\s+git|no\s+git)\b"
        r"|\b(commite|commitar|fazer\s+commit|fa[cç]a\s+um\s+commit|criar\s+um\s+commit)\b"
    )
    if re.search(git_pattern, p):
        return "MCP_GIT"

    # MCP Brave Search / Busca Web Independente (Fast-Pass 0ms)
    brave_pattern = (
        r"\b(mcp\s*brave|brave\s+search|busque\s+no\s+brave|pesquise\s+no\s+brave|procure\s+no\s+brave|no\s+brave\s+search)\b"
    )
    if re.search(brave_pattern, p):
        return "MCP_BRAVE_SEARCH"

    # MCP Database / Consultas SQL e Bancos de Dados (Fast-Pass 0ms)
    db_pattern = (
        r"\b(mcp\s*database|mcp\s*db|sqlite3|duckdb)\b"
        r"|\b(listar|ver|quais|mostre|liste)\s+(as\s+)?(tabelas|views)\s+(do\s+banco|no\s+banco|de\s+dados)\b"
        r"|\b(descreva|schema|estrutura\s+da\s+tabela)\s+[a-zA-Z0-9_]+\b"
        r"|\b(execute|executar|rode|rodar|fazer|fa[çc]a)\s+(um[a]?\s+)?(query|consulta\s+sql|select|script\s+sql)\b"
        r"|\b(SELECT\s+.*?\s+FROM\s+[a-zA-Z0-9_]+)\b"
    )
    if re.search(db_pattern, p, re.IGNORECASE):
        return "MCP_DATABASE"

    # MCP Code Runner / Execução Segura de Scripts e Terminal (Fast-Pass 0ms)
    code_runner_pattern = (
        r"\b(mcp\s*code\s*runner|mcp\s*runner|code\s*runner)\b"
        r"|\b(execute|executar|rode|rodar)\s+(este|esse|o)?\s*(c[oó]digo|script|comando)\s+(python|bash|sh|no\s+terminal|no\s+bash)\b"
        r"|\b(execute|executar|rode|rodar)\s+(o\s+)?script\s+[a-zA-Z0-9_\-./]+\.(py|sh|js)\b"
    )
    if re.search(code_runner_pattern, p, re.IGNORECASE):
        return "MCP_CODE_RUNNER"

    # Raciocínio Pesado / Matemática Avançada / Xadrez / Física Teórica (Fast-Pass 0ms)
    heavy_reasoning_pattern = r"(equa[cç][aã]o\s+diferencial|c[aá]lculo\s+(avan[cç]ado|diferencial|integral)|derivada|integral|matriz(es)?|autovalor|autovetor|transformada\s+de\s+(laplace|fourier)|f[ií]sica\s+te[oó]rica|mec[aâ]nica\s+qu[aâ]ntica|relatividade\s+geral|partida\s+de\s+xadrez|jogada\s+de\s+xadrez|enigma\s+matem[aá]tico)"
    if re.search(heavy_reasoning_pattern, p):
        return "HEAVY_REASONING"

    # Raciocínio Lógico / Problemas Matemáticos (Fast-Pass 0ms)
    thinking_pattern = r"(quantos\s+(gatos|ratos|homens|dias|minutos|horas)|se\s+\d+\s+(gatos|pessoas|trabalhadores)|charada|enigma|racioc[ií]nio\s+l[oó]gico|problema\s+de\s+l[oó]gica|resolva\s+a\s+equa[cç][aã]o)"
    if re.search(thinking_pattern, p):
        return "LOCAL_THINKING"

    # Vídeos Curtos / Reels / TikTok / Shorts / Vídeo Vertical (Fast-Pass 0ms)
    short_video_pattern = (
        r"\b(crie|criar|gere|gerar|fa[çc]a|fazer|produza|produzir|elabore|monte)\s+(um[a]?\s+)?(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok|mini\s+v[ií]deo)\b"
        r"|\b(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok)\s+(sobre|de|do|da|com|para)\b"
        r"|\b(roteiro\s+e\s+v[ií]deo|gerador\s+de\s+v[ií]deo|criar\s+um\s+v[ií]deo|gere\s+um\s+v[ií]deo)\b"
    )
    if re.search(short_video_pattern, p):
        return "SHORT_VIDEO"

    # Redes Sociais / Posts / Anúncios / Leads / Campanhas (Fast-Pass 0ms)
    social_networks = r"(?:\b(instagram|insta|linkedin|facebook|face|twitter|reddit|threads|tiktok|redes?\s+socia(is|l))\b|(?:\b(?:rede\s+x|no\s+x|do\s+x|twitter\s*/\s*x)\b))"
    social_content = r"(post\w*|postagem\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|lead\w*|legenda\w*|carrossel\w*|thread\w*|copy\w*|story|stories|feed)"
    social_preps = r"(para|pro|pra|no|na|de|do|da|em)(\s+(o|a|os|as|um|uma))?"

    social_pattern = (
        rf"{social_content}\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(criar|crie|fa[çc]a|gere|gerar|elabore|escreva|monte|produza)\s+(um[a]?\s+)?({social_content}\s+)?{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|postagem\w*|publica\w*|an[uú]nc\w*)\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|postagem\w*|an[uú]nc\w*|campanha\w*)\s+(de|para)\s+(capta[cç]\w+|leads?|marketing|vendas|engajamento)\b"
        rf"|\b(roteiro|ideia|estrat[eé]gia)\s+de\s+(post\w*|postagem\w*|conte[uú]do)\s+{social_preps}\s+{social_networks}\b"
        rf"|{social_networks}\b.*(com\s+a\s+legenda|utilizando\s+a\s+imagem|use\s+a\s+imagem|hashtag)"
        rf"|\b(criar|crie|fa[çc]a|gere|gerar|elabore|escreva|monte|produza)\s+(um[a]?\s+)?(postagem|post|publica[cç][aã]o)\b"
    )
    if re.search(social_pattern, p):
        return "SOCIAL_MEDIA_POST"

    # Geração de Imagem / Arte Visual (Fast-Pass 0ms)
    image_gen_pattern = (
        r"(gere|gerar|crie|criar|desenhe|desenhar|fa[çc]a|produza)\s+(uma?\s+)?(imagem|foto|desenho|ilustra[çc][aã]o|pintura|wallpaper|arte)\b"
        r"|(imagem|foto|desenho|ilustra[çc][aã]o|pintura)\s+d[oe]\s+"
        r"|^(gere|crie|desenhe)\s+.*(imagem|desenho|foto)"
        r"|\b(generate|create|draw)\s+(an?\s+)?(image|photo|picture|illustration|art)\b"
    )
    if re.search(image_gen_pattern, p) and not re.search(r"\b(post\w*|postagem\w*|publica[cç]\w*|carrossel|legenda)\b", p):
        return "IMAGE_GENERATION"

    # Conversa casual / Saudações / Cortesia / Chat informal (Fast-Pass 0ms)
    casual_pattern = (
        r"^(oi|ol[aá]|bom\s+dia|boa\s+tarde|boa\s+noite|e\s+a[ií]|fala|opa|hello|hi|hey)\b"
        r"([\s,!.?]+(tudo\s+(bem|bom|certo|em\s+cima)(\s+com\s+(voc[eê]|vc|ti))?|como\s+(vai|voc[eê]\s+est[aá]|est[aá]|c[eê]\s+t[aá])|beleza|tranquilo|amigo|assistente|pessoal|a[ií])?)*[\s,!.?]*$"
        r"|^(tudo\s+(bem|bom|certo|em\s+cima)(\s+com\s+(voc[eê]|vc|ti))?|como\s+(vai|voc[eê]\s+est[aá]|est[aá]|c[eê]\s+t[aá])|como\s+vai\s+voc[eê]|como\s+est[aá]s|beleza\??|tranquilo\??|obrigad[oa]|muito\s+obrigad[oa]|valeu|falou|at[eé]\s+(logo|mais|amanh[aã])|tchau)[\s,!.?]*$"
    )
    if re.search(casual_pattern, p.strip(), re.IGNORECASE):
        return "CASUAL_CHAT"

    return None


async def classify_intent_v1(
    user_prompt: str,
    messages: list,
    last_user_msg: dict,
    conversation_id: str = None,
    user_id: str = None
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Motor Classificador V1 (Legado): Dicionário Universal + Fast-pass Regex + LLM Livre.
    """
    has_image_now = message_has_image(last_user_msg)
    has_image_in_history = history_has_images(messages)

    is_title_request = "Provide a concise, 5-word-or-less title" in user_prompt
    routing_method = "llm_classifier"
    raw_classifier_output = None

    edit_keywords_pattern = (
        r"\b(alter[ea]|alterar|altere|mud[ea]|mudar|mude|troqu?e|trocar|troca|"
        r"edit[ea]|editar|edite|remov[ea]|remover|adicione?|adicionar|adiciona|"
        r"coloqu?e|colocar|coloca|substitu[ai]|substituir|substitua|modifiqu?e|modificar|modifica|"
        r"transform[ea]|transformar|transforme|apagu?e|apagar|apaga|retir[ea]|retirar|retire|"
        r"pint[ea]|pintar|pinte|retoc[ea]|retocar|bot[ea]|botar|bota|p[oõ]e|ponha|pôr|"
        r"insir[ea]|inserir|aplic[ea]|aplicar|cort[ea]|cortar|corte|recort[ea]|recortar|"
        r"clarei[ea]|clarear|escure[çc][ea]|escurecer|ajust[ea]|ajustar|deix[ea]|deixar|"
        r"fundo|background|remove|change|replace|edit|modify)\b"
    )
    image_gen_keywords_pattern = (
        r"(gere|gerar|crie|criar|desenhe|desenhar|fa[çc]a|produza)\s+(uma?\s+)?(imagem|foto|desenho|ilustra[çc][aã]o|pintura|wallpaper|arte)\b"
        r"|(imagem|foto|desenho|ilustra[çc][aã]o|pintura)\s+d[oe]\s+"
        r"|^(gere|crie|desenhe)\s+.*(imagem|desenho|foto)"
        r"|\b(generate|create|draw)\s+(an?\s+)?(image|photo|picture|illustration|art)\b"
    )

    # 1. Requisição de Título interna do LibreChat
    if is_title_request:
        routing_method = "title_rule"
        category = "VISION" if has_image_in_history else "CASUAL_CHAT"
        print(f"[Orquestrador] Requisição de Título detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 1.1. Consulta ao Dicionário Universal de Termos e Intenções (0ms)
    dict_match = match_universal_dictionary(
        prompt=user_prompt,
        has_image_now=has_image_now,
        has_image_in_history=has_image_in_history
    )
    if dict_match:
        category, routing_method = dict_match
        print(f"[Orquestrador Dicionário] Correspondência encontrada -> Rota: {category} ({routing_method})")
        return category, routing_method, None, None

    # 1.2. Consulta ao Dicionário Personalizado do Usuário (Aprendizado Contínuo 0ms)
    if user_id:
        user_match = match_user_lexicon(user_id=user_id, prompt=user_prompt)
        if user_match:
            category, routing_method = user_match
            print(f"[Orquestrador Lexicon Usuário] Padrão aprendido detectado para '{user_id}' -> Rota: {category} ({routing_method})")
            return category, routing_method, None, None

    # 1.1. Vídeos Curtos / Reels / TikTok / Shorts / Vídeo Vertical
    short_video_pattern = (
        r"\b(crie|criar|gere|gerar|fa[çc]a|fazer|produza|produzir|elabore|monte)\s+(um[a]?\s+)?(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok|mini\s+v[ií]deo)\b"
        r"|\b(v[ií]deo\s+curto|v[ií]deo\s+vertical|reels?|shorts?|tiktok)\s+(sobre|de|do|da|com|para)\b"
        r"|\b(roteiro\s+e\s+v[ií]deo|gerador\s+de\s+v[ií]deo|criar\s+um\s+v[ií]deo|gere\s+um\s+v[ií]deo)\b"
    )
    if re.search(short_video_pattern, user_prompt.lower()):
        routing_method = "short_video_rule"
        category = "SHORT_VIDEO"
        print(f"[Orquestrador] Vídeo Curto detectado -> Rota: {category}")
        return category, routing_method, None, None

    # 2. Post para Redes Sociais e Leads (precede edição e geração genérica de imagem)
    social_networks = r"(?:\b(instagram|insta|linkedin|facebook|face|twitter|reddit|threads|tiktok|redes?\s+socia(is|l))\b|(?:\b(?:rede\s+x|no\s+x|do\s+x|twitter\s*/\s*x)\b))"
    social_content = r"(post\w*|postg\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|lead\w*|legenda\w*|carrossel\w*|thread\w*|copy\w*|story|stories|feed)"
    social_preps = r"(para|pro|pra|no|na|de|do|da|em)(\s+(o|a|os|as|um|uma))?"

    social_keywords_pattern = (
        rf"{social_content}\s+.*?{social_networks}\b"
        rf"|\b(criar|crie|fa[çc]a|gere|gerar|elabore|escreva|monte|produza)\s+.*?{social_networks}\b"
        rf"|\b(post\w*|postg\w*|publica\w*|an[uú]nc\w*)\s+{social_preps}\s+{social_networks}\b"
        rf"|\b(post\w*|postg\w*|an[uú]nc\w*|campanha\w*)\s+(de|para)\s+(capta[cç]\w+|leads?|marketing|vendas|engajamento)\b"
        rf"|\b(roteiro|ideia|estrat[eé]gia)\s+de\s+(post\w*|conte[uú]do)\s+{social_preps}\s+{social_networks}\b"
        rf"|{social_networks}\b.*(com\s+a\s+legenda|utilizando\s+a\s+imagem|use\s+a\s+imagem|hashtag)"
    )
    # Com imagem anexa ou citando redes, "post/publicação/legenda" já basta
    social_with_image_pattern = r"\b(post\w*|postg\w*|publica[cç]\w*|legenda\w*|redes?\s+socia(is|l))\b"
    is_social_request = bool(re.search(social_keywords_pattern, user_prompt.lower())) or (
        (has_image_now or has_image_in_history or "imagem" in user_prompt.lower() or "foto" in user_prompt.lower()) and re.search(social_with_image_pattern, user_prompt.lower())
    ) or bool(re.search(social_networks, user_prompt.lower()) and re.search(r"\b(crie|criar|fa[çc]a|gere|gerar|escreva|monte|elabore)\b", user_prompt.lower()))

    if is_social_request:
        routing_method = "social_media_rule"
        category = "SOCIAL_MEDIA_POST"
        print(f"[Orquestrador] Post de Redes Sociais detectado -> Rota: {category}")
        return category, routing_method, None, None

    # 3. Edição de Imagem com anexo presente ou recente
    if (has_image_now or has_image_in_history) and re.search(edit_keywords_pattern, user_prompt.lower()):
        routing_method = "image_edit_attachment_rule"
        category = "IMAGE_EDIT"
        print(f"[Orquestrador] Edição de Imagem detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 4. Geração de Imagem Nova
    if re.search(image_gen_keywords_pattern, user_prompt.lower()):
        # Se for pedido de imagem mas citou explicitamente uma rede social, desvia para post de rede social
        if re.search(social_networks, user_prompt.lower()):
            routing_method = "social_media_network_fallback_rule"
            category = "SOCIAL_MEDIA_POST"
            print(f"[Orquestrador] Pedido de imagem com rede social citada -> Desviado para: {category}")
            return category, routing_method, None, None

        routing_method = "image_gen_rule"
        category = "IMAGE_GENERATION"
        print(f"[Orquestrador] Geração de Imagem detectada -> Rota: {category}")
        return category, routing_method, None, None

    # 4. Análise de Imagem (Visão pura)
    if has_image_now:
        routing_method = "vision_attachment_rule"
        category = "VISION"
        return category, routing_method, None, None

    # 5. Avaliação do Estado e Continuidade da Conversa
    session_state = get_conversation_state(conversation_id, messages)
    last_category = session_state.get("last_category")
    last_model = session_state.get("last_model")
    is_followup = is_context_dependent_prompt(user_prompt, messages)

    # Se for resposta ao menu Ask-Mode social, preserva SOCIAL_MEDIA_POST diretamente
    from ask_mode_service import is_ask_mode_reply
    is_reply, sel_mode, combined_prompt, orig_topic = is_ask_mode_reply(user_prompt, messages, session_state)
    if is_reply:
        print(f"[Orquestrador Ask-Mode Reply] Usuário respondeu ao menu Ask-Mode: opção '{sel_mode}' para tema '{orig_topic}'. Rota: SOCIAL_MEDIA_POST")
        return "SOCIAL_MEDIA_POST", "social_ask_mode_reply_rule", None, None

    # Se a categoria anterior era SOCIAL_MEDIA_POST, mas a mensagem atual NÃO cita nenhuma rede social,
    # postagem, imagem ou legenda, NUNCA herdar SOCIAL_MEDIA_POST!
    if last_category == "SOCIAL_MEDIA_POST":
        social_check_pattern = r"\b(post\w*|postg\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|legenda\w*|carrossel\w*|copy\w*|story|stories|feed|instagram|insta|linkedin|facebook|twitter|tiktok|hashtag|foto|imagem)\b"
        if not re.search(social_check_pattern, user_prompt.lower()):
            is_followup = False
            last_category = None

    # Se for pergunta de acompanhamento / continuação (ex: 'E para Senador?', 'E amanhã?', 'E em Santos?', 'Faça uma dissertação sobre o tema')
    if is_followup and last_category:
        # Verifica se o usuário não pediu explicitamente uma mudança de domínio via regra rígida
        explicit_rule = quick_rule_classifier(user_prompt)
        if not explicit_rule and last_category in [
            "FACTUAL_GENERAL",
            "REALTIME_SEARCH",
            "WEATHER_INFO",
            "DOCUMENT_RAG",
            "CODING",
            "HEAVY_REASONING",
            "LOCAL_THINKING",
            "SOCIAL_MEDIA_POST",
            "DATA_EXPORT_CHART"
        ]:
            # Se for pedido de planilha ou gráfico, respeita
            if any(k in user_prompt.lower() for k in ["gráfico", "grafico", "planilha", "excel", "csv"]):
                category = "DATA_EXPORT_CHART"
            else:
                category = last_category
            routing_method = "contextual_continuity_rule"
            print(f"[Orquestrador Contexto] Pergunta de acompanhamento detectada ('{user_prompt}'). Mantendo rota anterior: {category} (Modelo anterior: {last_model})")
            return category, routing_method, None, None

    # 6. Fast-Pass por Regras com Regex (0ms)
    rule_category = quick_rule_classifier(user_prompt)
    if rule_category:
        routing_method = "fast_pass_rule"
        category = rule_category
        print(f"[Orquestrador] Classificação por Regra Rápida -> Rota: {category}")
        return category, routing_method, None, None

    # 7. Classificador LLM Semântico com Suporte a Contexto Anterior, Guardrail e Query Rewriting
    routing_method = "llm_classifier"
    category = last_category if (is_followup and last_category) else "FACTUAL_GENERAL"
    rewritten_prompt = None

    # Extrai trecho da última resposta do assistente para contexto
    last_assistant_snippet = ""
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            c = m.get("content", "")
            if isinstance(c, str):
                last_assistant_snippet = c[:300]
            elif isinstance(c, list):
                t = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                last_assistant_snippet = " ".join(t)[:300]
            break

    context_str = f"Última resposta do assistente: '{last_assistant_snippet}'" if last_assistant_snippet else "Início da conversa."
    prompt_inputs = {
        "context": context_str,
        "last_category": last_category or "Nenhuma (primeira mensagem)",
        "last_model": last_model or "Nenhum",
        "message": user_prompt
    }

    # Obtém modelos configurados no Dashboard (Padrão: Classifier_Fast -> Combo_Free_Chat)
    try:
        from admin_config_service import get_category_models
        configured_models = get_category_models("SEMANTIC_CLASSIFIER") or ["Classifier_Fast", "Combo_Free_Chat"]
    except Exception:
        configured_models = ["Classifier_Fast", "Combo_Free_Chat"]

    primary_model = configured_models[0] if len(configured_models) > 0 else "Classifier_Fast"
    fallback_model = configured_models[1] if len(configured_models) > 1 else "Combo_Free_Chat"

    raw_classification = None
    try:
        pri_chain = get_classifier_chain(primary_model)
        raw_classification = await pri_chain.ainvoke(prompt_inputs)
    except Exception as llm_err:
        print(f"[Aviso] Classificador primário ({primary_model}) falhou: {llm_err}. Tentando fallback ({fallback_model})...")
        try:
            fb_chain = get_classifier_chain(fallback_model)
            raw_classification = await fb_chain.ainvoke(prompt_inputs)
        except Exception as fb_err:
            print(f"[Aviso] Classificador secundário ({fallback_model}) também falhou: {fb_err}.")

    llm_followup_detected = False
    if raw_classification:
        raw_classifier_output = raw_classification
        clean_raw = re.sub(r"(?s)<think>.*?</think>", "", raw_classification)

        # 1. Parse de IS_FOLLOWUP
        if re.search(r"IS_FOLLOWUP:\s*TRUE", clean_raw, re.IGNORECASE):
            llm_followup_detected = True

        # 2. Parse de REWRITTEN_PROMPT
        rw_match = re.search(r"REWRITTEN_PROMPT:\s*(.+)", clean_raw, re.IGNORECASE)
        if rw_match:
            candidate_rw = rw_match.group(1).strip()
            # Valida se a reescrita é válida e não repetiu a chave CATEGORY
            if candidate_rw and "CATEGORY:" not in candidate_rw and len(candidate_rw) > 3:
                rewritten_prompt = candidate_rw

        # 3. Parse de CATEGORY
        cat_match = re.search(r"CATEGORY:\s*([A-Z_]+)", clean_raw)
        if cat_match:
            category = cat_match.group(1).strip()
        else:
            cat_matches = re.findall(
                r"\b(DOCUMENT_RAG|WEATHER_INFO|REALTIME_SEARCH|CASUAL_CHAT|CODING|LOCAL_THINKING|HEAVY_REASONING|FACTUAL_GENERAL|VISION|IMAGE_GENERATION|IMAGE_EDIT|SOCIAL_MEDIA_POST|DATA_EXPORT_CHART|SHORT_VIDEO|MCP_FILESYSTEM|MCP_FETCH|MCP_GIT|MCP_BRAVE_SEARCH|MCP_DATABASE|MCP_CODE_RUNNER)\b",
                clean_raw.upper()
            )
            if cat_matches:
                category = cat_matches[-1]
            else:
                raw_upper = clean_raw.upper()
                if any(w in raw_upper for w in ["PROGRAMAÇÃO", "PROGRAMACAO", "CÓDIGO", "CODIGO", "SCRIPT", "SMB", "LINUX", "WINDOWS", "POWERSHELL", "SERVIDOR"]):
                    category = "CODING"
                elif any(w in raw_upper for w in ["SOCIAL_MEDIA", "POSTAGEM", "POST", "INSTAGRAM", "LINKEDIN", "REDE SOCIAL"]):
                    # Só confirma SOCIAL_MEDIA_POST se o prompt do usuário contiver termos de rede social
                    social_kw = r"\b(post\w*|publica\w*|an[uú]nc\w*|campanha\w*|legenda\w*|feed|stories|instagram|linkedin|facebook|twitter|tiktok)\b"
                    if re.search(social_kw, user_prompt.lower()):
                        category = "SOCIAL_MEDIA_POST"
                    else:
                        category = "CODING" if any(k in user_prompt.lower() for k in ["smb", "linux", "servidor", "windows", "script"]) else "FACTUAL_GENERAL"
                elif any(w in raw_upper for w in ["SAUDAÇÃO", "SAUDACAO", "CUMPRIMENTO", "CONVERSA INFORMAL", "CASUAL"]):
                    category = "CASUAL_CHAT"
                elif any(w in raw_upper for w in ["CLIMA", "TEMPO", "PREVISÃO", "METEOROLOGIA"]):
                    category = "WEATHER_INFO"
                elif any(w in raw_upper for w in ["NOTÍCIA", "NOTICIA", "ELEIÇÃO", "ELEICAO", "COTAÇÃO"]):
                    category = "REALTIME_SEARCH"

        # Guardrail anti-desvio falso para SOCIAL_MEDIA_POST
        if category == "SOCIAL_MEDIA_POST":
            social_guard = r"\b(post\w*|postg\w*|publica[cç]\w*|an[uú]nc\w*|campanha\w*|legenda\w*|carrossel\w*|copy\w*|story|stories|feed|instagram|insta|linkedin|facebook|twitter|tiktok|hashtag)\b"
            if not re.search(social_guard, user_prompt.lower()):
                tech_guard = r"\b(smb|samba|nfs|servidor|linux|windows|powershell|bash|terminal|código|script|python|sql|docker|comando)\b"
                category = "CODING" if re.search(tech_guard, user_prompt.lower()) else "FACTUAL_GENERAL"
                print(f"[Guardrail Classificador] Falso positivo de SOCIAL_MEDIA_POST sem termos sociais. Ajustado para: {category}")

        print(f"[Debug Classificador LLM] Retorno bruto: '{raw_classification}' -> Rota final: {category} | LLM Follow-up: {llm_followup_detected} | Reescrita: {rewritten_prompt}")
    else:
        # Se todos os classificadores falharam
        if is_followup and last_category and last_category != "SOCIAL_MEDIA_POST":
            category = last_category
            routing_method = "llm_error_context_fallback"
            print(f"[Aviso] Classificador LLM indisponível. Usando fallback contextual da categoria anterior: {category}")
        else:
            category = "CODING" if any(k in user_prompt.lower() for k in ["smb", "linux", "servidor", "windows", "powershell", "samba"]) else "FACTUAL_GENERAL"
            routing_method = "llm_error_default_fallback"
            print(f"[Aviso] Classificador LLM indisponível. Usando fallback padrão: {category}.")

    # Guardrail para pedidos de PDF: exportação ou geração em PDF NUNCA deve cair em DOCUMENT_RAG
    if category == "DOCUMENT_RAG":
        try:
            from pdf_export_service import is_pdf_export_request, prompt_wants_pdf_attachment
            if is_pdf_export_request(user_prompt):
                print("[Guardrail Classificador] Redirecionando de DOCUMENT_RAG para PDF_EXPORT.")
                category = "PDF_EXPORT"
            elif prompt_wants_pdf_attachment(user_prompt):
                category = last_category if (last_category and last_category != "DOCUMENT_RAG") else "FACTUAL_GENERAL"
                print(f"[Guardrail Classificador] Prompt pede PDF de conteúdo. Ajustando rota de DOCUMENT_RAG para {category}.")
        except Exception:
            pass

    # Guardrail adicional: Se o modelo identificou follow-up, mas a categoria anterior era FACTUAL_GENERAL e o usuário não pediu código/análise de arquivo
    if (is_followup or llm_followup_detected) and last_category == "FACTUAL_GENERAL" and category == "DOCUMENT_RAG":
        has_file_mention = any(k in user_prompt.lower() for k in ["arquivo interno", "documento interno", "manual técnico", "anexo que enviei", "texto que subi"])
        if not has_file_mention:
            print(f"[Guardrail Classificador] Bloqueando desvio para DOCUMENT_RAG em continuidade de FACTUAL_GENERAL.")
            category = "FACTUAL_GENERAL"

    # Proteção de Continuação Contextual: Se for instrução de acompanhamento sem menção a código ou infra, não enviar para CODING
    if category == "CODING" and (is_followup or llm_followup_detected):
        coding_explicit = re.search(r"(código|script|programa|python|sql|java|regex|função|function|dev|docker|smb|samba|nfs|linux|windows|powershell|servidor|terminal|bash|ssh|systemctl)\b", user_prompt.lower())
        if not coding_explicit:
            category = last_category if (last_category and last_category not in ["CODING", "SOCIAL_MEDIA_POST", "IMAGE_GENERATION", "IMAGE_EDIT"]) else "FACTUAL_GENERAL"
            print(f"[Orquestrador Contexto] Instrução '{user_prompt}' é continuação de conversa/dados, não programação. Rota ajustada para: {category}.")

    return category, routing_method, raw_classifier_output, rewritten_prompt


# =====================================================================
# Roteador / Despachante Unificado com Suporte a Feature Flag & Shadow Mode
# =====================================================================
async def classify_intent(
    user_prompt: str,
    messages: list,
    last_user_msg: dict,
    conversation_id: str = None,
    user_id: str = None
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Despachante Dinâmico de Roteamento de Intenções:
    - V2_SEMANTIC_VECTOR: Roteamento Semântico Vetorial + JSON Schema Rígido.
    - SHADOW_MODE: Motor V2 como padrão de resposta oficial e V1 em paralelo para comparação em log.
    - V1_LEGACY: Motor Legado (Dicionário Universal + Fast-pass Regex + LLM Livre).
    """
    try:
        from admin_config_service import get_classifier_engine
        engine = get_classifier_engine()
    except Exception:
        engine = "SHADOW_MODE"

    if engine == "V1_LEGACY":
        print(f"[Orquestrador Roteamento] Executando Motor V1 (Legado).")
        return await classify_intent_v1(
            user_prompt=user_prompt,
            messages=messages,
            last_user_msg=last_user_msg,
            conversation_id=conversation_id,
            user_id=user_id
        )

    elif engine == "V2_SEMANTIC_VECTOR":
        print(f"[Orquestrador Roteamento] Executando Motor V2 (Semântico Vetorial & JSON Schema).")
        try:
            from semantic_router_service import classify_intent_v2
            return await classify_intent_v2(
                user_prompt=user_prompt,
                messages=messages,
                last_user_msg=last_user_msg,
                conversation_id=conversation_id,
                user_id=user_id
            )
        except Exception as v2_err:
            print(f"[Aviso Roteador V2] Falha no motor V2: {v2_err}. Recorrendo ao motor V1...")
            return await classify_intent_v1(
                user_prompt=user_prompt,
                messages=messages,
                last_user_msg=last_user_msg,
                conversation_id=conversation_id,
                user_id=user_id
            )

    else:
        # Modo Padrão: SHADOW_MODE (Prioridade V2 como resposta oficial, V1 como comparador)
        print(f"[Orquestrador Roteamento] Modo Sombra Ativo (V2 Oficial de Resposta | V1 para Comparação).")
        from semantic_router_service import classify_intent_v2

        # 1. Rota oficial prioritária (V2)
        try:
            v2_cat, v2_method, v2_raw, v2_rew = await classify_intent_v2(
                user_prompt=user_prompt,
                messages=messages,
                last_user_msg=last_user_msg,
                conversation_id=conversation_id,
                user_id=user_id
            )
        except Exception as v2_err:
            print(f"[Aviso Roteador V2 Shadow] Falha no motor V2: {v2_err}. Usando V1 como contingência...")
            return await classify_intent_v1(
                user_prompt=user_prompt,
                messages=messages,
                last_user_msg=last_user_msg,
                conversation_id=conversation_id,
                user_id=user_id
            )

        # 2. Executa V1 em segundo plano assíncrono para logging comparativo de precisão (Zero Latência no V2)
        async def _run_v1_shadow_background():
            try:
                v1_cat, v1_method, _, _ = await asyncio.wait_for(
                    classify_intent_v1(
                        user_prompt=user_prompt,
                        messages=messages,
                        last_user_msg=last_user_msg,
                        conversation_id=conversation_id,
                        user_id=user_id
                    ),
                    timeout=5.0
                )
                match_status = (v2_cat == v1_cat)
                print(f"[SHADOW MODE COMPARISON] V2 (Oficial): {v2_cat} ({v2_method}) | V1 (Comparativo): {v1_cat} ({v1_method}) | Concordância: {match_status}")
            except asyncio.TimeoutError:
                print(f"[SHADOW MODE COMPARISON] V2 (Oficial): {v2_cat} ({v2_method}) | V1 (Comparativo): TIMEOUT (>5s)")
            except Exception as v1_err:
                print(f"[Aviso Shadow Mode] Erro ao executar motor V1 de comparação: {v1_err}")

        try:
            asyncio.create_task(_run_v1_shadow_background())
        except Exception:
            pass

        # Retorna imediatamente o resultado oficial do motor V2
        return v2_cat, v2_method, v2_raw, v2_rew

