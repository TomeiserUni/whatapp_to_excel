from pathlib import Path
import sys
import json
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer
import re
import math
from rapidfuzz import fuzz
import openpyxl

from utils import load_pickle, cosine_similarity, normalizar_unidades, remover_acentos
from parser import quantidade_para_produto

# =========================
# Claude AI (opcional)
# =========================
_ai_client = None

def init_ai_client(api_key: str):
    global _ai_client
    import anthropic
    _ai_client = anthropic.Anthropic(api_key=api_key)

def _ai_refine_match(trecho: str, candidatos_emb: list) -> tuple | None:
    """Pede ao Claude para confirmar o melhor produto entre os candidatos do embedding."""
    if _ai_client is None or not candidatos_emb:
        return None
    nomes = [p for p, _ in candidatos_emb[:10]]
    lista = "\n".join(f"- {n}" for n in nomes)
    prompt = (
        f"Lista de produtos de verniz/unhas:\n{lista}\n\n"
        f'Qual produto corresponde melhor a "{trecho}"?\n'
        f"Responde APENAS com o nome exato da lista, ou \"nenhum\"."
    )
    try:
        r = _ai_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        resposta = r.content[0].text.strip()
        for nome in nomes:
            if nome.lower() in resposta.lower() or fuzz.ratio(resposta.lower(), nome.lower()) > 80:
                return (nome, 0.82, 0.0, 0.82)
    except Exception as e:
        print(f"[AI] erro: {e}")
    return None

# =========================
# PATHS
# =========================
if getattr(sys, "frozen", False):
    _BUNDLE   = Path(sys._MEIPASS)
    _USER_DIR = Path.home() / "WhatsAppExcel"
else:
    _BUNDLE   = Path(__file__).resolve().parent.parent
    _USER_DIR = _BUNDLE

DATA_DIR   = _BUNDLE   / "data"
INPUT_DIR  = _USER_DIR / "input"
OUTPUT_DIR = _USER_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# MODELOS
# =========================
reader = easyocr.Reader(['pt'])
_MODEL_FINETUNED = _BUNDLE / "data" / "model_finetuned"
model = SentenceTransformer(str(_MODEL_FINETUNED) if _MODEL_FINETUNED.exists() else "all-MiniLM-L6-v2")

# =========================
# CORES TERMINAL
# =========================
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"

def header(texto, cor=C.CYAN):
    largura = 60
    print(f"\n{cor}{C.BOLD}{'═' * largura}")
    print(f"  {texto}")
    print(f"{'═' * largura}{C.RESET}")

def secao(texto, cor=C.BLUE):
    print(f"\n{cor}{C.BOLD}  ▶ {texto}{C.RESET}")

def score_cor(s):
    if s >= 0.95: return C.GREEN
    if s >= 0.85: return C.YELLOW
    return C.RED

def barra(s, largura=20):
    preenchido = int(s * largura)
    barra_str = "█" * preenchido + "░" * (largura - preenchido)
    return f"{score_cor(s)}{barra_str}{C.RESET}"

# =========================
# LOAD PRODUTOS
# =========================
def load_produtos():
    produtos = load_pickle(DATA_DIR / "prod.pkl")
    embeddings = np.load(DATA_DIR / "emb_prod.npy")
    sku_map_path = DATA_DIR / "sku_map.pkl"
    sku_map = load_pickle(sku_map_path) if sku_map_path.exists() else {}
    # Pré-computar uma vez — evita recalcular a cada linha no server
    _cache_freq_palavras = calcular_freq_palavras(produtos, STOPWORDS)
    _cache_palavras_unicas = calcular_palavras_unicas(produtos)
    return produtos, embeddings, sku_map, _cache_freq_palavras, _cache_palavras_unicas

def load_aliases() -> dict[str, str]:
    """
    Carrega data/aliases.json: mapeamentos fixos alias→produto.
    As chaves são normalizadas (lowercase, sem acentos) para casar com o input.
    """
    path = DATA_DIR / "aliases.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        remover_acentos(k.lower().strip()): v
        for k, v in raw.items()
    }

# =========================
# OCR
# =========================
def _claude_extrair_linhas(imagem_path):
    """OCR via Claude — usa quando _ai_client está disponível."""
    import base64
    ext  = Path(imagem_path).suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with open(imagem_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    try:
        r = _ai_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": img_b64}},
                {"type": "text", "text": (
                    "Transcreve APENAS o texto visível nesta imagem, linha por linha.\n"
                    "Não interpretes — copia exatamente o que está escrito."
                )}
            ]}]
        )
        texto = r.content[0].text.strip()
    except Exception as e:
        print(f"[Claude OCR] erro: {e}")
        return None  # fallback para EasyOCR
    linhas = []
    for linha in texto.splitlines():
        linha = linha.lower().strip()
        linha = remover_acentos(linha)
        linha = re.sub(r"[^\w\s]", "", linha).strip()
        linha = normalizar_unidades(linha)
        if linha:
            linhas.append(linha)
    return linhas


def extrair_linhas(imagem_path):
    result = reader.readtext(str(imagem_path))
    result.sort(key=lambda r: r[0][0][1])

    grupos = []
    grupo_atual = []
    y_atual = None
    TOLERANCIA_Y = 15

    for bbox, text, _ in result:
        y = bbox[0][1]
        text = text.lower()
        text = remover_acentos(text)
        text = re.sub(r"[^\w\s]", "", text).strip()
        text = normalizar_unidades(text)
        if not text:
            continue
        if y_atual is None or abs(y - y_atual) <= TOLERANCIA_Y:
            grupo_atual.append((bbox[0][0], text))
            y_atual = y
        else:
            if grupo_atual:
                grupos.append(grupo_atual)
            grupo_atual = [(bbox[0][0], text)]
            y_atual = y

    if grupo_atual:
        grupos.append(grupo_atual)

    linhas = []
    for grupo in grupos:
        grupo.sort(key=lambda x: x[0])
        linha = " ".join(texto for _, texto in grupo)
        linhas.append(linha)

    return linhas

# =========================
# SLIDING WINDOW
# =========================
def gerar_trechos_por_linha(linhas):
    trechos = []
    for linha in linhas:
        palavras = linha.split()
        if len(palavras) >= 2:
            trechos.append(linha)
        for i in range(len(palavras)):
            for j in range(i + 2, min(i + 6, len(palavras) + 1)):
                trecho = " ".join(palavras[i:j])
                if trecho != linha:
                    trechos.append(trecho)
    return trechos

# =========================
# ESPECIFICIDADE & COBERTURA
# =========================
def calcular_freq_palavras(produtos, stopwords):
    """Conta em quantos produtos cada palavra aparece."""
    freq = {}
    for p in produtos:
        for w in set(p.lower().split()) - stopwords:
            freq[w] = freq.get(w, 0) + 1
    return freq

def calcular_palavras_unicas(produtos: list) -> dict:
    """
    Mapeia cada palavra que aparece em exactamente um produto ao nome desse produto.
    Usado para reconhecer abreviações: se a palavra é única no catálogo, o produto
    é inequívoco mesmo quando o score está abaixo do threshold normal.
    """
    contagem: dict[str, int] = {}
    produto_de: dict[str, str] = {}
    for p in produtos:
        for w in p.lower().split():
            contagem[w] = contagem.get(w, 0) + 1
            produto_de[w] = p
    return {w: produto_de[w] for w, c in contagem.items() if c == 1}

def _trecho_identifica_univocamente(palavras_t: set, produto: str, palavras_unicas: dict) -> bool:
    """
    True se alguma palavra ALFABÉTICA do trecho identifica univocamente este produto:
    - match exacto: a palavra existe em apenas este produto no catálogo
    - match fuzzy: a palavra aproxima-se (≥82) de uma palavra única deste produto
    Números são excluídos — na mensagem indicam quantidade, não nome de produto.
    Permite reconhecer abreviações e typos como 'princepezinho'→'principezinho'.
    """
    prod_lower = produto.lower()
    prod_words = set(prod_lower.split())
    for w in palavras_t:
        if not w.isalpha():  # números (24, 36…) são quantidades, não identificadores
            continue
        if palavras_unicas.get(w) == prod_lower:
            return True
        for w_prod in prod_words:
            if (w_prod in palavras_unicas
                    and palavras_unicas[w_prod] == prod_lower
                    and fuzz.ratio(w, w_prod) >= 82):
                return True
    return False

def calcular_threshold(produto, freq_palavras, stopwords, base=0.82, penalidade=0.06):
    """
    Produtos com palavras muito comuns no catálogo (baixa especificidade)
    recebem threshold mais alto — até base + penalidade.
    Produtos com palavras únicas recebem threshold base.
    Números puros (ex: '50', '500') são excluídos: são qualificadores de variante,
    não palavras de categoria, e têm frequência artificialmente alta.
    """
    palavras = {w for w in produto.lower().split() if w not in stopwords and not w.isdigit()}
    if not palavras:
        return base
    especificidade = sum(1.0 / freq_palavras.get(w, 1) for w in palavras) / len(palavras)
    return min(base + (1.0 - min(especificidade, 1.0)) * penalidade, 0.93)

def ratio_cobertura(trecho, produto, stopwords):
    """Fração das palavras do produto que aparecem no trecho (0.0–1.0)."""
    palavras_prod = set(produto.lower().split()) - stopwords
    if not palavras_prod:
        return 1.0
    return len(palavras_prod & set(trecho.lower().split())) / len(palavras_prod)

def cobertura_produto(trecho, produto, stopwords, min_ratio=0.5):
    return ratio_cobertura(trecho, produto, stopwords) >= min_ratio

def unica_opcao_para_trecho(produto, trecho, todos_produtos, stopwords):
    """
    True se nenhum outro produto no catálogo contém todas as palavras do trecho.
    Usado no filtro OCR: se o produto é a única opção para este trecho, palavras
    extra ausentes no OCR são apenas abreviação — não devem desqualificar o produto.
    Se existem múltiplos produtos que correspondem ao mesmo trecho (ex: 'verniz gel rosa'
    → 'rosa pop' e 'rosa sakura'), as palavras diferenciadoras devem estar no OCR.
    """
    palavras_t = {w for w in trecho.lower().split() if w not in stopwords}
    for outro in todos_produtos:
        if outro.lower() == produto.lower():
            continue
        if palavras_t.issubset(set(outro.lower().split())):
            return False
    return True

def trecho_contido_em_produto(trecho, produto, stopwords):
    """
    Todas as palavras do trecho (exc. stopwords) devem existir no produto.
    O produto pode ter palavras extra — é o caso de nomes simplificados na mensagem.
    Ex: 'builder gel nude leitoso 30' → 'builder gel nude leitoso alta viscosidade 30' ✓
        'builder gel nude leitoso 30' → 'like gel 216 nude leitoso'               ✗ (falta 'builder','30')
    """
    palavras_trecho = {w for w in trecho.lower().split() if w not in stopwords}
    palavras_produto = set(produto.lower().split())
    return palavras_trecho.issubset(palavras_produto)

# =========================
# MATCHING
# =========================
def encontrar_produtos_ia(trecho, produtos, emb_prod):
    emb = model.encode(trecho, convert_to_numpy=True)
    scores = [(p, float(cosine_similarity(emb, e))) for p, e in zip(produtos, emb_prod)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]

def encontrar_produtos_levenshtein(trecho, produtos, freq_palavras=None):
    palavras_trecho = set(trecho.split())
    scores = []
    for p in produtos:
        palavras_produto = set(p.lower().split())
        s_set = fuzz.token_set_ratio(trecho, p.lower()) / 100
        # Comparação sem espaços: apanha palavras concatenadas (ex: "noblue" ≡ "no blue")
        s_set = max(s_set, fuzz.ratio(trecho.replace(" ", ""), p.lower().replace(" ", "")) / 100)

        # Parear palavras com fuzzy matching para não penalizar typos (ex: tammy↔tawny)
        matched_t, matched_p = set(), set()
        for w_t in palavras_trecho:
            if w_t in palavras_produto:
                matched_t.add(w_t); matched_p.add(w_t)
        for w_t in palavras_trecho - matched_t:
            best_r, best_p = 0, None
            for w_p in palavras_produto - matched_p:
                r = fuzz.ratio(w_t, w_p)
                if r > best_r:
                    best_r, best_p = r, w_p
            if best_r >= 58:
                matched_t.add(w_t); matched_p.add(best_p)

        # Penalizar apenas palavras verdadeiramente sem par
        # (excluir stopwords e unidades — o utilizador raramente as escreve)
        extra_produto = len((palavras_produto - matched_p) - STOPWORDS - UNIDADES) * 0.06
        extra_trecho = 0.0
        for w in palavras_trecho - matched_t:
            freq = freq_palavras.get(w, 0) if freq_palavras else 0
            # palavras comuns penalizam menos (ex: "maria" em muitos produtos)
            especificidade = min(1.0, 1.0 / max(1, freq))
            extra_trecho += 0.10 * especificidade

        s_final = max(0.0, s_set - extra_produto - extra_trecho)
        scores.append((p, s_final))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]

# =========================
# GREEDY EXPANSION
# =========================
def _match_trecho_best(
    trecho: str, produtos, emb_prod, freq_palavras,
    palavras_unicas: dict | None = None
) -> tuple[str, float, float, float] | None:
    """
    Avalia um trecho contra o catálogo.
    Retorna (melhor_produto, score_combinado, s_emb, s_lev) ou None.
    """
    palavras_t = {w for w in trecho.lower().split() if w not in STOPWORDS}
    if not palavras_t:
        return None

    candidatos = {}
    # Fuzzy primeiro (rápido) — se score muito alto, salta o sentence transformer
    lev_top = encontrar_produtos_levenshtein(trecho, produtos, freq_palavras)[:3]
    best_lev = lev_top[0][1] if lev_top else 0
    usar_emb = 0.60 <= best_lev <= 0.96  # só usa embedding na zona de incerteza
    if usar_emb:
        for p, s in encontrar_produtos_ia(trecho, produtos, emb_prod)[:3]:
            candidatos[p] = {"emb": s, "lev": 0}
    for p, s in lev_top:
        if p not in candidatos:
            candidatos[p] = {"emb": 0, "lev": s}
        else:
            candidatos[p]["lev"] = s

    melhor_p, melhor_s, melhor_emb, melhor_lev = None, 0.0, 0.0, 0.0
    for produto, sc in candidatos.items():
        s_emb, s_lev = sc["emb"], sc["lev"]
        if s_emb == 0:
            score = s_lev
        elif s_lev == 0:
            score = s_emb
        else:
            score = 0.6 * s_emb + 0.4 * s_lev
        if s_lev > 0.95:
            score += 0.05
        for k in KEYWORDS_BOOST:
            if k in produto.lower() and k in trecho:
                score += 0.05
        score = min(score, 1.0)

        threshold  = calcular_threshold(produto, freq_palavras, STOPWORDS)
        palavras_p = set(produto.lower().split())

        n_total = max(1, len(produtos))
        # Concatenações sem espaço de n-gramas do produto (ex: "no"+"blue" → "noblue")
        # para aceitar variantes escritas sem espaço pelo utilizador.
        palavras_p_list = produto.lower().split()
        concat_ngrams = {
            "".join(palavras_p_list[s:s+k])
            for k in range(2, min(5, len(palavras_p_list) + 1))
            for s in range(len(palavras_p_list) - k + 1)
        }
        sem_match = {
            w for w in palavras_t
            if w not in palavras_p
            and not any(fuzz.ratio(w, p) >= 82 for p in palavras_p)
            and not any(fuzz.ratio(w, ng) >= 82 for ng in concat_ngrams)
            # palavras comuns (aparecem em muitos produtos) não bloqueiam o match
            and freq_palavras.get(w, 0) < max(3, n_total * 0.01)
        }
        if sem_match:
            # Fallback para typos multi-caracter (ex: "tammy"/"tawny"):
            # aceita se apenas 1 palavra não fez match e a semelhança global é alta
            if len(sem_match) > 1 or fuzz.token_sort_ratio(trecho.lower(), produto.lower()) < 88:
                continue
        if len(palavras_t) >= 2:
            score = min(score + len(palavras_t) * 0.03, 1.0)

        # Se uma palavra do trecho identifica univocamente este produto, os filtros
        # de threshold e palavras genéricas não se aplicam (é uma abreviação válida).
        is_unique = (palavras_unicas is not None
                     and _trecho_identifica_univocamente(palavras_t, produto, palavras_unicas))

        if not is_unique:
            if score < 0.85 and any(g in produto.lower() for g in PALAVRAS_GENERICAS):
                continue
            if score <= threshold:
                continue
        elif score < 0.75:
            # Para palavras únicas o floor reflecte o custo máximo de palavras extra:
            # produtos até 4 palavras extra (ex: "verniz gel maria X") → score ≥ 0.76
            # produtos com 5+ palavras extra (ex: "verniz gel dancar ate ser dia") → score < 0.76 → rejeitado
            continue
        if score > melhor_s:
            melhor_s, melhor_p, melhor_emb, melhor_lev = score, produto, s_emb, s_lev

    # Complemento IA: só quando não há match nenhum (evita 100+ chamadas API por pedido)
    if melhor_p is None and _ai_client is not None:
        candidatos_emb = encontrar_produtos_ia(trecho, produtos, emb_prod)[:10]
        ai = _ai_refine_match(trecho, candidatos_emb)
        if ai:
            melhor_p, melhor_s, melhor_emb, melhor_lev = ai

    return (melhor_p, melhor_s, melhor_emb, melhor_lev) if melhor_p else None

_LEADING_QTY = re.compile(r"^\d+\s+")

def processar_linha(
    linha: str, produtos, emb_prod, freq_palavras,
    aliases: dict | None = None,
    palavras_unicas: dict | None = None,
    verbose: bool = False
) -> list[tuple[str, float, str, float, float]]:
    """
    Greedy window expansion ajustado.
    Retorna lista de (produto, score, trecho, s_emb, s_lev).
    Se verbose=True, imprime cada trecho tentado e o vencedor por posição.
    """
    # Remove número inicial de quantidade ("4 verniz gel..." → "verniz gel...")
    # para evitar que o número seja consumido no matching de produto.
    linha_match = _LEADING_QTY.sub("", linha)
    palavras = linha_match.split()
    n = len(palavras)
    consumed = [False] * n

    MIN_WIN = 1
    resultados = []
    i = 0

    while i < n:
        if consumed[i] or palavras[i] in STOPWORDS:
            i += 1
            continue

        best = None  # (produto, score, trecho, end_idx, s_emb, s_lev)

        # Aliases têm prioridade máxima — verifica do mais longo para o mais curto
        if aliases:
            for length in range(n - i, 0, -1):
                if any(consumed[i + k] for k in range(length)):
                    continue
                trecho_alias = " ".join(palavras[i:i + length])
                if trecho_alias in aliases:
                    produto_alias = aliases[trecho_alias]
                    best = (produto_alias, 1.0, trecho_alias, i + length, 0.0, 1.0)
                    if verbose:
                        print(f"      {C.GREEN}[alias]{C.RESET} {C.WHITE}'{trecho_alias}'{C.RESET}"
                              f" → {C.GREEN}{produto_alias}{C.RESET}  {C.GREEN}1.000{C.RESET}")
                    break

        if best is None:
            pass  # continua para o fuzzy expansion abaixo

        for size in range(MIN_WIN, n - i + 1) if best is None else []:
            j = i + size

            if any(consumed[k] for k in range(i, j)):
                break

            trecho = " ".join(palavras[i:j])
            result = _match_trecho_best(trecho, produtos, emb_prod, freq_palavras, palavras_unicas=palavras_unicas)

            if verbose:
                if result:
                    p_t, s_t, e_t, l_t = result
                    cor = score_cor(s_t)
                    print(f"      {C.DIM}·{C.RESET} {C.WHITE}'{trecho}'{C.RESET}"
                          f" → {cor}{p_t}{C.RESET}"
                          f"  {cor}{s_t:.3f}{C.RESET}"
                          f"  {C.DIM}emb{C.RESET} {score_cor(e_t)}{e_t:.3f}{C.RESET}"
                          f"  {C.DIM}lev{C.RESET} {score_cor(l_t)}{l_t:.3f}{C.RESET}")
                else:
                    print(f"      {C.DIM}· '{trecho}' → —{C.RESET}")

            if result is None:
                continue

            produto, score, s_emb, s_lev = result

            if best is None or score > best[1]:
                best = (produto, score, trecho, j, s_emb, s_lev)
            elif len(trecho) > len(best[2]) and score >= best[1] - 0.10:
                # trecho mais longo com score próximo → produto mais específico ganha
                best = (produto, score, trecho, j, s_emb, s_lev)
            else:
                if score < best[1] - 0.10:
                    break

        if best:
            end = best[3]

            while end > i + MIN_WIN and palavras[end - 1].lower() in STOPWORDS:
                end -= 1

            trecho_final = " ".join(palavras[i:end])

            if len(trecho_final.split()) == 1 and best[1] < 0.85:
                palavras_t_final = {w for w in trecho_final.split() if w not in STOPWORDS}
                word_is_unique = (palavras_unicas is not None
                                  and _trecho_identifica_univocamente(palavras_t_final, best[0], palavras_unicas))
                if not word_is_unique:
                    if verbose:
                        print(f"      {C.DIM}  → '{trecho_final}' rejeitado (1 palavra, score < 0.85){C.RESET}")
                    i += 1
                    continue

            if verbose:
                cor = score_cor(best[1])
                print(f"    {cor}✓{C.RESET} {C.DIM}'{trecho_final}'{C.RESET}"
                      f" → {cor}{best[0]}{C.RESET}  {cor}{best[1]:.3f}{C.RESET}"
                      f"  {C.DIM}emb{C.RESET} {score_cor(best[4])}{best[4]:.3f}{C.RESET}"
                      f"  {C.DIM}lev{C.RESET} {score_cor(best[5])}{best[5]:.3f}{C.RESET}")

            resultados.append((best[0], best[1], trecho_final, best[4], best[5]))

            for k in range(i, end):
                consumed[k] = True
            i = end
        else:
            i += 1

    return resultados

# =========================
# EXCEL OUTPUT
# =========================
def _guardar_excel(resultados: dict, caminho: Path, sku_map: dict = None) -> None:
    """
    Escreve os resultados num ficheiro Excel.
    Colunas: Imagem | Referência | Produto | Quantidade | Score
    Uma folha por execução, linhas agrupadas por imagem.
    """
    sku_map = sku_map or {}
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Resultados"

    # Cabeçalho
    ws.append(["Imagem", "Referência", "Produto", "Quantidade", "Score"])
    header_font = openpyxl.styles.Font(bold=True)
    for cell in ws[1]:
        cell.font = header_font

    # Dados
    for imagem, produtos in resultados.items():
        for p, s, qty in produtos:
            ref = sku_map.get(p, "")
            ws.append([imagem, ref, p, qty, round(s, 4)])

    # Largura das colunas
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 15
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 10

    wb.save(caminho)


# =========================
# CONSTANTES DE MATCHING
# =========================
STOPWORDS       = {"de", "da", "do", "com", "e", "para", "ola", "preciso",
                   "destes", "produtos", "seguintes", "enviarme", "podes", "os"}
UNIDADES        = {"ml", "gr", "grs", "g", "kg", "cm", "mm", "l", "lt", "lts",
                   "un", "und", "unid", "pcs", "pc"}
KEYWORDS_BOOST  = ["primer", "bailarina", "transparente"]
PALAVRAS_GENERICAS = {"gel", "tips", "builder"}

# Sinónimos aplicados antes do matching (informal → nome no catálogo)
SINONIMOS = {
    "direita": "reta",
    "diretas": "retas",
    "direto":  "reto",
}

_DE_CADA = re.compile(
    r'^(?:(\d+)\s+)?(.+?)\s+(?:\d+\s+)?(?:pack\s+)?de\s+cada$',
    re.IGNORECASE
)

def _aplicar_sinonimos(texto: str) -> str:
    for original, substituto in SINONIMOS.items():
        texto = re.sub(rf'\b{original}\b', substituto, texto)
    return texto

def _expandir_de_cada(linha: str, produtos: list, freq_palavras: dict, qty_override: int = 1) -> list[tuple[str, float, int]] | None:
    """Se a linha tiver padrão 'X de cada', devolve todos os produtos que correspondem a X."""
    m = _DE_CADA.match(linha.strip())
    if not m:
        return None
    qty = int(m.group(1)) if m.group(1) else qty_override
    trecho = _aplicar_sinonimos(m.group(2).strip())
    resultados = []
    for p in produtos:
        score = fuzz.token_set_ratio(trecho, p.lower()) / 100
        if score >= 0.75:
            resultados.append((p, score, qty))
    resultados.sort(key=lambda x: -x[1])
    return resultados if resultados else None


# =========================
# FUSÃO DE LINHAS CURTAS
# =========================
def _fundir_linhas_curtas(linhas: list[str], produtos, emb_prod, freq_palavras, aliases=None, palavras_unicas=None) -> list[str]:
    """
    Linhas com 1-2 palavras significativas são fundidas com a linha ANTERIOR:
    - 1 palavra significativa → funde sempre (ex: 'transparente' é atributo do
      produto acima, nunca um produto isolado)
    - 2 palavras significativas → funde só se não produzirem match
    """
    resultado = []

    for linha in linhas:
        palavras_sig = [w for w in _LEADING_QTY.sub("", linha).split() if w not in STOPWORDS]
        linha_sem_qtd = _LEADING_QTY.sub("", linha).strip()

        if len(palavras_sig) == 1 and resultado:
            # 1 palavra: funde sempre com a linha anterior
            resultado[-1] = resultado[-1] + " " + linha_sem_qtd
            continue

        if len(palavras_sig) == 2 and resultado:
            matches = processar_linha(linha, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas)
            if not matches:
                resultado[-1] = resultado[-1] + " " + linha_sem_qtd
                continue

        resultado.append(linha)

    return resultado


# =========================
# PROCESSAR IMAGEM (silent — sem prints, para usar no app)
# =========================
def processar_imagem(img_path: Path, produtos: list, emb_prod,
                     freq_palavras=None, palavras_unicas=None) -> list[tuple[str, float, int]]:
    """
    Processa uma imagem e devolve [(produto, score, quantidade), ...].
    Usa greedy expansion com word consumption por linha (ver processar_linha).
    Sem output no terminal — para uso no app.
    """
    aliases        = load_aliases()
    freq_palavras  = freq_palavras  or calcular_freq_palavras(produtos, STOPWORDS)
    palavras_unicas = palavras_unicas or calcular_palavras_unicas(produtos)
    linhas = (_claude_extrair_linhas(img_path) if _ai_client else None) or extrair_linhas(img_path)
    linhas         = _fundir_linhas_curtas(linhas, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas)

    ocr_tokens, ocr_tokens_num = set(), set()
    for linha in linhas:
        for w in linha.split():
            if w not in STOPWORDS:
                (ocr_tokens if w.isalpha() else ocr_tokens_num).add(w)

    # Greedy expansion por linha; se o mesmo produto aparecer em várias linhas → score máximo
    melhor: dict[str, tuple[float, str]] = {}   # produto → (score, trecho)
    ordem:  dict[str, int] = {}                 # produto → índice da primeira linha onde apareceu
    for i, linha in enumerate(linhas):
        for produto, score, trecho, *_ in processar_linha(linha, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas):
            if produto not in melhor or score > melhor[produto][0]:
                melhor[produto] = (score, trecho)
            if produto not in ordem:
                ordem[produto] = i

    resultado = [(p, s, t) for p, (s, t) in melhor.items()]
    resultado.sort(key=lambda x: ordem.get(x[0], 0))

    # Filtro OCR
    def no_ocr(w):
        return (w in ocr_tokens) if w.isalpha() else any(fuzz.ratio(w, t) >= 80 for t in ocr_tokens_num)

    validados = []
    for p, s, t in resultado:
        ausentes = {w for w in p.lower().split() if w not in STOPWORDS and not no_ocr(w)}
        if not ausentes or unica_opcao_para_trecho(p, t, produtos, STOPWORDS):
            validados.append((p, s, t))
    resultado = validados

    # Filtro subsumption
    words    = {p: set(p.lower().split()) - STOPWORDS for p, _, _ in resultado}
    subsumed = {p1 for p1, w1 in words.items()
                for p2, w2 in words.items() if p1 != p2 and w1.issubset(w2)}
    resultado = [(p, s, t) for p, s, t in resultado if p not in subsumed]
    resultado.sort(key=lambda x: ordem.get(x[0], 0))

    return [(p, s, quantidade_para_produto(t, linhas, produto_nome=p))
            for p, s, t in resultado]


# =========================
# PROCESSAR TEXTO (sem OCR — texto colado directamente)
# =========================
def processar_texto(texto: str, produtos: list, emb_prod,
                    freq_palavras=None, palavras_unicas=None) -> list[tuple[str, float, int]]:
    """
    Processa texto colado (sem OCR) e devolve [(produto, score, quantidade), ...].
    """
    linhas = []
    for linha in texto.splitlines():
        linha = re.sub(r"^\s*\d+\.\s*", "", linha.strip())  # remove "N. " de lista numerada
        linha = linha.lower()
        linha = remover_acentos(linha)
        linha = re.sub(r"[^\w\s]", "", linha).strip()
        linha = normalizar_unidades(linha)
        linha = _aplicar_sinonimos(linha)
        if linha:
            linhas.append(linha)

    aliases         = load_aliases()
    freq_palavras   = freq_palavras   or calcular_freq_palavras(produtos, STOPWORDS)
    palavras_unicas = palavras_unicas or calcular_palavras_unicas(produtos)
    linhas          = _fundir_linhas_curtas(linhas, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas)

    melhor: dict[str, tuple[float, str]] = {}
    ordem:  dict[str, int] = {}
    for i, linha in enumerate(linhas):
        # Padrão "X de cada" → expande para todas as variantes
        de_cada = _expandir_de_cada(linha, produtos, freq_palavras)
        if de_cada:
            for produto, score, qty in de_cada:
                if produto not in melhor or score > melhor[produto][0]:
                    melhor[produto] = (score, linha)
                if produto not in ordem:
                    ordem[produto] = i
            continue
        for produto, score, trecho, *_ in processar_linha(linha, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas):
            if produto not in melhor or score > melhor[produto][0]:
                melhor[produto] = (score, trecho)
            if produto not in ordem:
                ordem[produto] = i

    resultado = [(p, s, t) for p, (s, t) in melhor.items()]
    resultado.sort(key=lambda x: ordem.get(x[0], 0))

    words    = {p: set(p.lower().split()) - STOPWORDS for p, _, _ in resultado}
    subsumed = {p1 for p1, w1 in words.items()
                for p2, w2 in words.items() if p1 != p2 and w1.issubset(w2)}
    resultado = [(p, s, t) for p, s, t in resultado if p not in subsumed]
    resultado.sort(key=lambda x: ordem.get(x[0], 0))

    return [(p, s, quantidade_para_produto(t, linhas, produto_nome=p))
            for p, s, t in resultado]


# =========================
# PIPELINE (CLI com output detalhado)
# =========================
def run():
    produtos, emb_prod, sku_map = load_produtos()
    aliases   = load_aliases()
    resultados = {}

    freq_palavras   = calcular_freq_palavras(produtos, STOPWORDS)
    palavras_unicas = calcular_palavras_unicas(produtos)

    for img in INPUT_DIR.iterdir():
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        header(f"IMAGEM: {img.name}", C.MAGENTA)

        linhas = extrair_linhas(img)
        linhas = _fundir_linhas_curtas(linhas, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas)

        # Tokens alfabéticos (match exato) e numéricos/unidades (fuzzy, ex: "30ml" ≈ "30m1")
        ocr_tokens, ocr_tokens_num = set(), set()
        for linha in linhas:
            for w in linha.split():
                if w not in STOPWORDS:
                    (ocr_tokens if w.isalpha() else ocr_tokens_num).add(w)

        secao(f"OCR — {len(linhas)} linhas detectadas", C.CYAN)
        for i, l in enumerate(linhas, 1):
            print(f"    {C.DIM}{i:2}.{C.RESET} {C.WHITE}{l}{C.RESET}")

        # ---- GREEDY EXPANSION ----
        secao("GREEDY EXPANSION", C.BLUE)

        melhor_scores: dict[str, tuple[float, str]] = {}   # produto → (score, trecho)
        ordem_scores:  dict[str, int] = {}                 # produto → índice da primeira linha

        for i, linha in enumerate(linhas):
            if all(w in STOPWORDS for w in linha.split()):
                continue

            print(f"\n  {C.DIM}linha:{C.RESET} {C.WHITE}'{linha}'{C.RESET}")

            matches = processar_linha(linha, produtos, emb_prod, freq_palavras, aliases=aliases, palavras_unicas=palavras_unicas, verbose=True)
            if not matches:
                print(f"    {C.DIM}(sem match){C.RESET}")
            for produto, score, trecho, *_ in matches:
                if produto not in melhor_scores or score > melhor_scores[produto][0]:
                    melhor_scores[produto] = (score, trecho)
                if produto not in ordem_scores:
                    ordem_scores[produto] = i

        # ---- FILTROS ----
        resultado_final = [(p, round(s, 4), t) for p, (s, t) in melhor_scores.items()]
        resultado_final.sort(key=lambda x: ordem_scores.get(x[0], 0))

        def palavra_no_ocr(palavra):
            return (palavra in ocr_tokens) if palavra.isalpha() \
                else any(fuzz.ratio(palavra, tk) >= 80 for tk in ocr_tokens_num)

        ocr_filtrados = False
        validados = []
        for p, s, t in resultado_final:
            ausentes = {w for w in p.lower().split() if w not in STOPWORDS and not palavra_no_ocr(w)}
            if ausentes:
                if unica_opcao_para_trecho(p, t, produtos, STOPWORDS):
                    validados.append((p, s, t))
                    print(f"    {C.DIM}~ único match  {p}  (abreviado, ausentes: {', '.join(sorted(ausentes))}){C.RESET}")
                else:
                    if not ocr_filtrados:
                        print(f"\n  {C.RED}{C.BOLD}  ▶ FILTRO OCR{C.RESET}")
                        ocr_filtrados = True
                    print(f"    {C.RED}✗{C.RESET} {p}  {C.DIM}(ausentes no OCR: {', '.join(sorted(ausentes))}){C.RESET}")
            else:
                validados.append((p, s, t))
        resultado_final = validados

        nomes_words = {p: set(p.lower().split()) - STOPWORDS for p, _, _ in resultado_final}
        subsumed = {}
        for p1, w1 in nomes_words.items():
            for p2, w2 in nomes_words.items():
                if p1 != p2 and w1.issubset(w2):
                    subsumed[p1] = p2
                    break
        if subsumed:
            print(f"\n  {C.YELLOW}{C.BOLD}  ▶ FILTRO SUBSUMPTION{C.RESET}")
            for p, sup in subsumed.items():
                print(f"    {C.YELLOW}⊂{C.RESET} '{p}' contido em '{sup}' → removido")
        resultado_final = [(p, s, t) for p, s, t in resultado_final if p not in subsumed]
        resultado_final.sort(key=lambda x: ordem_scores.get(x[0], 0))

        resultado_com_qtd = [(p, s, quantidade_para_produto(t, linhas, produto_nome=p), t)
                             for p, s, t in resultado_final]
        resultados[img.name] = [(p, s, qty) for p, s, qty, _ in resultado_com_qtd]

        # ---- RESULTADO FINAL ----
        secao("RESULTADO FINAL", C.GREEN)
        if not resultado_com_qtd:
            print(f"  {C.RED} Nenhum produto encontrado{C.RESET}")
        else:
            print(f"  {'PRODUTO':<50} {'SCORE':>7}  {'QTD':>4}  TRECHO")
            print(f"  {'─'*50} {'─'*7}  {'─'*4}  {'─'*30}")
            for p, s, qty, trecho in resultado_com_qtd:
                cor = score_cor(s)
                print(f"  {cor}{p:<50}{C.RESET} {cor}{s:>7.4f}{C.RESET}  "
                      f"{C.CYAN}{qty:>4}x{C.RESET}  {C.DIM}'{trecho}'{C.RESET}")

    # ---- GUARDAR E ABRIR EXCEL ----
    excel_path = OUTPUT_DIR / "resultados.xlsx"
    _guardar_excel(resultados, excel_path, sku_map)
    print(f"\n{C.GREEN}{C.BOLD}  ✔ Resultados guardados em output/resultados.xlsx{C.RESET}\n")
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", str(excel_path)])

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()