from pathlib import Path
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer
import re
import math
from rapidfuzz import fuzz
import openpyxl

from utils import load_pickle, cosine_similarity, normalizar_unidades
from parser import quantidade_para_produto

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# =========================
# MODELOS
# =========================
reader = easyocr.Reader(['pt'])
model = SentenceTransformer("all-MiniLM-L6-v2")

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
    return produtos, embeddings

# =========================
# OCR
# =========================
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

def calcular_threshold(produto, freq_palavras, stopwords, base=0.82, penalidade=0.08):
    """
    Produtos com palavras muito comuns no catálogo (baixa especificidade)
    recebem threshold mais alto — até base + penalidade.
    Produtos com palavras únicas recebem threshold base.
    """
    palavras = set(produto.lower().split()) - stopwords
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

def encontrar_produtos_levenshtein(trecho, produtos):
    palavras_trecho = set(trecho.split())
    scores = []
    for p in produtos:
        palavras_produto = set(p.lower().split())
        s_set = fuzz.token_set_ratio(trecho, p.lower()) / 100
        palavras_extra = palavras_produto - palavras_trecho
        penalizacao = len(palavras_extra) * 0.06
        s_final = max(0.0, s_set - penalizacao)
        scores.append((p, s_final))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:3]

# =========================
# GREEDY EXPANSION
# =========================
def _match_trecho_best(
    trecho: str, produtos, emb_prod, freq_palavras
) -> tuple[str, float, float, float] | None:
    """
    Avalia um trecho contra o catálogo.
    Retorna (melhor_produto, score_combinado, s_emb, s_lev) ou None.
    """
    palavras_t = {w for w in trecho.lower().split() if w not in STOPWORDS}
    if not palavras_t:
        return None

    candidatos = {}
    for p, s in encontrar_produtos_ia(trecho, produtos, emb_prod)[:3]:
        candidatos[p] = {"emb": s, "lev": 0}
    for p, s in encontrar_produtos_levenshtein(trecho, produtos)[:3]:
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

        if palavras_t - palavras_p:
            continue
        if len(palavras_t) >= 2:
            score = min(score + len(palavras_t) * 0.03, 1.0)
        if score < 0.85 and any(g in produto.lower() for g in PALAVRAS_GENERICAS):
            continue
        if score <= threshold:
            continue
        if score > melhor_s:
            melhor_s, melhor_p, melhor_emb, melhor_lev = score, produto, s_emb, s_lev

    return (melhor_p, melhor_s, melhor_emb, melhor_lev) if melhor_p else None

def processar_linha(
    linha: str, produtos, emb_prod, freq_palavras,
    verbose: bool = False
) -> list[tuple[str, float, str, float, float]]:
    """
    Greedy window expansion ajustado.
    Retorna lista de (produto, score, trecho, s_emb, s_lev).
    Se verbose=True, imprime cada trecho tentado e o vencedor por posição.
    """
    palavras = linha.split()
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

        for size in range(MIN_WIN, n - i + 1):
            j = i + size

            if any(consumed[k] for k in range(i, j)):
                break

            trecho = " ".join(palavras[i:j])
            result = _match_trecho_best(trecho, produtos, emb_prod, freq_palavras)

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
            elif abs(score - best[1]) < 0.001:
                if len(trecho) > len(best[2]):
                    best = (produto, score, trecho, j, s_emb, s_lev)
            else:
                if score < best[1] - 0.05:
                    break

        if best:
            end = best[3]

            while end > i + MIN_WIN and palavras[end - 1].lower() in STOPWORDS:
                end -= 1

            trecho_final = " ".join(palavras[i:end])

            if len(trecho_final.split()) == 1 and best[1] < 0.85:
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
def _guardar_excel(resultados: dict, caminho: Path) -> None:
    """
    Escreve os resultados num ficheiro Excel.
    Colunas: Imagem | Produto | Quantidade | Score
    Uma folha por execução, linhas agrupadas por imagem.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Resultados"

    # Cabeçalho
    ws.append(["Imagem", "Produto", "Quantidade", "Score"])
    header_font = openpyxl.styles.Font(bold=True)
    for cell in ws[1]:
        cell.font = header_font

    # Dados
    for imagem, produtos in resultados.items():
        for p, s, qty in produtos:
            ws.append([imagem, p, qty, round(s, 4)])

    # Largura das colunas
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 10

    wb.save(caminho)


# =========================
# CONSTANTES DE MATCHING
# =========================
STOPWORDS       = {"de", "da", "do", "com", "e", "para", "ola", "preciso",
                   "destes", "produtos", "seguintes", "enviarme", "podes", "os"}
KEYWORDS_BOOST  = ["primer", "bailarina", "transparente"]
PALAVRAS_GENERICAS = {"gel", "tips", "builder"}


# =========================
# PROCESSAR IMAGEM (silent — sem prints, para usar no app)
# =========================
def processar_imagem(img_path: Path, produtos: list, emb_prod) -> list[tuple[str, float, int]]:
    """
    Processa uma imagem e devolve [(produto, score, quantidade), ...].
    Usa greedy expansion com word consumption por linha (ver processar_linha).
    Sem output no terminal — para uso no app.
    """
    freq_palavras = calcular_freq_palavras(produtos, STOPWORDS)
    linhas        = extrair_linhas(img_path)

    ocr_tokens, ocr_tokens_num = set(), set()
    for linha in linhas:
        for w in linha.split():
            if w not in STOPWORDS:
                (ocr_tokens if w.isalpha() else ocr_tokens_num).add(w)

    # Greedy expansion por linha; se o mesmo produto aparecer em várias linhas → score máximo
    melhor: dict[str, tuple[float, str]] = {}   # produto → (score, trecho)
    for linha in linhas:
        for produto, score, trecho, *_ in processar_linha(linha, produtos, emb_prod, freq_palavras):
            if produto not in melhor or score > melhor[produto][0]:
                melhor[produto] = (score, trecho)

    resultado = [(p, s, t) for p, (s, t) in melhor.items()]
    resultado.sort(key=lambda x: x[1], reverse=True)

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
    resultado.sort(key=lambda x: x[1], reverse=True)

    return [(p, s, quantidade_para_produto(t, linhas, produto_nome=p))
            for p, s, t in resultado]


# =========================
# PIPELINE (CLI com output detalhado)
# =========================
def run():
    produtos, emb_prod = load_produtos()
    resultados = {}

    freq_palavras = calcular_freq_palavras(produtos, STOPWORDS)

    for img in INPUT_DIR.iterdir():
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        header(f"IMAGEM: {img.name}", C.MAGENTA)

        linhas = extrair_linhas(img)

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

        for linha in linhas:
            if all(w in STOPWORDS for w in linha.split()):
                continue

            print(f"\n  {C.DIM}linha:{C.RESET} {C.WHITE}'{linha}'{C.RESET}")

            matches = processar_linha(linha, produtos, emb_prod, freq_palavras, verbose=True)
            if not matches:
                print(f"    {C.DIM}(sem match){C.RESET}")
            for produto, score, trecho, *_ in matches:
                if produto not in melhor_scores or score > melhor_scores[produto][0]:
                    melhor_scores[produto] = (score, trecho)

        # ---- FILTROS ----
        resultado_final = [(p, round(s, 4), t) for p, (s, t) in melhor_scores.items()]
        resultado_final.sort(key=lambda x: x[1], reverse=True)

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
        resultado_final.sort(key=lambda x: x[1], reverse=True)

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
    _guardar_excel(resultados, excel_path)
    print(f"\n{C.GREEN}{C.BOLD}  ✔ Resultados guardados em output/resultados.xlsx{C.RESET}\n")
    import subprocess, sys
    if sys.platform == "darwin":
        subprocess.run(["open", str(excel_path)])

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()