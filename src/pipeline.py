from pathlib import Path
import numpy as np
import easyocr
from sentence_transformers import SentenceTransformer
import json
import re
import math
from rapidfuzz import fuzz

from utils import load_pickle, cosine_similarity

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

def cobertura_produto(trecho, produto, stopwords, min_ratio=0.5):
    """
    Verifica se o trecho cobre pelo menos min_ratio das palavras do produto.
    Evita que fragmentos curtos ('gel rosa') façam match com produtos longos.
    """
    palavras_prod = set(produto.lower().split()) - stopwords
    palavras_trecho = set(trecho.lower().split())
    if not palavras_prod:
        return True
    return len(palavras_prod & palavras_trecho) / len(palavras_prod) >= min_ratio

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
# PIPELINE
# =========================
def run():
    produtos, emb_prod = load_produtos()
    resultados = {}

    STOPWORDS = {"de", "da", "do", "com", "e", "para", "ola", "preciso",
                 "destes", "produtos", "seguintes", "enviarme", "podes", "os"}
    KEYWORDS_BOOST = ["primer", "bailarina", "transparente"]
    PALAVRAS_GENERICAS = {"gel", "tips", "builder"}

    freq_palavras = calcular_freq_palavras(produtos, STOPWORDS)

    for img in INPUT_DIR.iterdir():
        if img.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        header(f"IMAGEM: {img.name}", C.MAGENTA)

        linhas = extrair_linhas(img)

        secao(f"OCR — {len(linhas)} linhas detectadas", C.CYAN)
        for i, l in enumerate(linhas, 1):
            print(f"    {C.DIM}{i:2}.{C.RESET} {C.WHITE}{l}{C.RESET}")

        trechos = gerar_trechos_por_linha(linhas)
        print(f"\n  {C.DIM}Trechos gerados: {len(trechos)}{C.RESET}")

        scores_produtos = {}

        # ---- PROCESSAMENTO ----
        secao("PROCESSAMENTO DE TRECHOS", C.BLUE)

        for trecho in trechos:
            palavras_trecho = trecho.split()
            if all(p in STOPWORDS for p in palavras_trecho):
                continue

            matches_emb = encontrar_produtos_ia(trecho, produtos, emb_prod)[:1]
            matches_lev = encontrar_produtos_levenshtein(trecho, produtos)[:1]

            candidatos = {}
            for p, s in matches_emb:
                candidatos[p] = {"emb": s, "lev": 0}
            for p, s in matches_lev:
                if p not in candidatos:
                    candidatos[p] = {"emb": 0, "lev": s}
                else:
                    candidatos[p]["lev"] = s

            trecho_tem_match = False

            for produto, scores in candidatos.items():
                s_emb = scores["emb"]
                s_lev = scores["lev"]

                if s_emb == 0:
                    score_final = s_lev
                elif s_lev == 0:
                    score_final = s_emb
                else:
                    score_final = 0.6 * s_emb + 0.4 * s_lev

                if s_lev > 0.95:
                    score_final += 0.05
                for k in KEYWORDS_BOOST:
                    if k in produto.lower() and k in trecho:
                        score_final += 0.05
                score_final = min(score_final, 1.0)

                threshold = calcular_threshold(produto, freq_palavras, STOPWORDS)

                if score_final < 0.85 and any(p in produto.lower() for p in PALAVRAS_GENERICAS):
                    continue

                if not cobertura_produto(trecho, produto, STOPWORDS):
                    continue

                if score_final > threshold:
                    tamanho_trecho = len(palavras_trecho)
                    if produto not in scores_produtos:
                        scores_produtos[produto] = {"scores": [], "max_trecho": 0, "melhor_trecho": ""}
                    scores_produtos[produto]["scores"].append(score_final)
                    if tamanho_trecho > scores_produtos[produto]["max_trecho"]:
                        scores_produtos[produto]["max_trecho"] = tamanho_trecho
                        scores_produtos[produto]["melhor_trecho"] = trecho

                    if not trecho_tem_match:
                        trecho_tem_match = True
                        print(f"\n  {C.DIM}trecho:{C.RESET} {C.WHITE}'{trecho}'{C.RESET}")

                    cor = score_cor(score_final)
                    print(f"    {cor}✓{C.RESET} {produto}")
                    print(f"      {barra(score_final)} {cor}{score_final:.3f}{C.RESET}  "
                          f"{C.DIM}emb:{s_emb:.2f}  lev:{s_lev:.2f}{C.RESET}")

        # ---- AGREGAÇÃO ----
        secao("AGREGAÇÃO", C.BLUE)

        resultado_final = []
        for produto, dados in scores_produtos.items():
            lista_scores = dados["scores"]
            media = sum(lista_scores) / len(lista_scores)
            ocorrencias = len(lista_scores)
            score_final = media + 0.03 * math.log1p(min(ocorrencias, 5))
            resultado_final.append((produto, round(score_final, 4), dados["melhor_trecho"], ocorrencias))

        resultado_final = [(p, s, t, o) for p, s, t, o in resultado_final if s > 0.85]
        resultado_final.sort(key=lambda x: x[1], reverse=True)
        resultados[img.name] = [(p, s) for p, s, _, _ in resultado_final]

        # ---- RESULTADO FINAL ----
        secao("RESULTADO FINAL", C.GREEN)

        if not resultado_final:
            print(f"  {C.RED}⚠ Nenhum produto encontrado{C.RESET}")
        else:
            print(f"  {'PRODUTO':<50} {'SCORE':>7}  {'OCORR':>5}  MELHOR TRECHO")
            print(f"  {'─'*50} {'─'*7}  {'─'*5}  {'─'*30}")
            for p, s, melhor_trecho, ocorr in resultado_final:
                cor = score_cor(s)
                print(f"  {cor}{p:<50}{C.RESET} "
                      f"{cor}{s:>7.4f}{C.RESET}  "
                      f"{C.DIM}{ocorr:>5}x{C.RESET}  "
                      f"{C.DIM}'{melhor_trecho}'{C.RESET}")

    with open(OUTPUT_DIR / "resultados.json", "w") as f:
        json.dump(resultados, f, indent=2)

    print(f"\n{C.GREEN}{C.BOLD}  ✔ Resultados guardados em output/resultados.json{C.RESET}\n")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run()