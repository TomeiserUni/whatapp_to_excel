import io
import os
import re
import secrets
import sys
import tempfile
import threading
import webbrowser
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, session

# ── Ambiente ─────────────────────────────────────────────────────
IS_CLOUD  = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("RENDER"))
IS_FROZEN = getattr(sys, "frozen", False)

# ── Caminhos ─────────────────────────────────────────────────────
if IS_FROZEN:
    _BUNDLE   = Path(sys._MEIPASS)
    _USER_DIR = Path.home() / "WhatsAppExcel"
else:
    _BUNDLE   = Path(__file__).resolve().parent.parent
    _USER_DIR = _BUNDLE

sys.path.insert(0, str(_BUNDLE / "src"))

# Na cloud o output é em memória; localmente usa pasta
OUTPUT_DIR = None if IS_CLOUD else _USER_DIR / "output"
if OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask ─────────────────────────────────────────────────────────
template_folder = str(_BUNDLE / "src" / "templates")
app = Flask(__name__, template_folder=template_folder)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

_pipeline = None


# ── Autenticação ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if os.environ.get("APP_PASSWORD") and not session.get("authenticated"):
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        ok_user = (username == os.environ.get("APP_USERNAME", ""))
        ok_pass = (password == os.environ.get("APP_PASSWORD", ""))
        if ok_user and ok_pass:
            session["authenticated"] = True
            return redirect("/")
        error = "Utilizador ou password incorretos."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


def _load_pipeline():
    global _pipeline
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if IS_CLOUD:
        import anthropic
        import ai_pipeline as pl
        client = anthropic.Anthropic(api_key=api_key)
        produtos, sku_map, aliases, exemplos = pl.load_produtos(_BUNDLE / "data")
        _pipeline = {"pl": pl, "produtos": produtos, "emb_prod": None,
                     "sku_map": sku_map, "aliases": aliases, "exemplos": exemplos, "ai_client": client}
    else:
        # Local: pipeline completo com embeddings + EasyOCR
        import pipeline as pl
        if api_key:
            pl.init_ai_client(api_key)
            print("[AI] Claude inicializado como complemento.")
        produtos, emb_prod, sku_map, freq_palavras, palavras_unicas, aliases = pl.load_produtos()
        _pipeline = {"pl": pl, "produtos": produtos, "emb_prod": emb_prod,
                     "sku_map": sku_map, "aliases": aliases, "ai_client": None,
                     "freq_palavras": freq_palavras, "palavras_unicas": palavras_unicas}

    print("[pipeline] Pronto.")


# ── Pré-processamento de texto ─────────────────────────────────────
def _expandir_lista_qty_partilhada(linha: str) -> list[str] | None:
    """
    "Prod1, Prod2 e Prod3 50g-16un" → ["Prod1 16", "Prod2 16", "Prod3 16"]
    Só expande se houver pelo menos uma vírgula na parte dos produtos.
    """
    m = re.search(
        r'(?:[\d,.]+\s*(?:g|gr|gramas?|ml)\s*-\s*)?(\d+)\s*(?:un(?:idades?)?|pcs?)\s*$',
        linha, re.IGNORECASE
    )
    if not m:
        return None
    qty = m.group(1)
    parte = linha[:m.start()].strip().rstrip(',').strip()
    if ',' not in parte:
        return None
    nomes = re.split(r'\s*,\s*|\s+e\s+', parte, flags=re.IGNORECASE)
    nomes = [n.strip() for n in nomes if n.strip()]
    if len(nomes) < 2:
        return None
    return [f"{nome} {qty}" for nome in nomes]


def _preprocessar_texto(texto: str) -> list[str]:
    """
    1. Une linhas de continuação (começam com ',' ou 'e ', ou a linha anterior
       termina em ',' ou ' e').
    2. Expande listas com quantidade partilhada no fim.
       Ex: "Gel transparente ,\nbranco leitoso\n,branco leitoso intenso\nPanacota 50g-16un"
           → ["Gel transparente 16", "branco leitoso 16", "branco leitoso intenso 16", "Panacota 16"]
    """
    raw = [l.strip() for l in texto.splitlines()]

    joined = []
    buf = ""
    for l in raw:
        if not l:
            if buf:
                joined.append(buf)
                buf = ""
            continue

        l_sep = l.startswith(',') or l.lower().startswith('e ')
        b_sep = buf.endswith(',') or buf.lower().endswith(' e')

        if buf and (l_sep or b_sep):
            # limpar fim do buf
            b = buf.rstrip()
            if b.endswith(','):
                b = b[:-1].rstrip()
            elif b.lower().endswith(' e'):
                b = b[:-2].rstrip()
            # limpar início de l
            c = l.lstrip()
            if c.startswith(','):
                c = c[1:].lstrip()
            elif c.lower().startswith('e '):
                c = c[2:].lstrip()
            buf = b + ', ' + c
        else:
            if buf:
                joined.append(buf)
            buf = l
    if buf:
        joined.append(buf)

    resultado = []
    for linha in joined:
        exp = _expandir_lista_qty_partilhada(linha)
        resultado.extend(exp if exp else [linha])
    return resultado


# ── Rotas ─────────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html")


@app.route("/status")
@login_required
def status():
    return jsonify({"ready": _pipeline is not None})


@app.route("/config", methods=["GET"])
@login_required
def get_config():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return jsonify({"configured": bool(key)})


@app.route("/config", methods=["POST"])
@login_required
def set_config():
    data = request.json or {}
    env_path = _USER_DIR / ".env"
    from dotenv import set_key as dotenv_set_key

    key = data.get("api_key", "").strip()
    if key:
        dotenv_set_key(str(env_path), "ANTHROPIC_API_KEY", key)
        os.environ["ANTHROPIC_API_KEY"] = key
        if _pipeline is not None and not IS_CLOUD:
            _pipeline["pl"].init_ai_client(key)

    username = data.get("app_username", "").strip()
    if username:
        dotenv_set_key(str(env_path), "APP_USERNAME", username)
        os.environ["APP_USERNAME"] = username

    password = data.get("app_password", "").strip()
    if password:
        dotenv_set_key(str(env_path), "APP_PASSWORD", password)
        os.environ["APP_PASSWORD"] = password

    return jsonify({"ok": True})


@app.route("/processar", methods=["POST"])
@login_required
def processar():
    if _pipeline is None:
        return jsonify({"error": "Modelos ainda a carregar…"}), 503

    files = request.files.getlist("imagens")
    pl = _pipeline
    resultados = []

    for f in files:
        with tempfile.NamedTemporaryFile(suffix=Path(f.filename).suffix, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = Path(tmp.name)
        try:
            if IS_CLOUD:
                rows, ocr_text = pl["pl"].processar_imagem(tmp_path, pl["produtos"], pl["sku_map"], pl["aliases"], pl["ai_client"], pl.get("exemplos", []))
                texto_origem_img = ocr_text[:300] if ocr_text else ""
            else:
                rows = pl["pl"].processar_imagem(tmp_path, pl["produtos"], pl["emb_prod"],
                                                  pl.get("freq_palavras"), pl.get("palavras_unicas"))
                texto_origem_img = None  # cada row traz o seu trecho
            if rows:
                for row in rows:
                    if IS_CLOUD:
                        produto, score, qty = row
                        trecho = texto_origem_img
                    else:
                        produto, score, qty, trecho = row
                    resultados.append({
                        "ficheiro":      f.filename,
                        "produto":       produto,
                        "qtd":           qty,
                        "score":         round(score, 3),
                        "ref":           pl["sku_map"].get(produto, ""),
                        "texto_origem":  trecho or "",
                    })
            else:
                resultados.append({"ficheiro": f.filename, "produto": "", "qtd": "", "score": 0, "ref": "", "texto_origem": ""})
        finally:
            tmp_path.unlink(missing_ok=True)

    return jsonify(resultados)


@app.route("/processar_texto", methods=["POST"])
@login_required
def processar_texto():
    if _pipeline is None:
        return jsonify({"error": "Modelos ainda a carregar…"}), 503

    texto = (request.json or {}).get("texto", "").strip()
    if not texto:
        return jsonify([])

    pl = _pipeline
    linhas = [l for l in _preprocessar_texto(texto) if l.strip()]
    resultados = []

    if IS_CLOUD:
        # Batches paralelos: divide em grupos de 15 e chama a API em simultâneo
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        from rapidfuzz import fuzz as _rfuzz

        _BATCH = 15
        batches = [linhas[i:i + _BATCH] for i in range(0, len(linhas), _BATCH)]

        raw_rows: list[tuple] = []  # (row, batch)
        with ThreadPoolExecutor(max_workers=min(len(batches), 6)) as pool:
            futs = {
                pool.submit(
                    pl["pl"].processar_texto,
                    "\n".join(b),
                    pl["produtos"], pl["sku_map"], pl["aliases"],
                    pl["ai_client"], pl.get("exemplos", [])
                ): b
                for b in batches
            }
            for fut in _as_completed(futs):
                b = futs[fut]
                try:
                    for row in (fut.result() or []):
                        raw_rows.append((row, b))
                except Exception as e:
                    print(f"[batch] erro: {e}")

        # Deduplicar por produto (mantém score mais alto)
        melhor: dict[str, tuple] = {}
        for row, b in raw_rows:
            p = row[0]
            if p not in melhor or row[1] > melhor[p][0][1]:
                melhor[p] = (row, b)

        if melhor:
            for p, (row, b) in melhor.items():
                _, s, q = row[0], row[1], row[2]
                origem = max(b, key=lambda l: _rfuzz.partial_ratio(p.lower(), l.lower()), default="")
                resultados.append({
                    "ficheiro":     "texto colado",
                    "produto":      p,
                    "qtd":          q,
                    "score":        round(s, 3),
                    "ref":          pl["sku_map"].get(p, ""),
                    "texto_origem": origem,
                })
        else:
            resultados.append({"ficheiro": "texto colado", "produto": "", "qtd": "", "score": 0, "ref": "", "texto_origem": ""})
    else:
        for linha in linhas:
            rows = pl["pl"].processar_texto(linha, pl["produtos"], pl["emb_prod"],
                                             pl.get("freq_palavras"), pl.get("palavras_unicas"))
            if rows:
                for row in rows:
                    p, s, q = row[0], row[1], row[2]
                    trecho = row[3] if len(row) > 3 else linha
                    resultados.append({
                        "ficheiro":     "texto colado",
                        "produto":      p,
                        "qtd":          q,
                        "score":        round(s, 3),
                        "ref":          pl["sku_map"].get(p, ""),
                        "texto_origem": trecho or linha,
                    })
            else:
                resultados.append({"ficheiro": "texto colado", "produto": "", "qtd": "", "score": 0, "ref": "", "texto_origem": linha})

    return jsonify(resultados)


@app.route("/exportar", methods=["POST"])
@login_required
def exportar():
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment

    FILL_GREEN  = PatternFill("solid", fgColor="C6EFCE")
    FILL_YELLOW = PatternFill("solid", fgColor="FFEB9C")
    FILL_RED    = PatternFill("solid", fgColor="FFC7CE")

    def _fill(score, produto):
        if not produto:
            return FILL_RED
        if score >= 0.90:
            return FILL_GREEN
        if score >= 0.70:
            return FILL_YELLOW
        return FILL_RED

    data = request.json or []
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Resultados"
    ws.append(["Referência", "Produto", "Quantidade", "Texto reconhecido"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.column_dimensions["A"].width = 15
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 50

    for row in data:
        produto = row.get("produto", "")
        score   = float(row.get("score", 0))
        texto   = row.get("texto_origem", "")
        label   = produto if produto else "ERRO — produto não identificado"
        ws.append([row.get("ref", ""), label, row.get("qtd", ""), texto])
        fill = _fill(score, produto)
        for cell in ws[ws.max_row]:
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=False)

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="resultados.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_load_pipeline, daemon=True).start()
    PORT = int(os.environ.get("PORT", 5000 if sys.platform == "win32" else 5001))
    HOST = "0.0.0.0"
    if not IS_CLOUD:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    from waitress import serve
    serve(app, host=HOST, port=PORT)
