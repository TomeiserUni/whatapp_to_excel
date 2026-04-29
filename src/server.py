import io
import os
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
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    resultados = []
    for linha in linhas:
        if IS_CLOUD:
            rows = pl["pl"].processar_texto(linha, pl["produtos"], pl["sku_map"], pl["aliases"], pl["ai_client"], pl.get("exemplos", []))
        else:
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
