import io
import os
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

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

_pipeline = None


def _load_pipeline():
    global _pipeline
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("NVIDIA_API_KEY")

    if IS_CLOUD:
        # Cloud: pipeline leve via NVIDIA API (sem torch/easyocr)
        from openai import OpenAI
        import ai_pipeline as pl
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
        produtos, sku_map, aliases = pl.load_produtos(_BUNDLE / "data")
        _pipeline = {"pl": pl, "produtos": produtos, "emb_prod": None,
                     "sku_map": sku_map, "aliases": aliases, "ai_client": client}
    else:
        # Local: pipeline completo com embeddings + EasyOCR
        import pipeline as pl
        if api_key:
            pl.init_ai_client(api_key)
            print("[AI] Cliente NVIDIA inicializado.")
        produtos, emb_prod, sku_map = pl.load_produtos()
        _pipeline = {"pl": pl, "produtos": produtos,
                     "emb_prod": emb_prod, "sku_map": sku_map, "aliases": {}, "ai_client": None}

    print("[pipeline] Pronto.")


# ── Rotas ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({"ready": _pipeline is not None})


@app.route("/config", methods=["GET"])
def get_config():
    key = os.environ.get("NVIDIA_API_KEY", "")
    return jsonify({"configured": bool(key)})


@app.route("/config", methods=["POST"])
def set_config():
    data = request.json or {}
    key = data.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "Chave vazia"}), 400
    env_path = _USER_DIR / ".env"
    from dotenv import set_key as dotenv_set_key
    dotenv_set_key(str(env_path), "NVIDIA_API_KEY", key)
    os.environ["NVIDIA_API_KEY"] = key
    if _pipeline is not None and not IS_CLOUD:
        _pipeline["pl"].init_ai_client(key)
    return jsonify({"ok": True})


@app.route("/processar", methods=["POST"])
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
                rows = pl["pl"].processar_imagem(tmp_path, pl["produtos"], pl["sku_map"], pl["aliases"], pl["ai_client"])
            else:
                rows = pl["pl"].processar_imagem(tmp_path, pl["produtos"], pl["emb_prod"])
            for produto, score, qty in rows:
                resultados.append({
                    "ficheiro": f.filename,
                    "produto":  produto,
                    "qtd":      qty,
                    "score":    round(score, 3),
                    "ref":      pl["sku_map"].get(produto, ""),
                })
        finally:
            tmp_path.unlink(missing_ok=True)

    return jsonify(resultados)


@app.route("/processar_texto", methods=["POST"])
def processar_texto():
    if _pipeline is None:
        return jsonify({"error": "Modelos ainda a carregar…"}), 503

    texto = (request.json or {}).get("texto", "").strip()
    if not texto:
        return jsonify([])

    pl = _pipeline
    if IS_CLOUD:
        rows = pl["pl"].processar_texto(texto, pl["produtos"], pl["sku_map"], pl["aliases"], pl["ai_client"])
    else:
        rows = pl["pl"].processar_texto(texto, pl["produtos"], pl["emb_prod"])
    return jsonify([{
        "ficheiro": "texto colado",
        "produto":  p,
        "qtd":      q,
        "score":    round(s, 3),
        "ref":      pl["sku_map"].get(p, ""),
    } for p, s, q in rows])


@app.route("/exportar", methods=["POST"])
def exportar():
    import openpyxl
    data = request.json or []
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Resultados"
    ws.append(["Imagem", "Referência", "Produto", "Quantidade", "Score"])
    for cell in ws[1]:
        cell.font = openpyxl.styles.Font(bold=True)
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 15
    ws.column_dimensions["C"].width = 55
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 10
    for row in data:
        ws.append([row["ficheiro"], row["ref"], row["produto"], row["qtd"], row["score"]])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="resultados.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_load_pipeline, daemon=True).start()
    PORT = int(os.environ.get("PORT", 5000 if sys.platform == "win32" else 5001))
    HOST = "0.0.0.0" if IS_CLOUD else "127.0.0.1"
    if not IS_CLOUD:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    from waitress import serve
    serve(app, host=HOST, port=PORT)
