import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

# ── Caminhos ────────────────────────────────────────────────────
if getattr(sys, "frozen", False):
    _BUNDLE   = Path(sys._MEIPASS)
    _USER_DIR = Path.home() / "WhatsAppExcel"
else:
    _BUNDLE   = Path(__file__).resolve().parent.parent
    _USER_DIR = _BUNDLE

sys.path.insert(0, str(_BUNDLE / "src"))

OUTPUT_DIR = _USER_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask ────────────────────────────────────────────────────────
template_folder = str(_BUNDLE / "src" / "templates")
app = Flask(__name__, template_folder=template_folder)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

_pipeline = None


def _load_pipeline():
    global _pipeline
    import pipeline as pl
    produtos, emb_prod, sku_map = pl.load_produtos()
    _pipeline = {"pl": pl, "produtos": produtos,
                 "emb_prod": emb_prod, "sku_map": sku_map}


# ── Rotas ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({"ready": _pipeline is not None})


@app.route("/processar", methods=["POST"])
def processar():
    if _pipeline is None:
        return jsonify({"error": "Modelos ainda a carregar…"}), 503

    files = request.files.getlist("imagens")
    pl = _pipeline
    resultados = []

    for f in files:
        tmp = OUTPUT_DIR / f.filename
        f.save(tmp)
        try:
            rows = pl["pl"].processar_imagem(tmp, pl["produtos"], pl["emb_prod"])
            for produto, score, qty in rows:
                resultados.append({
                    "ficheiro": f.filename,
                    "produto":  produto,
                    "qtd":      qty,
                    "score":    round(score, 3),
                    "ref":      pl["sku_map"].get(produto, ""),
                })
        finally:
            tmp.unlink(missing_ok=True)

    return jsonify(resultados)


@app.route("/processar_texto", methods=["POST"])
def processar_texto():
    if _pipeline is None:
        return jsonify({"error": "Modelos ainda a carregar…"}), 503

    texto = (request.json or {}).get("texto", "").strip()
    if not texto:
        return jsonify([])

    pl = _pipeline
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
    path = OUTPUT_DIR / "resultados.xlsx"
    wb.save(path)
    return send_file(path, as_attachment=True, download_name="resultados.xlsx")


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_load_pipeline, daemon=True).start()
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(port=5000, debug=False, use_reloader=False)
