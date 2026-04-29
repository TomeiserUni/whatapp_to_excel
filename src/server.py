import io
import os
import re
import secrets
import sys
import tempfile
import threading
import unicodedata
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


def _int_env(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _normalizar_token(texto: str) -> str:
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower())
        if unicodedata.category(c) != "Mn"
    )
    return texto.strip()


def _linha_so_contexto(linha: str) -> bool:
    lower = linha.lower().strip()
    lower_sem_qtd = re.sub(r"\b\d+\b", "", lower)
    lower_sem_qtd = re.sub(r"\b(?:unidades?|un|pcs?|cada)\b", "", lower_sem_qtd).strip()
    return (
        lower in {"de cada", "verniz normal", "verniz gel"}
        or lower.startswith(("bom dia", "boa tarde", "boa noite"))
        or lower.startswith(("encomenda ", "pedido "))
        or "cores novas" in lower
        or lower_sem_qtd in {"cores novas", "verniz normal", "verniz gel"}
    )


def _linha_origem(linhas: list[str], trecho: str) -> str:
    trecho = (trecho or "").strip()
    if not trecho:
        return ""
    trecho_lower = trecho.lower()
    for idx, linha in enumerate(linhas, 1):
        if trecho_lower == linha.lower() or trecho_lower in linha.lower():
            return f"linha {idx}: {linha}"
    return trecho


def _linha_idx(linhas: list[str], trecho: str) -> int | None:
    trecho = (trecho or "").strip()
    if not trecho:
        return None
    trecho_lower = trecho.lower()
    for idx, linha in enumerate(linhas, 1):
        if trecho_lower == linha.lower() or trecho_lower in linha.lower():
            return idx
    return None


def _erro_linha(idx: int, linha: str) -> dict:
    return {
        "ficheiro":     "texto colado",
        "produto":      "",
        "qtd":          "",
        "score":        0,
        "ref":          "",
        "texto_origem": f"linha {idx}: {linha}",
    }


def _intercalar_erros_linhas(linhas: list[str], resultados: list[dict]) -> list[dict]:
    por_linha: dict[int, list[dict]] = {}
    sem_linha: list[dict] = []

    for row in resultados:
        idx = row.pop("_linha_idx", None)
        if idx is None:
            sem_linha.append(row)
            continue
        por_linha.setdefault(idx, []).append(row)

    final = []
    for idx, linha in enumerate(linhas, 1):
        rows = por_linha.get(idx)
        if rows:
            final.extend(rows)
        else:
            final.append(_erro_linha(idx, linha))
    final.extend(sem_linha)
    return final


def _dedupe_resultados_por_linha(resultados: list[dict]) -> list[dict]:
    por_linha: dict[int, dict] = {}
    sem_linha: list[dict] = []

    for row in resultados:
        idx = row.get("_linha_idx")
        if idx is None:
            sem_linha.append(row)
            continue
        atual = por_linha.get(idx)
        if atual is None or float(row.get("score", 0)) > float(atual.get("score", 0)):
            por_linha[idx] = row

    return list(por_linha.values()) + sem_linha


def _dedupe_resultados_por_linha_ou_de_cada(resultados: list[dict], linhas: list[str]) -> list[dict]:
    por_linha: dict[int, dict[str, dict]] = {}
    sem_linha: list[dict] = []

    for row in resultados:
        idx = row.get("_linha_idx")
        if idx is None:
            sem_linha.append(row)
            continue

        produtos_linha = por_linha.setdefault(idx, {})
        produto = row.get("produto", "")
        atual = produtos_linha.get(produto)
        if atual is None or float(row.get("score", 0)) > float(atual.get("score", 0)):
            produtos_linha[produto] = row

    final = []
    for produtos_linha in por_linha.values():
        final.extend(produtos_linha.values())
    final.extend(sem_linha)
    return final


def _quantidade_de_cada(linha: str) -> int:
    antes = linha.lower().split("de cada", 1)[0]
    nums = re.findall(r"\b\d+\b", antes)
    return int(nums[-1]) if nums else 1


def _quantidade_linha(linha: str) -> int:
    m = re.search(r"\b(\d+)\s*(?:unidades?|un|und|pcs?)\b", linha, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\bx\s*(\d+)\b|\b(\d+)\s*x\b", linha, re.IGNORECASE)
    if m:
        return int(m.group(1) or m.group(2))
    nums = [
        m.group(1)
        for m in re.finditer(r"\b(\d+)\b", linha)
        if not re.match(r"\s*(?:g|gr|gramas?|ml)\b", linha[m.end():], re.IGNORECASE)
    ]
    return int(nums[-1]) if nums else 1


def _token_e_quantidade(linha: str, match: re.Match) -> bool:
    after = linha[match.end():match.end() + 12]
    before = linha[max(0, match.start() - 2):match.start()]
    return bool(
        re.match(r"\s*(?:unidades?|un|und|pcs?|cada)\b", after, re.IGNORECASE)
        or re.search(r"x\s*$", before, re.IGNORECASE)
    )


def _mapa_tokens_unicos(produtos: list[str]) -> dict[str, str]:
    token_produto: dict[str, str] = {}
    repetidos: set[str] = set()

    for produto in produtos:
        for token in set(re.findall(r"\w+", _normalizar_token(produto))):
            if len(token) < 3 and not token.isdigit():
                continue
            if token in token_produto and token_produto[token] != produto:
                repetidos.add(token)
                continue
            token_produto[token] = produto

    return {token: produto for token, produto in token_produto.items() if token not in repetidos}


def _expandir_identificadores_unicos(linhas: list[str], produtos: list[str], sku_map: dict) -> list[dict]:
    unicos = _mapa_tokens_unicos(produtos)
    resultados = []

    for idx, linha in enumerate(linhas, 1):
        if _linha_so_contexto(linha):
            continue

        produtos_linha: dict[str, str] = {}
        linha_norm = _normalizar_token(linha)
        for match in re.finditer(r"\w+", linha_norm):
            token = match.group(0)
            if token.isdigit() and _token_e_quantidade(linha_norm, match):
                continue
            produto = unicos.get(token)
            if produto:
                produtos_linha[produto] = token

        if not produtos_linha:
            continue

        qty = _quantidade_linha(linha)
        for produto in produtos_linha:
            resultados.append({
                "ficheiro":     "texto colado",
                "produto":      produto,
                "qtd":          qty,
                "score":        1.0,
                "ref":          sku_map.get(produto, ""),
                "texto_origem": f"linha {idx}: {linha}",
                "_linha_idx":   idx,
            })

    return resultados


def _expandir_aliases_diretos(linhas: list[str], aliases: dict, sku_map: dict) -> list[dict]:
    resultados = []
    aliases_ordenados = sorted(
        ((alias, produto) for alias, produto in (aliases or {}).items() if produto),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for idx, linha in enumerate(linhas, 1):
        if _linha_so_contexto(linha):
            continue

        linha_norm = _normalizar_token(linha)
        for alias, produto in aliases_ordenados:
            alias_norm = _normalizar_token(alias)
            if not alias_norm:
                continue
            if not re.search(rf"(?<!\w){re.escape(alias_norm)}(?!\w)", linha_norm):
                continue

            resultados.append({
                "ficheiro":     "texto colado",
                "produto":      produto,
                "qtd":          _quantidade_linha(linha),
                "score":        1.0,
                "ref":          sku_map.get(produto, ""),
                "texto_origem": f"linha {idx}: {linha}",
                "_linha_idx":   idx,
            })
            break

    return resultados


def _expandir_regras_de_cada(linhas: list[str], sku_map: dict) -> list[dict]:
    produtos_moldes_f1 = [
        "moldes f1 quadrado",
        "silicones para moldes f1",
        "moldes f1 amendoa russa",
    ]

    resultados = []
    for idx, linha in enumerate(linhas, 1):
        lower = linha.lower()
        if "moldes f1" not in lower or "de cada" not in lower:
            continue

        qty = _quantidade_de_cada(linha)
        for produto in produtos_moldes_f1:
            resultados.append({
                "ficheiro":     "texto colado",
                "produto":      produto,
                "qtd":          qty,
                "score":        1.0,
                "ref":          sku_map.get(produto, ""),
                "texto_origem": f"linha {idx}: {linha}",
                "_linha_idx":   idx,
            })
    return resultados


def _melhor_linha_origem(produto: str, linhas: list[str], fuzz_mod, todos_produtos: list[str] = None) -> str:
    genericas = {
        "verniz", "gel", "normal", "unidade", "unidades", "un", "pcs", "cada",
        "base", "top", "coat", "ml", "g", "gr",
    }
    if todos_produtos:
        token_counts: dict[str, int] = {}
        for p in todos_produtos:
            for t in re.findall(r"\w+", p.lower()):
                if len(t) >= 3 and not t.isdigit():
                    token_counts[t] = token_counts.get(t, 0) + 1
        genericas = genericas | {t for t, c in token_counts.items() if c > 1}
    produto_lower = produto.lower()
    produto_tokens = set(re.findall(r"\w+", produto_lower))
    distintivas = {t for t in produto_tokens if t not in genericas and not t.isdigit()}
    numeros_produto = {t for t in produto_tokens if t.isdigit()}

    melhor_linha = ""
    melhor_score = -1.0
    for linha in linhas:
        if _linha_so_contexto(linha):
            continue
        linha_lower = linha.lower()
        linha_tokens = set(re.findall(r"\w+", linha_lower))
        score = fuzz_mod.partial_ratio(produto_lower, linha_lower)
        score += fuzz_mod.token_set_ratio(produto_lower, linha_lower) * 0.5
        score += len(distintivas & linha_tokens) * 80
        score += len(numeros_produto & linha_tokens) * 120
        if score > melhor_score:
            melhor_linha = linha
            melhor_score = score

    return melhor_linha


def _enable_ai_pipeline(api_key: str) -> None:
    if _pipeline is None:
        return

    import anthropic
    import ai_pipeline as ai_pl

    client = anthropic.Anthropic(api_key=api_key)
    ai_prods, ai_sku, ai_aliases, exemplos = ai_pl.load_produtos(_BUNDLE / "data")
    _pipeline.update({
        "ai_pl":      ai_pl,
        "ai_client":  client,
        "ai_produtos": ai_prods,
        "ai_sku_map": ai_sku,
        "ai_aliases": ai_aliases,
        "exemplos":   exemplos,
    })


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
        import ai_pipeline as ai_pl
        client = anthropic.Anthropic(api_key=api_key)
        produtos, sku_map, aliases, exemplos = ai_pl.load_produtos(_BUNDLE / "data")
        _pipeline = {"pl": ai_pl, "ai_pl": ai_pl, "produtos": produtos, "emb_prod": None,
                     "sku_map": sku_map, "aliases": aliases, "exemplos": exemplos, "ai_client": client}
    else:
        # Local: pipeline ML para imagens (EasyOCR + embeddings)
        import pipeline as pl
        produtos, emb_prod, sku_map, freq_palavras, palavras_unicas, aliases = pl.load_produtos()
        _pipeline = {"pl": pl, "produtos": produtos, "emb_prod": emb_prod,
                     "sku_map": sku_map, "aliases": aliases, "ai_client": None,
                     "freq_palavras": freq_palavras, "palavras_unicas": palavras_unicas,
                     "ai_pl": None, "exemplos": []}

        # Se tiver API key: carrega também ai_pipeline para texto (muito mais rápido)
        if api_key:
            import anthropic
            import ai_pipeline as ai_pl
            client = anthropic.Anthropic(api_key=api_key)
            ai_prods, ai_sku, ai_aliases, exemplos = ai_pl.load_produtos(_BUNDLE / "data")
            pl.init_ai_client(api_key)
            _pipeline.update({
                "ai_pl":      ai_pl,
                "ai_client":  client,
                "ai_produtos": ai_prods,
                "ai_sku_map": ai_sku,
                "ai_aliases": ai_aliases,
                "exemplos":   exemplos,
            })
            print("[AI] Claude ativo para texto (batches paralelos).")

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

        if buf and l.lower() == "de cada":
            buf = f"{buf} de cada"
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
            _enable_ai_pipeline(key)

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

    ai_pl     = pl.get("ai_pl")
    ai_client = pl.get("ai_client")
    use_ai    = bool(ai_pl and ai_client)

    if use_ai:
        # Batches paralelos com Claude (cloud ou local com API key)
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
        from rapidfuzz import fuzz as _rfuzz

        prods_t   = pl["ai_produtos"] if not IS_CLOUD else pl["produtos"]
        sku_t     = pl["ai_sku_map"]  if not IS_CLOUD else pl["sku_map"]
        aliases_t = pl["ai_aliases"]  if not IS_CLOUD else pl["aliases"]
        exemplos_t = pl.get("exemplos", [])

        batch_size = _int_env("AI_BATCH_LINES", 60, minimum=10, maximum=120)
        max_workers = _int_env("AI_MAX_WORKERS", 8, minimum=1, maximum=12)
        batches = _chunks(linhas, batch_size)
        print(f"[AI texto] {len(linhas)} linhas em {len(batches)} lote(s) de até {batch_size}.")

        raw_rows: list[tuple] = []
        batch_errors: list[str] = []
        with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as pool:
            futs = {
                pool.submit(
                    ai_pl.processar_texto,
                    "\n".join(b),
                    prods_t, sku_t, aliases_t, ai_client, exemplos_t
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
                    batch_errors.append(str(e))

        if batch_errors and len(batch_errors) == len(batches):
            return jsonify({"error": "Erro ao processar com AI. Verifica a API key ou tenta novamente."}), 502

        melhor: dict[str, tuple] = {}
        for row, b in raw_rows:
            p = row[0]
            if p not in melhor or row[1] > melhor[p][0][1]:
                melhor[p] = (row, b)

        sku_final = pl.get("ai_sku_map") or pl["sku_map"]
        resultados.extend(_expandir_aliases_diretos(linhas, aliases_t, sku_final))
        resultados.extend(_expandir_identificadores_unicos(linhas, prods_t, sku_final))
        resultados.extend(_expandir_regras_de_cada(linhas, sku_final))
        if melhor:
            for p, (row, b) in melhor.items():
                _, s, q = row[0], row[1], row[2]
                ai_linha_idx = row[3] if len(row) > 3 else None
                if ai_linha_idx and 1 <= ai_linha_idx <= len(linhas):
                    linha_idx = ai_linha_idx
                    texto_orig = f"linha {ai_linha_idx}: {linhas[ai_linha_idx - 1]}"
                else:
                    origem = _melhor_linha_origem(p, b, _rfuzz, prods_t)
                    linha_idx = _linha_idx(linhas, origem)
                    texto_orig = _linha_origem(linhas, origem)
                resultados.append({
                    "ficheiro":     "texto colado",
                    "produto":      p,
                    "qtd":          q,
                    "score":        round(s, 3),
                    "ref":          sku_final.get(p, ""),
                    "texto_origem": texto_orig,
                    "_linha_idx":   linha_idx,
                })
            resultados = _intercalar_erros_linhas(linhas, _dedupe_resultados_por_linha_ou_de_cada(resultados, linhas))
        else:
            if not IS_CLOUD and pl.get("emb_prod") is not None:
                texto_proc = "\n".join(linhas)
                rows = pl["pl"].processar_texto(texto_proc, pl["produtos"], pl["emb_prod"],
                                                 pl.get("freq_palavras"), pl.get("palavras_unicas"))
                for row in rows:
                    p, s, q = row[0], row[1], row[2]
                    trecho = row[3] if len(row) > 3 else ""
                    resultados.append({
                        "ficheiro":     "texto colado",
                        "produto":      p,
                        "qtd":          q,
                        "score":        round(s, 3),
                        "ref":          pl["sku_map"].get(p, ""),
                        "texto_origem": _linha_origem(linhas, trecho),
                        "_linha_idx":   _linha_idx(linhas, trecho),
                    })
                resultados = _intercalar_erros_linhas(linhas, _dedupe_resultados_por_linha_ou_de_cada(resultados, linhas))
            elif resultados:
                resultados = _intercalar_erros_linhas(linhas, _dedupe_resultados_por_linha_ou_de_cada(resultados, linhas))
            if not resultados:
                return jsonify(_intercalar_erros_linhas(linhas, []))
    else:
        # Sem API key: pipeline local ML (lento para ordens grandes)
        texto_proc = "\n".join(linhas)
        rows = pl["pl"].processar_texto(texto_proc, pl["produtos"], pl["emb_prod"],
                                         pl.get("freq_palavras"), pl.get("palavras_unicas"))
        resultados.extend(_expandir_aliases_diretos(linhas, pl.get("aliases", {}), pl["sku_map"]))
        resultados.extend(_expandir_identificadores_unicos(linhas, pl["produtos"], pl["sku_map"]))
        resultados.extend(_expandir_regras_de_cada(linhas, pl["sku_map"]))
        if rows:
            for row in rows:
                p, s, q = row[0], row[1], row[2]
                trecho = row[3] if len(row) > 3 else ""
                resultados.append({
                    "ficheiro":     "texto colado",
                    "produto":      p,
                    "qtd":          q,
                    "score":        round(s, 3),
                    "ref":          pl["sku_map"].get(p, ""),
                    "texto_origem": _linha_origem(linhas, trecho),
                    "_linha_idx":   _linha_idx(linhas, trecho),
                })
            resultados = _intercalar_erros_linhas(linhas, _dedupe_resultados_por_linha_ou_de_cada(resultados, linhas))
        else:
            return jsonify(_intercalar_erros_linhas(linhas, _dedupe_resultados_por_linha_ou_de_cada(resultados, linhas)))

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
    ws.append(["Referência", "Produto", "Quantidade", "Onde foi identificado", "Texto reconhecido"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.column_dimensions["A"].width = 15
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 34
    ws.column_dimensions["E"].width = 60

    for row in data:
        produto = row.get("produto", "")
        score   = float(row.get("score", 0))
        texto   = row.get("texto_origem", "")
        ficheiro = row.get("ficheiro", "")
        onde = ficheiro
        if texto:
            onde = f"{ficheiro} - {texto}" if ficheiro else texto
        label   = produto if produto else "ERRO — produto não identificado"
        ws.append([row.get("ref", ""), label, row.get("qtd", ""), onde, texto])
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
