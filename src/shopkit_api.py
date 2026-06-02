
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from html import unescape
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.shopk.it/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Categorias / títulos que não são produtos vendáveis no contexto da encomenda
_BLOCKLIST_CATEGORIAS = {"workshops inocos academy"}
_TAGS_HTML = re.compile(r"<[^>]+>")
_ESPACOS   = re.compile(r"\s+")


def _strip_html(texto: str | None) -> str:
    if not texto:
        return ""
    return _ESPACOS.sub(" ", unescape(_TAGS_HTML.sub(" ", texto))).strip()


def _normalizar_nome(titulo: str) -> str:
    """
    Mesma normalização do crawler: lowercase, remove a marca, e despoja
    sufixos de unidades coladas a números ('30g'→'30').
    """
    palavras = []
    for p in titulo.split():
        low = p.lower()
        if low in {"inocos", "hifans"}:
            continue
        if re.match(r"^\d+[a-zA-Z]", low):
            low = re.sub(r"^(\d+).*", r"\1", low)
        palavras.append(low)
    return " ".join(palavras)


def fetch_pagina(api_key: str, page: int, limit: int = 100) -> tuple[list[dict], int]:
    """Devolve (produtos_da_pagina, total_count)."""
    r = requests.get(
        f"{API_BASE}/product",
        headers={"X-API-KEY": api_key},
        params={"page": page, "limit": limit},
        timeout=30,
    )
    r.raise_for_status()
    total = int(r.headers.get("x-total-count", 0))
    body = r.json()
    # Resposta vem como {"0": {...}, "1": {...}, "paging": {...}}
    produtos = [v for k, v in body.items() if k.isdigit() and isinstance(v, dict)]
    return produtos, total


def fetch_todos(api_key: str, limit: int = 100) -> list[dict]:
    produtos: list[dict] = []
    page = 1
    while True:
        lote, total = fetch_pagina(api_key, page, limit)
        if not lote:
            break
        produtos.extend(lote)
        print(f"  [pag {page}] +{len(lote)} (acumulado {len(produtos)}/{total})")
        if len(produtos) >= total:
            break
        page += 1
        time.sleep(0.2)
    return produtos


def _produto_vendavel(p: dict) -> bool:
    if p.get("status_alias") != "active":
        return False
    if not (p.get("reference") or "").strip():
        return False
    categorias = {(c.get("title") or "").lower() for c in (p.get("categories") or [])}
    if categorias & _BLOCKLIST_CATEGORIAS:
        return False
    return True


def extrair_para_catalogo(produtos_api: list[dict]) -> tuple[list[str], dict[str, str], dict[str, dict]]:
    nomes: list[str] = []
    sku_map: dict[str, str] = {}
    meta: dict[str, dict] = {}

    vistos: set[str] = set()
    for p in produtos_api:
        if not _produto_vendavel(p):
            continue
        nome = _normalizar_nome(p.get("title", ""))
        if not nome or nome in vistos:
            continue
        vistos.add(nome)
        nomes.append(nome)
        sku_map[nome] = (p.get("reference") or "").strip()
        descricao = _strip_html(p.get("description_short") or p.get("excerpt") or p.get("description"))
        categorias = [c.get("title", "") for c in (p.get("categories") or []) if c.get("title")]
        if descricao or categorias:
            meta[nome] = {"descricao": descricao, "categorias": categorias}
    return nomes, sku_map, meta


def guardar(nomes: list[str], sku_map: dict[str, str], meta: dict[str, dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "prod.pkl", "wb") as f:
        pickle.dump(nomes, f)
    with open(DATA_DIR / "sku_map.pkl", "wb") as f:
        pickle.dump(sku_map, f)
    with open(DATA_DIR / "prod_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def gerar_embeddings(nomes: list[str]) -> None:
    """
    Re-gera data/emb_prod.npy para os novos nomes. Usa o model_finetuned
    se existir (mesma lógica do pipeline em runtime), senão all-MiniLM-L6-v2.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model_dir = DATA_DIR / "model_finetuned"
    base = str(model_dir) if model_dir.exists() else "all-MiniLM-L6-v2"
    print(f"[shopkit] A gerar embeddings com {base}…")
    model = SentenceTransformer(base)
    embs = model.encode(
        [n.lower() for n in nomes],
        convert_to_numpy=True,
        batch_size=64,
        show_progress_bar=True,
    )
    np.save(DATA_DIR / "emb_prod.npy", embs)
    print(f"[shopkit] emb_prod.npy actualizado ({len(nomes)} × {embs.shape[1]} dims).")


def main() -> int:
    api_key = os.environ.get("SHOPKIT_API_KEY")
    if not api_key:
        print("ERRO: SHOPKIT_API_KEY não definida no .env", file=sys.stderr)
        return 1

    print("[shopkit] A obter catálogo via API…")
    produtos_api = fetch_todos(api_key)
    print(f"[shopkit] {len(produtos_api)} entradas recebidas do API.")

    nomes, sku_map, meta = extrair_para_catalogo(produtos_api)
    print(f"[shopkit] {len(nomes)} produtos válidos após filtros (ativos, com referência).")

    guardar(nomes, sku_map, meta)
    print(f"[shopkit] Guardado em {DATA_DIR}/{{prod.pkl, sku_map.pkl, prod_meta.json}}.")

    gerar_embeddings(nomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
