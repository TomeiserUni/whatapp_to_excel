
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
import unicodedata
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


def _sem_acentos(texto: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", (texto or "").lower())
        if unicodedata.category(c) != "Mn"
    )


# Unidades que o cliente escreve mas não fazem parte do nome do produto
# (ex: "lâmpada 90 w" → o nome é "lâmpada led/uv 90"). São ignoradas na pesquisa.
_UNIDADES_PESQUISA = {"w", "watt", "watts", "g", "gr", "gramas", "grama", "ml", "mm", "cm"}

# A marca é removida dos nomes do catálogo (ver _normalizar_nome), por isso também
# tem de ser descartada do termo de pesquisa: colar "Verniz Gel Fada INOCOS" do
# site deve encontrar "verniz gel fada", não dar 0 resultados pelo filtro AND.
_MARCAS_PESQUISA = {"inocos", "hifans"}


def _palavras_pesquisa(termo: str) -> list[str]:
    """
    Palavras úteis do termo para o filtro AND: sem acentos, com a unidade
    separada do número ('90w'→'90') e as unidades/marca comuns descartadas.
    """
    palavras: list[str] = []
    for bruto in _sem_acentos(termo).split():
        # separa número colado a unidade: "90w" -> "90", "w"
        m = re.match(r"^(\d+)([a-z]+)$", bruto)
        partes = [m.group(1), m.group(2)] if m else [bruto]
        for parte in partes:
            if parte and parte not in _UNIDADES_PESQUISA and parte not in _MARCAS_PESQUISA:
                palavras.append(parte)
    return palavras


def _query_api(api_key: str, palavra: str) -> list[dict]:
    """
    Uma query ?q= à loja; devolve os produtos brutos (dicts) dessa palavra.
    Com retry/backoff em 429 (Too Many Requests): numa encomenda grande há muitas
    pesquisas seguidas e a Shopkit limita o ritmo; em vez de falhar a linha,
    espera-se um pouco e tenta-se de novo (até 3 vezes).
    """
    for tentativa in range(3):
        r = requests.get(
            f"{API_BASE}/product",
            headers={"X-API-KEY": api_key},
            params={"q": palavra, "limit": 100},
            timeout=15,
        )
        if r.status_code == 429 and tentativa < 2:
            espera = float(r.headers.get("Retry-After", 0)) or (0.5 * (tentativa + 1))
            time.sleep(espera)
            continue
        break
    r.raise_for_status()
    if not r.text.strip():
        return []
    body = r.json()
    return [v for k, v in body.items() if k.isdigit() and isinstance(v, dict)]


def pesquisar(api_key: str, termo: str, limit: int = 30) -> list[dict]:
    """
    Pesquisa ao vivo na loja (parâmetro ?q=) e filtra para devolver SÓ os
    produtos cujo nome contém todas as palavras do termo (AND, sem acentos).
    Remove o ruído da API (matches por descrição). Estado ignorado — esgotados
    incluídos. Devolve {nome, ref} normalizado como o catálogo.
    """
    termo = (termo or "").strip()
    if not termo:
        return []
    palavras = _palavras_pesquisa(termo)
    if not palavras:
        return []

    # A API pesquisa pela frase e limita resultados por palavra (uma palavra
    # comum como "super" pode truncar e esconder "super shine"). Por isso
    # consultamos por CADA palavra (as mais distintivas primeiro), juntamos os
    # candidatos e aplicamos o AND localmente sobre o nome.
    palavras_query = sorted(set(palavras), key=len, reverse=True)[:3]
    brutos: list[dict] = []
    ids_vistos: set = set()
    for palavra in palavras_query:
        for p in _query_api(api_key, palavra):
            pid = p.get("id") or p.get("title")
            if pid in ids_vistos:
                continue
            ids_vistos.add(pid)
            brutos.append(p)

    resultados: list[dict] = []
    vistos: set[str] = set()
    for p in brutos:
        if not _produto_vendavel(p):
            continue
        nome = _normalizar_nome(p.get("title", ""))
        if not nome or nome in vistos:
            continue
        nome_sa = _sem_acentos(nome)
        if not all(palavra in nome_sa for palavra in palavras):
            continue  # ruído: a palavra não está no nome
        vistos.add(nome)
        resultados.append({"produto": nome, "ref": (p.get("reference") or "").strip()})
        if len(resultados) >= limit:
            break
    return resultados


def pesquisar_em_catalogo(termo: str, nomes: list[str], sku_map: dict,
                          limit: int = 30) -> list[dict]:
    """
    Mesma filtragem AND da ``pesquisar`` (todas as palavras do termo no nome,
    sem acentos), mas sobre o catálogo já em RAM — SEM bater na API. Usada como
    fonte de candidatos por linha no processamento: evita uma chamada Shopkit por
    linha (e o 429 daí resultante), já que o catálogo completo foi carregado no
    arranque. Os ``nomes`` já vêm normalizados (mesmo formato de _normalizar_nome).
    """
    palavras = _palavras_pesquisa(termo or "")
    if not palavras:
        return []
    resultados: list[dict] = []
    for nome in nomes:
        nome_sa = _sem_acentos(nome)
        if all(palavra in nome_sa for palavra in palavras):
            resultados.append({"produto": nome, "ref": (sku_map.get(nome) or "").strip()})
            if len(resultados) >= limit:
                break
    return resultados


def _produto_vendavel(p: dict) -> bool:
    # O estado (active / out_of_stock) não importa: o stock é gerido depois,
    # no programa de faturação. Basta o produto existir na loja, ou seja, ter
    # referência. Excluem-se apenas itens sem referência (workshops, catálogo…)
    # e categorias da blocklist.
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


_CACHE_CATALOGO = DATA_DIR / "catalogo_cache.json"


def _guardar_cache(nomes: list[str], sku_map: dict[str, str]) -> None:
    """Salvaguarda do último catálogo bem-sucedido (gerida pelo programa)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_CATALOGO, "w", encoding="utf-8") as f:
            json.dump({"nomes": nomes, "sku_map": sku_map}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[shopkit] não consegui gravar cache do catálogo: {e}")


def _ler_cache() -> tuple[list[str], dict[str, str]] | None:
    try:
        with open(_CACHE_CATALOGO, encoding="utf-8") as f:
            d = json.load(f)
        return d.get("nomes", []), d.get("sku_map", {})
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[shopkit] cache do catálogo ilegível: {e}")
        return None


def carregar_catalogo(api_key: str) -> tuple[list[str], dict[str, str]]:
    """
    Catálogo completo da loja em runtime: uma chamada à API (fetch_todos), sem
    depender de ficheiros .pkl. Em sucesso, atualiza a cache em disco. Se a loja
    estiver indisponível, recorre à cache do último arranque bem-sucedido.
    Devolve (nomes, sku_map). Levanta RuntimeError se não houver API nem cache.
    """
    try:
        brutos = fetch_todos(api_key)
        nomes, sku_map, _meta = extrair_para_catalogo(brutos)
        if nomes:
            _guardar_cache(nomes, sku_map)
            print(f"[shopkit] catálogo carregado da loja: {len(nomes)} produtos.")
            return nomes, sku_map
        print("[shopkit] loja devolveu catálogo vazio; a tentar cache local.")
    except Exception as e:
        print(f"[shopkit] falha ao obter catálogo da loja ({e}); a tentar cache local.")

    cache = _ler_cache()
    if cache is not None:
        print(f"[shopkit] catálogo lido da cache: {len(cache[0])} produtos.")
        return cache
    raise RuntimeError("Sem catálogo: loja indisponível e sem cache local.")


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
    print(f"[shopkit] {len(nomes)} produtos válidos após filtros (com referência).")

    guardar(nomes, sku_map, meta)
    print(f"[shopkit] Guardado em {DATA_DIR}/{{prod.pkl, sku_map.pkl, prod_meta.json}}.")

    gerar_embeddings(nomes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
