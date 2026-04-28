import re
import time
import random
import csv
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL  = "https://www.inocos.com"
START_URL = "https://www.inocos.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

_REMOVER = {"inocos", "hifans"}


def limpar_slug(slug):
    """'verniz-gel-inocos-agenda-cheia' → 'verniz gel agenda cheia'"""
    partes = slug.replace("-", " ").split()
    resultado = []
    for p in partes:
        if p.lower() in _REMOVER:
            continue
        # "30g" → "30", "15ml" → "15"
        if re.match(r"^\d+[a-zA-Z]", p):
            p = re.sub(r"^(\d+).*", r"\1", p)
        resultado.append(p.lower())
    return " ".join(resultado)


def extrair_links_categorias(soup):
    categorias = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/category/" in href:
            full = urljoin(BASE_URL, href)
            slug_categoria = full.split("/category/")[1].split("/")[0]
            if slug_categoria:
                categorias.add(slug_categoria)
    return categorias


def extrair_links_produtos(soup):
    produtos = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/product/" in href:
            full = urljoin(BASE_URL, href)
            slug = full.split("/product/")[1].split("?")[0].strip("/")
            if slug:
                produtos.add(slug)
    return produtos


def encontrar_proxima_pagina(soup):
    """Paginação do Shopkit: /category/xxx/12, /category/xxx/24, ...
    O link '>' aponta para a página seguinte."""
    pag_div = soup.find("div", class_="pagination")
    if not pag_div:
        return None
    for li in pag_div.find_all("li"):
        a = li.find("a")
        if a and a.get_text(strip=True) == ">":
            href = a.get("href")
            if href and href != "#":
                return urljoin(BASE_URL, href)
    return None


def extrair_sku(session, slug):
    """Extrai a referência do produto a partir de <span class="sku">REF</span>."""
    url = f"{BASE_URL}/product/{slug}"
    try:
        r = session.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        sku_el = soup.find(class_="sku")
        return sku_el.get_text(strip=True) if sku_el else None
    except Exception as e:
        print(f"  [ERRO SKU] {slug}: {e}")
        return None


def crawl_categoria(session, slug_categoria):
    url = f"{BASE_URL}/category/{slug_categoria}"
    produtos_categoria = set()

    while url:
        try:
            print(f"  [A visitar] {url}")
            r = session.get(url, headers=HEADERS, timeout=10)

            if r.status_code != 200:
                print(f"  [ERRO HTTP] {r.status_code}")
                break

            soup = BeautifulSoup(r.text, "html.parser")
            produtos = extrair_links_produtos(soup)
            produtos_categoria.update(produtos)
            url = encontrar_proxima_pagina(soup)

            time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            print(f"  [ERRO] {e}")
            break

    return produtos_categoria


def crawl():
    session = requests.Session()
    todos_slugs = set()

    print(f"[A recolher categorias de] {START_URL}")
    try:
        r = session.get(START_URL, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"[ERRO HTTP] homepage devolveu {r.status_code}")
            return []
    except Exception as e:
        print(f"[ERRO] Não foi possível aceder a {START_URL}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    categorias = extrair_links_categorias(soup)

    print(f"[CATEGORIAS] {len(categorias)} encontradas\n")

    for slug_cat in sorted(categorias):
        print(f"\n[CATEGORIA] {slug_cat}")
        produtos = crawl_categoria(session, slug_cat)
        novos = produtos - todos_slugs
        todos_slugs.update(produtos)
        print(f"  [SUBTOTAL] {len(produtos)} produtos, {len(novos)} novos")

    print(f"\n[TOTAL] {len(todos_slugs)} produtos únicos. A recolher SKUs...\n")

    slugs_sorted = sorted(todos_slugs)
    total = len(slugs_sorted)
    resultados = [None] * total

    def fetch_sku(args):
        i, slug = args
        sku  = extrair_sku(session, slug)
        nome = limpar_slug(slug)
        print(f"  [{i+1}/{total}] {nome} → {sku or 'N/A'}")
        return i, {"nome": nome, "slug": slug, "sku": sku or ""}

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_sku, (i, s)): i for i, s in enumerate(slugs_sorted)}
        for future in as_completed(futures):
            i, row = future.result()
            resultados[i] = row

    return resultados


# -----------------------------
# RUN
# -----------------------------
resultados = crawl()

output_file = "/Users/tomassilveira/Downloads/inocos_produtos.csv"
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["nome", "slug", "sku"])
    writer.writeheader()
    writer.writerows(resultados)

print(f"\n{len(resultados)} produtos guardados em {output_file}")
