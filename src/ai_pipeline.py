import base64
import json
import pickle
import re
from pathlib import Path

from rapidfuzz import fuzz, process


def load_produtos(data_dir: Path):
    with open(data_dir / "prod.pkl", "rb") as f:
        produtos = pickle.load(f)
    try:
        with open(data_dir / "sku_map.pkl", "rb") as f:
            sku_map = pickle.load(f)
    except FileNotFoundError:
        sku_map = {}
    return produtos, sku_map


def _match_produto(nome_ai: str, produtos: list) -> str | None:
    for p in produtos:
        if p.lower() == nome_ai.lower():
            return p
    result = process.extractOne(nome_ai, produtos, scorer=fuzz.token_sort_ratio)
    if result and result[1] >= 70:
        return result[0]
    return None


def _parse_json(texto: str) -> list:
    match = re.search(r"\[.*?\]", texto, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def processar_imagem(image_path: Path, produtos: list, client) -> list:
    ext  = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    catalogo = "\n".join(f"- {p}" for p in produtos)
    prompt = (
        "Esta imagem é uma mensagem WhatsApp com uma encomenda de vernizes/produtos de unhas.\n\n"
        f"Catálogo disponível:\n{catalogo}\n\n"
        "Identifica os produtos encomendados e as quantidades.\n"
        'Responde APENAS com JSON válido: [{"produto": "nome exato do catálogo", "quantidade": número}]\n'
        "Se não houver produtos reconhecíveis, responde []."
    )

    try:
        r = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]}],
            max_tokens=1024,
            temperature=0,
        )
        items = _parse_json(r.choices[0].message.content)
    except Exception as e:
        print(f"[AI imagem] erro: {e}")
        return []

    return [
        (produto_real, 1.0, int(item.get("quantidade", 1)))
        for item in items
        if (produto_real := _match_produto(item.get("produto", ""), produtos))
    ]


def processar_texto(texto: str, produtos: list, client) -> list:
    catalogo = "\n".join(f"- {p}" for p in produtos)
    prompt = (
        f"Catálogo de produtos de vernizes/unhas:\n{catalogo}\n\n"
        f"Mensagem de encomenda:\n{texto}\n\n"
        "Identifica os produtos e quantidades.\n"
        'Responde APENAS com JSON válido: [{"produto": "nome exato do catálogo", "quantidade": número}]\n'
        "Se não houver produtos, responde []."
    )

    try:
        r = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
        )
        items = _parse_json(r.choices[0].message.content)
    except Exception as e:
        print(f"[AI texto] erro: {e}")
        return []

    return [
        (produto_real, 1.0, int(item.get("quantidade", 1)))
        for item in items
        if (produto_real := _match_produto(item.get("produto", ""), produtos))
    ]
