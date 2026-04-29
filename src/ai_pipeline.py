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
    try:
        with open(data_dir / "aliases.json") as f:
            aliases = json.load(f)
    except FileNotFoundError:
        aliases = {}
    return produtos, sku_map, aliases


def _build_catalogo(candidatos: list, sku_map: dict) -> str:
    return "\n".join(f"- {p} (REF: {sku_map.get(p, 'N/D')})" for p in candidatos)


def _build_aliases_txt(aliases: dict, sku_map: dict) -> str:
    if not aliases:
        return ""
    lines = "\n".join(
        f'  "{k}" → {v} (REF: {sku_map.get(v, "N/D")})'
        for k, v in aliases.items()
    )
    return f"\nNomes alternativos conhecidos:\n{lines}\n"


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


def _extrair_texto_imagem(image_path: Path, client) -> str:
    """Passo 1: usa o modelo de visão apenas para OCR (sem catálogo)."""
    ext  = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    try:
        r = client.chat.completions.create(
            model="meta/llama-3.2-11b-vision-instruct",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": (
                    "Esta imagem é uma mensagem WhatsApp com uma encomenda de produtos.\n"
                    "Transcreve APENAS o texto visível na imagem, linha por linha.\n"
                    "Não interpretes nem traduz — copia exatamente o que está escrito."
                )},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
            ]}],
            max_tokens=512,
            temperature=0,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[AI OCR] erro: {e}")
        return ""


def processar_imagem(image_path: Path, produtos: list, sku_map: dict, aliases: dict, client) -> list:
    """Passo 1: OCR com visão. Passo 2: matching igual ao texto."""
    texto = _extrair_texto_imagem(image_path, client)
    if not texto:
        return []
    print(f"[AI OCR] texto extraído: {texto[:200]}")
    return processar_texto(texto, produtos, sku_map, aliases, client)


def processar_texto(texto: str, produtos: list, sku_map: dict, aliases: dict, client) -> list:
    """Pré-filtra com rapidfuzz → envia só os 40 melhores candidatos para a AI."""
    top = process.extract(texto, produtos, scorer=fuzz.token_set_ratio, limit=40)
    candidatos = [r[0] for r in top] if top else produtos[:40]

    catalogo = _build_catalogo(candidatos, sku_map)
    alias_txt = _build_aliases_txt(aliases, sku_map)

    prompt = (
        f"Catálogo de produtos de vernizes/unhas (nome + referência):\n{catalogo}\n"
        f"{alias_txt}\n"
        f"Mensagem de encomenda:\n{texto}\n\n"
        "Identifica os produtos e quantidades. Usa o nome EXATO do catálogo.\n"
        'Responde APENAS com JSON válido: [{"produto": "nome exato do catálogo", "referencia": "REF", "quantidade": número}]\n'
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
