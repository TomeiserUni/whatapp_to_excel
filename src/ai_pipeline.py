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


def _match_produto(nome_ai: str, produtos: list) -> tuple[str, float] | None:
    for p in produtos:
        if p.lower() == nome_ai.lower():
            return p, 1.0
    result = process.extractOne(nome_ai, produtos, scorer=fuzz.token_set_ratio)
    if result and result[1] >= 70:
        return result[0], result[1] / 100.0
    return None


def _parse_json(texto: str) -> list:
    # Tenta diretamente primeiro
    try:
        return json.loads(texto.strip())
    except json.JSONDecodeError:
        pass
    # Fallback: extrai o array do meio do texto (guloso para apanhar arrays grandes)
    match = re.search(r"\[.*\]", texto, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _candidatos_por_linha(linha: str, produtos: list, limit: int = 20) -> list:
    """Seleciona os melhores candidatos para uma linha de texto."""
    top = process.extract(linha, produtos, scorer=fuzz.token_set_ratio, limit=limit)
    return [r[0] for r in top] if top else []


def _extrair_texto_imagem(image_path: Path, client) -> str:
    """OCR puro: Claude transcreve o texto visível na imagem."""
    ext  = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": img_b64}
                },
                {
                    "type": "text",
                    "text": (
                        "Transcreve APENAS o texto visível nesta imagem, linha por linha.\n"
                        "Não interpretes nem traduz — copia exatamente o que está escrito.\n"
                        "Inclui números e quantidades tal como aparecem."
                    )
                }
            ]}]
        )
        return r.content[0].text.strip()
    except Exception as e:
        print(f"[AI OCR] erro: {e}")
        return ""


def processar_imagem(image_path: Path, produtos: list, sku_map: dict, aliases: dict, client) -> list:
    texto = _extrair_texto_imagem(image_path, client)
    if not texto:
        return []
    print(f"[AI OCR] texto extraído: {texto[:200]}")
    return processar_texto(texto, produtos, sku_map, aliases, client)


def processar_texto(texto: str, produtos: list, sku_map: dict, aliases: dict, client) -> list:
    """
    Para cada linha, seleciona candidatos com rapidfuzz.
    Envia texto + candidatos ao Claude para matching final.
    """
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    if not linhas:
        return []

    # Candidatos por linha (linha a linha evita diluição de score)
    candidatos_set: set[str] = set()
    for linha in linhas:
        candidatos_set.update(_candidatos_por_linha(linha, produtos, limit=20))

    # Aliases diretos têm sempre prioridade — garante que entram no catálogo
    if aliases:
        texto_lower = texto.lower()
        for alias, produto in aliases.items():
            if alias.lower() in texto_lower and produto in produtos:
                candidatos_set.add(produto)

    candidatos = list(candidatos_set)[:60]
    catalogo   = _build_catalogo(candidatos, sku_map)
    alias_txt  = _build_aliases_txt(aliases, sku_map)

    prompt = (
        f"Catálogo de produtos de vernizes/unhas:\n{catalogo}\n"
        f"{alias_txt}\n"
        f"Mensagem de encomenda:\n{texto}\n\n"
        "Identifica TODOS os produtos mencionados e as suas quantidades.\n"
        "Usa o nome EXATO do catálogo acima — não inventes nomes.\n"
        "Se um produto não estiver no catálogo, ignora-o.\n"
        'Responde APENAS com JSON válido (sem texto antes ou depois):\n'
        '[{"produto": "nome exato do catálogo", "quantidade": número}, ...]\n'
        "Se não houver produtos, responde []."
    )

    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = r.content[0].text
        print(f"[AI texto] resposta: {raw[:300]}")
        items = _parse_json(raw)
    except Exception as e:
        print(f"[AI texto] erro: {e}")
        return []

    resultados = []
    for item in items:
        match = _match_produto(item.get("produto", ""), produtos)
        if match:
            produto_real, score = match
            resultados.append((produto_real, score, int(item.get("quantidade", 1))))
    return resultados
