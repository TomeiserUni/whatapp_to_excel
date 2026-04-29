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
    try:
        with open(data_dir / "exemplos.json") as f:
            exemplos = json.load(f)
    except FileNotFoundError:
        exemplos = []
    return produtos, sku_map, aliases, exemplos


def _build_catalogo(candidatos: list, sku_map: dict) -> str:
    return "\n".join(f"- {p} (REF: {sku_map.get(p, 'N/D')})" for p in candidatos)



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


_SINONIMOS = {"direita": "reta", "diretas": "retas", "direto": "reto", "moon": "meia lua", "moons": "meia lua"}

def _normalizar_linha(linha: str) -> str:
    for orig, sub in _SINONIMOS.items():
        linha = re.sub(rf'\b{orig}\b', sub, linha, flags=re.IGNORECASE)
    return linha

def _candidatos_por_linha(linha: str, produtos: list, limit: int = 20) -> list:
    """Seleciona os melhores candidatos para uma linha de texto."""
    linha_norm = _normalizar_linha(linha)
    top = process.extract(linha_norm, produtos, scorer=fuzz.token_set_ratio, limit=limit)
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


def _construir_base_rag(exemplos: list, aliases: dict) -> list[dict]:
    """Constrói a base de conhecimento: todos os pares escrito→produto."""
    base = []
    for escrito, produto in (aliases or {}).items():
        base.append({"escrito": escrito, "produto": produto})
    for e in exemplos:
        escrito = e.get("escrito", "")
        if "produtos" in e:
            produto = ", ".join(e["produtos"])
        else:
            produto = e.get("produto", "")
        if escrito and produto:
            base.append({"escrito": escrito, "produto": produto})
    return base


def _recuperar_exemplos(texto: str, base_rag: list[dict], top_k: int = 12) -> str:
    """RAG: recupera os exemplos mais relevantes para o texto atual."""
    if not base_rag:
        return ""
    scored = sorted(
        base_rag,
        key=lambda e: fuzz.partial_ratio(e["escrito"].lower(), texto.lower()),
        reverse=True
    )
    relevantes = scored[:top_k]
    lines = [f'  "{e["escrito"]}" → {e["produto"]}' for e in relevantes]
    return "\nExemplos de correspondências anteriores (mais relevantes para este pedido):\n" + "\n".join(lines) + "\n"


def processar_imagem(image_path: Path, produtos: list, sku_map: dict, aliases: dict, client, exemplos: list = None) -> list:
    texto = _extrair_texto_imagem(image_path, client)
    if not texto:
        return []
    print(f"[AI OCR] texto extraído: {texto[:200]}")
    return processar_texto(texto, produtos, sku_map, aliases, client, exemplos or [])


def processar_texto(texto: str, produtos: list, sku_map: dict, aliases: dict, client, exemplos: list = None) -> list:
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
        # Se "de cada", garante que TODAS as variantes do padrão entram no catálogo
        linha_norm = _normalizar_linha(linha.lower())
        if "de cada" in linha_norm:
            prefixo = re.sub(r'\b\d+\b', '', linha_norm.split("de cada")[0]).strip()
            top_todos = process.extract(prefixo, produtos, scorer=fuzz.token_set_ratio, limit=10)
            candidatos_set.update(r[0] for r in top_todos if r[1] >= 70)

    # Aliases diretos têm sempre prioridade — garante que entram no catálogo
    if aliases:
        texto_lower = texto.lower()
        for alias, produto in aliases.items():
            if alias.lower() in texto_lower and produto in produtos:
                candidatos_set.add(produto)

    candidatos = list(candidatos_set)[:60]
    catalogo    = _build_catalogo(candidatos, sku_map)
    base_rag    = _construir_base_rag(exemplos or [], aliases)
    exemplos_txt = _recuperar_exemplos(texto, base_rag)

    prompt = (
        f"Catálogo de produtos de vernizes/unhas:\n{catalogo}\n"
        f"{exemplos_txt}\n"
        f"Mensagem de encomenda:\n{texto}\n\n"
        "Identifica TODOS os produtos mencionados e as suas quantidades.\n"
        "Usa o nome EXATO do catálogo acima — não inventes nomes.\n"
        "Se um produto não estiver no catálogo, ignora-o.\n"
        "IMPORTANTE: Se a mensagem disser 'X de cada' (ex: 'limas retas 1 de cada'), "
        "lista TODOS os produtos do catálogo que correspondam a X, cada um com a quantidade indicada.\n"
        "IMPORTANTE: 'direita'/'diretas' = 'reta'/'retas' no contexto de limas.\n"
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
