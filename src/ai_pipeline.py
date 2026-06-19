import base64
import json
import pickle
import re
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz, process

# Unidade que segue a quantidade ("12 unidades", "3 un"). 'unid\w*' tolera os
# typos do WhatsApp ('unidases', 'unidaees'…). Mantido em sincronia com o mesmo
# padrão em server.py (_UNIDADES_RE).
_UNIDADES_RE = r"(?:unid\w*|un|pcs?|cada)"


def _chave_nome(nome: str) -> str:
    """
    Chave canónica para comparar nomes de produto entre fontes (aliases escritos
    à mão vs. catálogo da Shopkit): minúsculas, sem acentos, pontuação tornada
    espaço e espaços colapsados. Faz 'desidrat - desidratante' == 'desidrat
    desidratante' e 'esfoliante | grão grosso' == 'esfoliante grao grosso'.
    """
    sem_acentos = "".join(
        c for c in unicodedata.normalize("NFD", (nome or "").lower())
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", sem_acentos)).strip()


def _alias_no_texto(alias: str, texto_chave: str) -> bool:
    r"""
    True se o ``alias`` aparece em ``texto_chave`` (ambos já passados por
    _chave_nome), tolerante a espaços/pontuação dentro do termo: '3d' apanha
    '3 d' e vice-versa, mantendo fronteiras de palavra (não dispara em
    'abc3def'). Constrói-se um padrão a partir dos caracteres do alias (sem
    espaços), com \s* opcional entre eles, ancorado em fronteiras de palavra.
    """
    alias = alias.strip()
    if not alias:
        return False
    # \s* opcional só onde faz sentido: onde o alias já tinha espaço (entre
    # tokens) e nas transições dígito↔letra ('3d'↔'3 d'). NÃO entre dígitos
    # seguidos, para '104' não disparar em '1 0 4'.
    partes: list[str] = []
    anterior = ""
    for c in alias:
        if c == " ":
            anterior = " "
            continue
        if anterior and anterior != " " and (anterior.isdigit() == c.isdigit()):
            sep = ""              # mesmo tipo, sem espaço original → colado
        elif anterior:
            sep = r"\s*"          # fronteira de token ou dígito↔letra
        else:
            sep = ""              # primeiro caráter
        partes.append(sep + re.escape(c))
        anterior = c
    padrao = r"\b" + "".join(partes) + r"\b"
    return re.search(padrao, texto_chave) is not None


def load_produtos(data_dir: Path):
    # prod.pkl é o catálogo local (fallback). Pode não existir no modo só-API,
    # em que a Shopkit é a única fonte: nesse caso o catálogo local fica vazio.
    try:
        with open(data_dir / "prod.pkl", "rb") as f:
            produtos = pickle.load(f)
    except FileNotFoundError:
        produtos = []
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
    try:
        with open(data_dir / "context.json") as f:
            context = json.load(f)
    except FileNotFoundError:
        context = {}
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


def _linha_sem_quantidade(linha: str) -> str:
    """
    Linha sem a quantidade da encomenda, para pontuar candidatos no fuzzy.
    'super mãe 12 unidades' dilui o token_set_ratio do produto certo (cai de
    100 para ~60); tirar '12 unidades' devolve a linha à forma que o fuzzy
    apanha bem ('super mãe' → 'verniz gel super mãe maria' a 100).

    Só se remove 'número + unidade' (12 unidades, 3 un, 6 pcs) e palavras de
    unidade soltas. Números SEM unidade são preservados, porque podem ser
    distintivos do nome ('like gel 104', 'broca 13', 'lâmpada 90').
    """
    s = re.sub(rf"\b\d+\s*{_UNIDADES_RE}\b", " ", linha, flags=re.IGNORECASE)
    s = re.sub(rf"\b{_UNIDADES_RE}\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s or linha  # nunca devolver vazio (linhas que são só quantidade)


def _linhas_com_indices(texto: str) -> list[tuple[int, str]]:
    """
    Devolve (numero_original, texto_da_linha).

    O servidor pode enviar batches já numerados com o índice global ("23: texto").
    Preservar esse número evita que filtros de contexto, como "Cores novas",
    façam a resposta da AI ficar desalinhada com a tabela final.
    """
    linhas = []
    for fallback_idx, raw in enumerate(texto.splitlines(), 1):
        linha = raw.strip()
        if not linha:
            continue
        m = re.match(r"^(\d+)\s*[:.)-]\s+(.+)$", linha)
        if m:
            linhas.append((int(m.group(1)), m.group(2).strip()))
        else:
            linhas.append((fallback_idx, linha))
    return linhas


def _linha_so_contexto(linha: str) -> bool:
    lower = linha.lower().strip()
    lower_sem_qtd = re.sub(r"\b\d+\b", "", lower)
    lower_sem_qtd = re.sub(rf"\b{_UNIDADES_RE}\b", "", lower_sem_qtd).strip()
    return (
        lower in {"de cada", "verniz normal", "verniz gel"}
        or lower.startswith(("bom dia", "boa tarde", "boa noite"))
        or lower.startswith(("encomenda ", "pedido "))
        or "cores novas" in lower
        or lower_sem_qtd in {"cores novas", "verniz normal", "verniz gel"}
    )


def _candidatos_por_linha(linha: str, produtos: list, limit: int = 20) -> list:
    """Seleciona os melhores candidatos para uma linha de texto."""
    linha_norm = _normalizar_linha(linha)
    top = process.extract(linha_norm, produtos, scorer=fuzz.token_set_ratio, limit=limit)
    return [r[0] for r in top] if top else []


def _extrair_texto_imagem(image_path: Path, client) -> str:
    """ Claude transcreve o texto visível na imagem."""
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


def processar_imagem(image_path: Path, produtos: list, sku_map: dict, aliases: dict, client, exemplos: list = None, buscar_candidatos=None) -> tuple[list, str]:
    texto = _extrair_texto_imagem(image_path, client)
    if not texto:
        return [], ""
    print(f"[AI OCR] texto extraído: {texto[:200]}")
    return processar_texto(texto, produtos, sku_map, aliases, client, exemplos or [], buscar_candidatos), texto


def processar_texto(
    texto: str,
    produtos: list,
    sku_map: dict,
    aliases: dict,
    client,
    exemplos: list = None,
    buscar_candidatos=None,
) -> list:
    """
    Para cada linha, seleciona candidatos e envia texto + candidatos ao Claude
    para matching final.

    Candidatos: por defeito vêm de rapidfuzz sobre ``produtos`` (catálogo local).
    Se ``buscar_candidatos`` for dado, é uma função ``linha -> list[{"produto","ref"}]``
    (pesquisa Shopkit ao vivo): nesse caso a loja passa a ser a fonte dos candidatos
    e do sku_map, e o catálogo local só é usado como fallback por linha (quando a
    pesquisa devolve vazio — p. ex. falha de rede).
    """
    linhas_indexadas = _linhas_com_indices(texto)
    if not linhas_indexadas:
        return []

    # Filtrar linhas com termos ignorados (alias com valor "")
    _ignorar = {k for k, v in (aliases or {}).items() if v == ""}
    linhas_indexadas = [
        (idx, linha)
        for idx, linha in linhas_indexadas
        if not _linha_so_contexto(linha) and not any(t in linha.lower() for t in _ignorar)
    ]
    if not linhas_indexadas:
        return []
    linhas = [linha for _idx, linha in linhas_indexadas]

    # sku_map efetivo: começa pelo local e é enriquecido com as refs que a loja
    # devolver (a fonte principal passa a ser a API quando buscar_candidatos existe).
    sku_map = dict(sku_map or {})

    def _candidatos_rapidfuzz(linha: str) -> None:
        """Fallback local: pontua candidatos do catálogo .pkl para uma linha."""
        linha_norm = _normalizar_linha(_linha_sem_quantidade(linha))
        for produto, score, *_ in process.extract(linha_norm, produtos, scorer=fuzz.token_set_ratio, limit=20):
            candidatos_score[produto] = max(candidatos_score.get(produto, 0), score)
        linha_low = _normalizar_linha(linha.lower())
        if "de cada" in linha_low:
            prefixo = re.sub(r'\b\d+\b', '', linha_low.split("de cada")[0]).strip()
            for produto, score, *_ in process.extract(prefixo, produtos, scorer=fuzz.token_set_ratio, limit=10):
                if score >= 70:
                    candidatos_score[produto] = max(candidatos_score.get(produto, 0), score)

    # Candidatos por linha (linha a linha evita diluição de score).
    # A Shopkit é a fonte principal; o catálogo local (.pkl) é a rede de segurança.
    # O fallback dispara sempre que a loja NÃO dá candidatos para a linha:
    #   None  -> sem API key, ou erro/timeout real
    #   []    -> a API respondeu mas não encontrou nada (ex: 'clear' vs 'transparente',
    #            termos abreviados que o filtro AND da loja não apanha)
    # Em ambos os casos o rapidfuzz sobre o .pkl recupera a linha.
    candidatos_score: dict[str, float] = {}
    for linha in linhas:
        hits = buscar_candidatos(linha) if buscar_candidatos else None
        if not hits:
            _candidatos_rapidfuzz(linha)
            continue
        # Shopkit encontrou: o nome já vem normalizado como o catálogo.
        for h in hits:
            nome = h.get("produto", "")
            if not nome:
                continue
            candidatos_score[nome] = max(candidatos_score.get(nome, 0), 100)
            ref = (h.get("ref") or "").strip()
            if ref:
                sku_map[nome] = ref

    # Aliases diretos têm sempre prioridade — garante que entram no catálogo.
    # O alvo do alias é escrito à mão e pode diferir do nome da loja em pontuação,
    # acentos ou espaços (ex.: 'desidrat desidratante' vs 'desidrat - desidratante').
    # Resolvemo-lo contra o nome REAL do catálogo por chave canónica, em vez de exigir
    # igualdade exata — senão o alias é silenciosamente ignorado.
    # Produtos cujo nome não partilha palavras com o termo escrito pelo cliente
    # (ex.: alias 'top coat 3d' → 'top coat tempered'). A prova de que não é
    # alucinação é o alias ter disparado, não a palavra do produto na linha —
    # por isso isentamo-los do filtro anti-alucinação mais abaixo.
    produtos_via_alias: set[str] = set()
    if aliases:
        texto_chave = _chave_nome(texto)
        por_chave = {_chave_nome(p): p for p in candidatos_score}
        for p in produtos:
            por_chave.setdefault(_chave_nome(p), p)
        for alias, produto in aliases.items():
            if not _alias_no_texto(_chave_nome(alias), texto_chave):
                continue
            real = por_chave.get(_chave_nome(produto))
            if real:
                candidatos_score[real] = max(candidatos_score.get(real, 0), 100)
                produtos_via_alias.add(real)

    cap = max(60, min(len(linhas) * 8, 180))
    candidatos = [
        produto for produto, _score in sorted(candidatos_score.items(), key=lambda item: item[1], reverse=True)
    ][:cap]
    # Conjunto efetivo contra o qual a resposta do Claude é validada: os candidatos
    # enviados no prompt (Shopkit ao vivo) ou, no modo local, o catálogo .pkl.
    produtos_efetivos = candidatos or list(produtos)
    catalogo    = _build_catalogo(candidatos, sku_map)
    base_rag    = _construir_base_rag(exemplos or [], aliases)
    exemplos_txt = _recuperar_exemplos(texto, base_rag)

    linhas_numeradas = "\n".join(f"{idx}: {linha}" for idx, linha in linhas_indexadas)
    prompt = (
        f"Catálogo de produtos de vernizes/unhas:\n{catalogo}\n"
        f"{exemplos_txt}\n"
        f"Mensagem de encomenda (com número de linha):\n{linhas_numeradas}\n\n"
        "Identifica os produtos mencionados e as suas quantidades.\n"
        "\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1. Usa o nome EXATO do catálogo acima — não inventes nomes.\n"
        "2. Cada palavra distintiva do produto escolhido (nome próprio, cor específica, "
        "número de variante) TEM de aparecer na linha de onde dizes que vem — exata OU "
        "como typo razoável OU como abreviação. Categorias genéricas como 'verniz gel', "
        "'top coat', 'builder' sozinhas NÃO chegam para escolher uma variante.\n"
        "3. Uma linha normalmente corresponde a UM produto. NUNCA atribuas vários produtos "
        "à mesma linha, exceto se essa linha tiver o padrão 'X de cada' OU listar "
        "explicitamente várias cores/variantes separadas por vírgula.\n"
        "4. Se uma linha menciona só uma categoria genérica (ex: 'verniz gel') sem cor/nome "
        "específico, ignora-a.\n"
        "5. Se um produto não estiver no catálogo, ignora-o.\n"
        "\n"
        "TOLERÂNCIA A IMPERFEIÇÕES — quando o nome escrito não bate exato com o catálogo:\n"
        "• Typos de uma ou duas letras: aceita se só houver UM candidato no catálogo "
        "razoavelmente próximo. Ex: 'tammy natura' → 'tawny natura' (única opção parecida); "
        "'princepezinho' → 'principezinho'; 'matte' → 'matte finish'. "
        "Se houver dois candidatos próximos (ex: 'rosa' pode ser 'rosa pop' ou 'rosa sakura'), "
        "NÃO escolhas — ignora.\n"
        "• Ordem de palavras trocada: 'top coat like gel' = 'like gel top coat'. Aceita.\n"
        "• Ordem trocada E typo ao mesmo tempo contam como UMA só imperfeição combinada — "
        "aceita à mesma se sobrar UM candidato claro. Ex: 'base gummy tranparente' = "
        "'gummy base transparente' (palavras trocadas + falta o 's'); a cor 'transparente' "
        "está escrita, logo distingue de 'gummy base nude'/'gummy base rosa claro'.\n"
        "• Números aproximados (potência/volume): se o cliente pede um número que não existe "
        "no catálogo MAS existe UM produto da mesma categoria com número próximo, devolve "
        "esse. Ex: 'lâmpada LED 99W' → 'lâmpada 90W' (única que existe). "
        "Se há vários números possíveis (30ml, 50ml, ambos existem), NÃO troques — devolve "
        "o que o cliente pediu se existir, senão ignora.\n"
        "• Palavras a mais/a menos: 'brigadeiro' = 'verniz gel brigadeiro natura' (abreviação).\n"
        "\n"
        "Linhas-contexto a ignorar: 'Verniz normal', 'Cores novas ...', 'De cada' isolado, "
        "saudações ('bom dia'…), 'encomenda', 'pedido'.\n"
        "Padrão 'X de cada' (ex: 'limas retas 1 de cada'): aí sim, lista TODOS os produtos "
        "do catálogo que correspondam a X, cada um com a quantidade indicada.\n"
        "Sinónimos: 'direita'/'diretas' = 'reta'/'retas' (limas).\n"
        "\n"
        'Responde APENAS com JSON válido (sem texto antes ou depois):\n'
        '[{"produto": "nome exato do catálogo", "quantidade": número, "linha": número_da_linha_de_onde_veio}, ...]\n'
        'O campo "linha" é OBRIGATÓRIO e tem de corresponder à linha real onde o nome '
        "específico do produto aparece. Se não consegues apontar uma linha concreta com "
        "a palavra distintiva, não incluas o produto.\n"
        "Se não houver produtos, responde []."
    )

    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max(1024, len(linhas) * 80),
            messages=[{"role": "user", "content": prompt}]
        )
        raw = r.content[0].text
        print(f"[AI texto] resposta: {raw[:300]}")
        items = _parse_json(raw)
    except Exception as e:
        print(f"[AI texto] erro: {e}")
        raise RuntimeError(f"Erro ao chamar Claude: {e}") from e

    # Mapa idx_original → texto da linha, para validar atribuições
    linha_por_idx = {idx: linha for idx, linha in linhas_indexadas}
    genericas = _palavras_genericas(produtos_efetivos)

    resultados = []
    for item in items:
        match = _match_produto(item.get("produto", ""), produtos_efetivos)
        if not match:
            continue
        produto_real, score = match
        raw_idx = item.get("linha")
        linha_idx = int(raw_idx) if raw_idx is not None else None

        # Validação anti-alucinação: o nome distintivo tem de aparecer na linha apontada,
        # excepto se a linha for um padrão "de cada" (que expande várias variantes) ou
        # se o produto veio de um alias acionado (o termo do cliente justifica-o, mesmo
        # que não partilhe palavras com o nome do catálogo — ex.: '3 d' → 'tempered').
        if linha_idx is not None and linha_idx in linha_por_idx and produto_real not in produtos_via_alias:
            linha_texto = linha_por_idx[linha_idx]
            if "de cada" not in linha_texto.lower():
                if not _produto_referenciado_na_linha(produto_real, linha_texto, genericas):
                    print(f"[AI] descartado (alucinação): {produto_real!r} → linha {linha_idx} {linha_texto!r}")
                    continue

        # 5º elemento: ref resolvida (Shopkit ou .pkl). Permite ao servidor preencher
        # o SKU mesmo para produtos que só existem na loja, não no catálogo local.
        ref = sku_map.get(produto_real, "")
        resultados.append((produto_real, score, int(item.get("quantidade", 1)), linha_idx, ref))
    return resultados


# Palavras estruturais: tipo de produto, unidades e ligações. São sempre ruído
# (o cliente quase nunca as escreve), independentemente da frequência no catálogo.
_PALAVRAS_ESTRUTURAIS = {
    "verniz", "gel", "normal", "base", "top", "coat", "ml", "g", "gr",
    "unidade", "unidades", "un", "pcs", "cada", "de", "da", "do", "com",
    "para", "e", "natura", "pop", "kit",
}


def _palavras_genericas(produtos: list[str]) -> set[str]:
    """Tokens que aparecem em muitos produtos — não identificam variantes específicas."""
    base = set(_PALAVRAS_ESTRUTURAIS)
    contagem: dict[str, int] = {}
    for p in produtos:
        for t in re.findall(r"\w+", p.lower()):
            if len(t) >= 3 and not t.isdigit():
                contagem[t] = contagem.get(t, 0) + 1
    # Token genérico = aparece em >5% do catálogo
    limite = max(3, int(len(produtos) * 0.05))
    return base | {t for t, c in contagem.items() if c >= limite}


def _produto_referenciado_na_linha(produto: str, linha: str, genericas: set[str]) -> bool:
    """
    True se pelo menos uma palavra distintiva do produto aparece na linha.
    Tolerante a typos curtos (ex: tammy↔tawny, 99w↔90w) — confiando que a AI
    aplicou as regras do prompt para escolher candidato único.
    """
    linha_norm = _normalizar_linha(linha).lower()
    linha_tokens = set(re.findall(r"\w+", linha_norm))
    if not linha_tokens:
        return False

    palavras_produto = re.findall(r"\w+", produto.lower())
    distintivas_alfa = [t for t in palavras_produto if t.isalpha() and len(t) >= 3 and t not in genericas]
    distintivas_num  = [t for t in palavras_produto if not t.isalpha() and any(c.isdigit() for c in t)]

    if not distintivas_alfa and not distintivas_num:
        # Todas as palavras do produto foram marcadas genéricas (acontece quando
        # os candidatos de uma linha são poucos e parecidos — 'maria ...' ou
        # 'gummy base ...' — inflando o que conta como 'genérico'). Caímos para
        # as palavras essenciais (sem estruturais) e exigimos que TODAS apareçam
        # na linha, com tolerância a typo por palavra (não token_set_ratio sobre
        # a frase, que colapsa com um único typo numa palavra longa, p.ex.
        # 'transparente' vs 'tranparente').
        essenciais = [
            t for t in re.findall(r"\w+", produto.lower())
            if t not in _PALAVRAS_ESTRUTURAIS and len(t) >= 3
        ]
        if not essenciais:
            return fuzz.token_set_ratio(produto.lower(), linha_norm) >= 80
        for e in essenciais:
            if e in linha_tokens:
                continue
            if any(len(t) >= 3 and fuzz.ratio(e, t) >= 80 for t in linha_tokens):
                continue
            return False  # falta uma palavra essencial → não está nesta linha
        return True

    # Palavras alfabéticas: match exato OU typo (≥65) OU substring (abreviação)
    for d in distintivas_alfa:
        if d in linha_tokens:
            return True
        for t in linha_tokens:
            if not t.isalpha() or len(t) < 3:
                continue
            if fuzz.ratio(d, t) >= 65:
                return True
            # Abreviação curta (princip→principezinho): só se a token da linha
            # for prefixo distintivo da palavra do produto, com ≥4 chars
            if len(t) >= 4 and d.startswith(t):
                return True

    # Números: match exato OU número próximo na linha (potência/volume similar)
    def _so_digitos(tok: str) -> int | None:
        m = re.search(r"\d+", tok)
        return int(m.group()) if m else None

    for d in distintivas_num:
        if d in linha_tokens:
            return True
        d_num = _so_digitos(d)
        if d_num is None:
            continue
        for t in linha_tokens:
            t_num = _so_digitos(t)
            if t_num is not None and abs(d_num - t_num) <= max(10, d_num // 10):
                return True

    return False
