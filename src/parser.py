import re

# Padrões de quantidade em português
_QTDE_COM_UNIDADE = re.compile(r"\b(\d+)\s*(?:unidades?|und\.?|un\.?|pcs?)\b", re.IGNORECASE)
_QTDE_X           = re.compile(r"\bx\s*(\d+)\b|\b(\d+)\s*x\b", re.IGNORECASE)
_QTDE_CADA        = re.compile(r"\b(\d+)\s*(?:unidades?|und\.?|un\.?|pcs?)?\s*cada\b", re.IGNORECASE)
_QTDE_ISO         = re.compile(r"\b(\d+)\b")


def extrair_quantidade(texto: str) -> int | None:
    """
    Extrai a primeira quantidade mencionada num texto.
    Prioridade: "N unidades" > "Nx" / "xN" > número isolado
    Retorna None se nenhum número encontrado.
    """
    m = _QTDE_COM_UNIDADE.search(texto)
    if m:
        return int(m.group(1))
    m = _QTDE_X.search(texto)
    if m:
        return int(m.group(1) or m.group(2))
    m = _QTDE_ISO.search(texto)
    if m:
        return int(m.group(1))
    return None


def quantidade_para_produto(melhor_trecho: str, linhas: list[str], produto_nome: str = "") -> int:
    """
    Determina a quantidade pedida para um produto.

    Estratégia (por ordem de prioridade):
    1. "N unidades cada" / "N cada" em qualquer linha  → aplica a todos
    2. Quantidade na linha que contém as palavras do trecho
    3. Default: 1

    Os números que fazem parte do nome do produto (ex: "50" em "tips 50 unidades",
    "30" em "oleo 30ml", "100" e "180" em "lima 100 180") são excluídos —
    são atributos do produto, não quantidades encomendadas.
    """
    # Números que aparecem no nome do produto → atributos, não quantidades
    nums_produto = set(re.findall(r"\b\d+\b", produto_nome.lower()))

    def _qty_excluindo_produto(texto: str) -> int | None:
        """Extrai quantidade ignorando números que fazem parte do produto."""
        for m in re.finditer(r"\b(\d+)\b", texto):
            n = m.group(1)
            if n not in nums_produto:
                # Preferir se vier acompanhado de unidade ("N unidades")
                vizinho = texto[m.end():m.end() + 15].strip()
                if re.match(r"(?:unidades?|und\.?|un\.?|pcs?|cada)", vizinho, re.IGNORECASE):
                    return int(n)
                # Ou padrão xN / Nx
                contexto = texto[max(0, m.start()-2):m.end()+2]
                if re.search(r"\bx\s*\d|\d\s*x\b", contexto, re.IGNORECASE):
                    return int(n)
                # Número isolado (sem ser atributo do produto)
                return int(n)
        return None

    # Quantidade na linha onde o produto foi detectado
    # ("N cada" também é captado aqui pelo _qty_excluindo_produto via vizinho "cada")
    palavras = set(melhor_trecho.lower().split())
    for linha in linhas:
        if palavras.issubset(set(linha.lower().split())):
            qty = _qty_excluindo_produto(linha)
            if qty is not None:
                return qty

    return 1


def guardar_exemplo_treino(linhas_ocr: list[str], resultado: list[dict], caminho: str) -> None:
    """
    Guarda um par (input, output) para futuramente treinar um modelo seq2seq.

    Formato do ficheiro JSONL:
      {"input": "linha1 | linha2 | ...", "output": "produto1:qty1 | produto2:qty2"}

    Usar quando: resultado for validado pelo utilizador como correcto.
    """
    import json
    from pathlib import Path

    entrada = " | ".join(linhas_ocr)
    saida   = " | ".join(f"{r['produto']}:{r['quantidade']}" for r in resultado)

    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps({"input": entrada, "output": saida}, ensure_ascii=False) + "\n")
