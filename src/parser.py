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


def quantidade_para_produto(melhor_trecho: str, linhas: list[str]) -> int:
    """
    Determina a quantidade pedida para um produto.

    Estratégia (por ordem de prioridade):
    1. "N unidades cada" / "N cada" em qualquer linha  → aplica a todos
    2. Quantidade na linha que contém as palavras do trecho
    3. Default: 1

    NOTE: Para casos conversacionais complexos ("de X queria A e B, 10 cada")
    esta lógica cobre a maioria dos padrões estruturados.
    Para treinar um modelo seq2seq (mT5-small) no futuro, guardar os pares
    (linhas_ocr, resultado) em data/exemplos_parser.jsonl como dados de treino.
    """
    # 1. "N cada" — quantidade global para todos os produtos da mensagem
    for linha in linhas:
        m = _QTDE_CADA.search(linha)
        if m:
            return int(m.group(1))

    # 2. Quantidade na linha onde o produto foi detectado
    palavras = set(melhor_trecho.lower().split())
    for linha in linhas:
        if palavras.issubset(set(linha.lower().split())):
            qty = extrair_quantidade(linha)
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
