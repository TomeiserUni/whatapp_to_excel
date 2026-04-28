import re
import unicodedata
import numpy as np
import pickle


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def remover_acentos(texto: str) -> str:
    """'espírito' → 'espirito', 'última' → 'ultima'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )


def normalizar_unidades(texto):
    """Remove sufixos de unidades após números: '30ml' → '30', '500grs' → '500'."""
    tokens = texto.split()
    result = []
    for t in tokens:
        # Corrigir confusão OCR: sequência inicial de dígitos misturada com 'O'/'o'
        # ex: '5OO' → '500', '5OOgrs' → '500grs', '5O0grs' → '500grs'
        if re.match(r"^\d", t):
            t = re.sub(r"^([\dOo]+)", lambda m: m.group(1).upper().replace("O", "0"), t)
        if re.match(r"^\d+[a-zA-Z]", t):
            result.append(re.sub(r"^(\d+).*", r"\1", t))
        else:
            result.append(t)
    return " ".join(result)