import re
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


def normalizar_unidades(texto):
    """Remove sufixos de unidades após números: '30ml' → '30', '30m1' → '30', '50g' → '50'."""
    tokens = texto.split()
    return " ".join(
        re.sub(r"^(\d+).*", r"\1", t) if re.match(r"^\d+[a-zA-Z]", t) else t
        for t in tokens
    )