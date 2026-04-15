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