"""
Fine-tuning do SentenceTransformer com pares produto/alias.

Dois modos (seleccionado automaticamente):
  • NVIDIA KD  — se NVIDIA_API_KEY estiver em .env:
                  usa nvidia/nv-embed-v2 como professor; treina o aluno
                  local por distilação de conhecimento (pairwise cosine MSE).
  • Contrastivo — sem chave: MultipleNegativesRankingLoss normal.

Após treino re-gera emb_prod.npy com os novos embeddings.

Correr: python src/train_embeddings.py
"""
import json
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_pickle, remover_acentos

BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
MODEL_DIR     = DATA_DIR / "model_finetuned"
TEACHER_CACHE = DATA_DIR / "nvidia_teacher_embs.npz"   # cache para não re-chamar API

BATCH_SIZE    = 32
EPOCHS        = 5
LR            = 2e-5


# ──────────────────────────────────────────────
# Utilitários comuns
# ──────────────────────────────────────────────

def _normalizar(texto: str) -> str:
    return remover_acentos(texto.lower().strip())


def _gerar_pares(produtos: list, aliases: dict) -> list[tuple[str, str]]:
    """
    Gera pares (anchor, positive) para treino.
    Fontes: aliases · auto-pares · variantes parciais.
    """
    produtos_set = {p.lower() for p in produtos}
    pares = []

    for chave, alvo in aliases.items():
        chave_n, alvo_n = _normalizar(chave), alvo.lower()
        if alvo_n in produtos_set:
            pares.append((chave_n, alvo_n))
            pares.append((alvo_n,  chave_n))

    for p in produtos:
        pares.append((p.lower(), p.lower()))

    random.seed(42)
    for p in produtos:
        palavras = p.lower().split()
        n = len(palavras)
        if n < 3:
            continue
        pares.append((" ".join(palavras[:-1]), p.lower()))
        pares.append((" ".join(palavras[1:]),  p.lower()))
        if n >= 4:
            pares.append((" ".join(palavras[:-2]), p.lower()))
            pares.append((" ".join(random.sample(palavras, 2)), p.lower()))

    seen, unicos = set(), []
    for par in pares:
        if par not in seen:
            seen.add(par)
            unicos.append(par)
    return unicos


# ──────────────────────────────────────────────
# Modo NVIDIA — knowledge distillation
# ──────────────────────────────────────────────

def _nvidia_embed(texts: list[str], api_key: str, model_id: str) -> np.ndarray:
    """
    Chama a API NVIDIA e devolve embeddings normalizados.
    Processa em batches de 96 (limite da API).
    """
    from openai import OpenAI

    client  = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    result  = []
    BATCH   = 96

    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        resp  = client.embeddings.create(
            input=chunk,
            model=model_id,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"},
        )
        result.extend([r.embedding for r in resp.data])
        print(f"  NVIDIA API: {min(start + BATCH, len(texts))}/{len(texts)} textos", end="\r")

    print()
    arr = np.array(result, dtype=np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def _treinar_com_distilacao(
    model,
    textos_unicos: list[str],
    teacher_embs: np.ndarray,
):
    """
    Treina o modelo aluno para reproduzir as similaridades coseno do professor.
    Perda: MSE entre a matriz de similaridade do aluno e a do professor.
    Dimensão-agnóstica — funciona mesmo que professor e aluno tenham dims diferentes.
    """
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    device     = next(model.parameters()).device
    teacher_t  = torch.tensor(teacher_embs, dtype=torch.float32, device=device)

    optimizer  = AdamW(model.parameters(), lr=LR)
    n_steps    = (len(textos_unicos) // BATCH_SIZE + 1) * EPOCHS
    scheduler  = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(n_steps * 0.10)),
        num_training_steps=n_steps,
    )

    # Acesso directo ao transformer e tokenizer (API v5)
    transformer_module = model[0]
    tokenizer          = transformer_module.tokenizer
    auto_model         = transformer_module.auto_model.to(device)

    idx_list = list(range(len(textos_unicos)))
    model.train()

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(idx_list)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, len(idx_list), BATCH_SIZE):
            batch_idx = idx_list[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue

            batch_texts = [textos_unicos[i] for i in batch_idx]

            # Embeddings do aluno com gradientes (mean pooling + normalização)
            enc    = tokenizer(batch_texts, return_tensors="pt",
                               padding=True, truncation=True, max_length=128).to(device)
            out    = auto_model(**enc)
            tok_e  = out.last_hidden_state                          # [B, L, H]
            mask   = enc["attention_mask"].unsqueeze(-1).float()
            s_embs = (tok_e * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            s_norm = F.normalize(s_embs, p=2, dim=1)               # [B, 384]

            # Embeddings do professor (sem gradientes)
            t_norm   = teacher_t[batch_idx]                         # [B, D_teacher]

            # Matrizes de similaridade coseno
            s_sim    = s_norm @ s_norm.T                            # [B, B]
            t_sim    = t_norm @ t_norm.T                            # [B, B]

            loss = F.mse_loss(s_sim, t_sim)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg = epoch_loss / max(1, n_batches)
        print(f"  epoch {epoch}/{EPOCHS}  loss={avg:.4f}")

    model.eval()


# ──────────────────────────────────────────────
# Modo contrastivo (sem chave NVIDIA)
# ──────────────────────────────────────────────

def _treinar_contrastivo(model, pares: list[tuple[str, str]]):
    from sentence_transformers import InputExample
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from torch.utils.data import DataLoader

    exemplos = [InputExample(texts=[a, p]) for a, p in pares]
    loader   = DataLoader(exemplos, shuffle=True, batch_size=BATCH_SIZE)
    loss     = MultipleNegativesRankingLoss(model)
    n_steps  = len(loader) * EPOCHS
    warmup   = max(1, int(n_steps * 0.10))

    print(f"  {n_steps} steps  |  warmup {warmup}  |  lr {LR}")
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup,
        optimizer_params={"lr": LR},
        show_progress_bar=True,
    )


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def treinar():
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer

    load_dotenv(BASE_DIR / ".env")

    api_key  = os.environ.get("NVIDIA_API_KEY", "").strip()
    model_id = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v2").strip()
    modo_kd  = bool(api_key)

    # ── Dados ──────────────────────────────────────────────────────
    print("A carregar catálogo e aliases...")
    produtos = load_pickle(DATA_DIR / "prod.pkl")

    alias_path = DATA_DIR / "aliases.json"
    aliases    = {}
    if alias_path.exists():
        with open(alias_path, encoding="utf-8") as f:
            aliases = json.load(f)

    pares = _gerar_pares(produtos, aliases)
    print(f"  {len(pares):,} pares  |  {len(aliases)} aliases  |  {len(produtos):,} produtos")

    # ── Modelo aluno ───────────────────────────────────────────────
    base  = str(MODEL_DIR) if MODEL_DIR.exists() else "all-MiniLM-L6-v2"
    print(f"A carregar modelo aluno: {base}")
    model = SentenceTransformer(base)

    # ── Treino ─────────────────────────────────────────────────────
    if modo_kd:
        print(f"\nModo: NVIDIA Knowledge Distillation  ({model_id})")

        textos_unicos = sorted({t for par in pares for t in par})
        print(f"  {len(textos_unicos):,} textos únicos para o professor")

        # Cache: evita re-chamar a API se os textos não mudaram
        cache_textos = None
        if TEACHER_CACHE.exists():
            cache = np.load(TEACHER_CACHE, allow_pickle=True)
            cache_textos = list(cache["textos"])

        if cache_textos == textos_unicos:
            print("  A usar cache NVIDIA (nvidia_teacher_embs.npz)")
            teacher_embs = np.load(TEACHER_CACHE)["embs"]
        else:
            print(f"  A chamar NVIDIA API…")
            teacher_embs = _nvidia_embed(textos_unicos, api_key, model_id)
            np.savez(TEACHER_CACHE,
                     textos=np.array(textos_unicos, dtype=object),
                     embs=teacher_embs)
            print(f"  Cache guardada ({teacher_embs.shape})")

        print(f"\nA treinar por distilação ({EPOCHS} epochs, batch {BATCH_SIZE})...")
        _treinar_com_distilacao(model, textos_unicos, teacher_embs)

    else:
        print("\nModo: Contrastivo (sem chave NVIDIA)")
        print("  Dica: define NVIDIA_API_KEY em .env para activar knowledge distillation")
        _treinar_contrastivo(model, pares)

    # ── Guardar modelo ─────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))
    print(f"\nModelo guardado em {MODEL_DIR.relative_to(BASE_DIR)}")

    # ── Re-embeddar catálogo ────────────────────────────────────────
    print("A re-calcular emb_prod.npy com o modelo actualizado...")
    emb = model.encode(
        [p.lower() for p in produtos],
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    )
    np.save(DATA_DIR / "emb_prod.npy", emb)
    print(f"  emb_prod.npy actualizado ({len(produtos):,} × {emb.shape[1]} dims)")


if __name__ == "__main__":
    treinar()
1