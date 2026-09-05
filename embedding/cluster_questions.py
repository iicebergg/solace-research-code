"""
Cosine distance: scikit-learn's KMeans is Euclidean-only, so cosine is
implemented the standard way, as spherical k-means. Vectors are re-normalized
to unit length after PCA; on the unit sphere, squared Euclidean distance
equals 2*(1 - cosine similarity), so Euclidean KMeans on normalized vectors
clusters by cosine proximity. Silhouette scoring uses metric="cosine" so
model selection matches the clustering geometry.

Choosing k: every run sweeps K_RANGE and records inertia (elbow) and cosine
silhouette per k, printing the table and saving <subject>_qwen_elbow_plot.png.
N_CLUSTERS is used for the final clustering.

Model: Qwen/Qwen3-Embedding-8B is the most accurate generally available Qwen
embedding model.

Outputs:
    <subject>_qwen_elbow_plot.png   (inertia + cosine silhouette vs k)
    <subject>_qwen_clusters_plot.png
    <subject>_qwen_questions_with_clusters.csv
        columns: source_file, test_id, qid, type, cluster, tsne_x, tsne_y, clean_text
        (any of those identifying columns present in the input are carried through;
         clean_text, cluster, tsne_x, tsne_y are always written)
    <subject>_qwen_cluster_keywords.csv
        columns: cluster, size, top_keywords, category_name  (category_name left blank)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SUBJECT = "math"  # set to either "math", "reading", or "science"

# The elbow/silhouette sweep runs for diagnostics.
N_CLUSTERS = 6  # set to any integer value from 2 to 21, inclusive
K_RANGE = range(2, 21)  # candidate k values swept for the elbow and silhouette

# Alternatives with the same API: "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B"
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Matryoshka (MRL) output dim. "None" keeps the full 4096-d vector (highest accuracy).
TRUNCATE_DIM = 1024

INSTRUCTION = "Identify the topic and skill that the test question assesses."

BATCH_SIZE = 16
PCA_COMPONENTS = 50
TSNE_PERPLEXITY = 30
RANDOM_STATE = 30

TEXT_COLUMN = "clean_text"
PASSTHROUGH = ["source_file", "test_id", "qid", "type"]

OUTPUT_TAG = "qwen"

INPUT_CSV = f"{SUBJECT}_questions.csv"


def _tag(stem: str, ext: str) -> str:
    return f"{SUBJECT}_{OUTPUT_TAG}_{stem}.{ext}" if OUTPUT_TAG else f"{SUBJECT}_{stem}.{ext}"


PLOT_PNG = _tag("clusters_plot", "png")
ELBOW_PNG = _tag("elbow_plot", "png")
CLUSTERS_CSV = _tag("questions_with_clusters", "csv")
KEYWORDS_CSV = _tag("cluster_keywords", "csv")


# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    print(f"Loading {MODEL_NAME} on {device} ...")

    st_kwargs: dict = {"device": device}
    if TRUNCATE_DIM is not None:
        st_kwargs["truncate_dim"] = TRUNCATE_DIM  # MRL: shorten output vectors

    model = SentenceTransformer(MODEL_NAME, **st_kwargs)

    encode_kwargs: dict = dict(
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # cosine for KMeans
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    if INSTRUCTION:
        # Qwen instruction format applied uniformly across the symmetric set.
        encode_kwargs["prompt"] = f"Instruct: {INSTRUCTION}\nQuery:"

    emb = model.encode(texts, **encode_kwargs)
    print(f"Embedded {emb.shape[0]} questions to {emb.shape[1]}-d vectors.")
    return emb.astype(np.float32)


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
def sweep_k(X: np.ndarray) -> tuple[dict[int, np.ndarray], list[int], list[float], list[float]]:
    """Fit KMeans at every k in K_RANGE on the (unit-normalized) vectors.
    Returns fitted labels per k, plus the inertia and cosine-silhouette curves."""
    labels_by_k: dict[int, np.ndarray] = {}
    ks: list[int] = []
    inertias: list[float] = []
    silhouettes: list[float] = []

    print(f"{'k':>4}  {'inertia':>12}  {'cosine silhouette':>18}")
    for k in K_RANGE:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, metric="cosine")
        labels_by_k[k] = labels
        ks.append(k)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(sil))
        print(f"{k:>4}  {km.inertia_:>12.2f}  {sil:>18.4f}")
    return labels_by_k, ks, inertias, silhouettes


def plot_elbow(ks: list[int], inertias: list[float], silhouettes: list[float],
               chosen_k: int, path: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ks, inertias, "o-", color="#1f77b4", label="inertia (elbow)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("inertia (within-cluster sum of squares)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(ks)

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, "s--", color="#d62728", label="cosine silhouette")
    ax2.set_ylabel("cosine silhouette", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.axvline(chosen_k, color="#9467bd", linestyle="-.",
                label=f"configured k={chosen_k}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(f"{SUBJECT} — k diagnostics (spherical KMeans on Qwen embeddings)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


def choose_k_and_cluster(X: np.ndarray) -> tuple[np.ndarray, int]:
    """Sweep K_RANGE for diagnostics, then cluster using N_CLUSTERS."""
    labels_by_k, ks, inertias, silhouettes = sweep_k(X)
    sil_k = ks[int(np.argmax(silhouettes))]
    chosen_k = N_CLUSTERS
    print(f"Best cosine silhouette (diagnostic only): k={sil_k}")
    print(f"Using configured k={chosen_k}.")

    plot_elbow(ks, inertias, silhouettes, chosen_k, ELBOW_PNG)

    if chosen_k in labels_by_k:
        return labels_by_k[chosen_k], chosen_k
    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    return km.fit_predict(X), chosen_k


# ----------------------------------------------------------------------------
# Keywords per cluster (c-TF-IDF; Qwen vectors are not word-based)
# ----------------------------------------------------------------------------
def cluster_keywords(df: pd.DataFrame, k: int, top_n: int = 12) -> pd.DataFrame:
    docs, ids = [], []
    for c in range(k):
        joined = " ".join(df.loc[df["cluster"] == c, TEXT_COLUMN].astype(str))
        docs.append(joined)
        ids.append(c)

    vec = TfidfVectorizer(
        ngram_range=(1, 1),
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        stop_words="english",
    )
    tfidf = vec.fit_transform(docs)
    terms = np.array(vec.get_feature_names_out())

    rows = []
    for i, c in enumerate(ids):
        row = tfidf[i].toarray().ravel()
        top_idx = row.argsort()[::-1][:top_n]
        kw = ", ".join(terms[j] for j in top_idx if row[j] > 0)
        rows.append(
            {
                "cluster": c,
                "size": int((df["cluster"] == c).sum()),
                "top_keywords": kw,
                "category_name": "",  # fill in by hand
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
PALETTE_20 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def plot_tsne(df: pd.DataFrame, k: int, path: str) -> None:
    cmap = ListedColormap(PALETTE_20[:k])
    bounds = np.arange(-0.5, k + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(11, 9))
    sc = ax.scatter(
        df["tsne_x"], df["tsne_y"], c=df["cluster"],
        cmap=cmap, norm=norm, s=14, alpha=0.8, linewidths=0,
    )
    ax.set_title(f"{SUBJECT} — Qwen ({MODEL_NAME.split('/')[-1]}) KMeans k={k}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    cbar = fig.colorbar(sc, ax=ax, ticks=range(k))
    cbar.set_label("cluster")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote {path}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    in_path = Path(INPUT_CSV)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path} (run step 1 first, or check SUBJECT).")

    df = pd.read_csv(in_path)

    text_col = TEXT_COLUMN if TEXT_COLUMN in df.columns else "text"
    if text_col not in df.columns:
        sys.exit(f"No '{TEXT_COLUMN}' or 'text' column in {in_path}.")
    globals()["TEXT_COLUMN"] = text_col  # keyword step reads this

    df = df[df[text_col].astype(str).str.strip().ne("")].reset_index(drop=True)
    texts = df[text_col].astype(str).tolist()
    print(f"Loaded {len(texts)} questions from {in_path}.")

    emb = embed_texts(texts)

    n_comp = min(PCA_COMPONENTS, emb.shape[1], max(2, emb.shape[0] - 1))
    Xr = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(emb)
    print(f"PCA to {n_comp} components.")

    # Cosine (spherical) k-means: PCA breaks the unit norms the encoder set,
    # so re-normalize. On unit vectors, Euclidean distance ranks pairs exactly
    # as cosine similarity does, making KMeans below a cosine clustering.
    Xr = normalize(Xr)
    print("Re-normalized to unit length (cosine / spherical k-means).")

    labels, k = choose_k_and_cluster(Xr)
    df["cluster"] = labels

    perp = float(min(TSNE_PERPLEXITY, max(5, (len(df) - 1) // 3)))
    coords = TSNE(
        n_components=2, perplexity=perp, init="pca",
        random_state=RANDOM_STATE, learning_rate="auto",
    ).fit_transform(Xr)
    df["tsne_x"], df["tsne_y"] = coords[:, 0], coords[:, 1]

    # Assemble output columns: passthrough identifiers, then cluster + coords + text.
    keep = [c for c in PASSTHROUGH if c in df.columns]
    out_cols = keep + ["cluster", "tsne_x", "tsne_y", text_col]
    df[out_cols].to_csv(CLUSTERS_CSV, index=False)
    print(f"Wrote {CLUSTERS_CSV}")

    cluster_keywords(df, k).to_csv(KEYWORDS_CSV, index=False)
    print(f"Wrote {KEYWORDS_CSV}")

    plot_tsne(df, k, PLOT_PNG)
    print("Done.")


if __name__ == "__main__":
    main()