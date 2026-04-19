import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

OUT_DIR = "data/processed"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def make_rca_matrix(panel, year, countries, products):
    c_idx = {c: i for i, c in enumerate(countries)}
    p_idx = {p: i for i, p in enumerate(products)}
    mat = np.zeros((len(countries), len(products)), dtype=np.float32)
    sub = panel[panel["year"] == year][["country", "product", "rca_binary"]]

    for row in sub.itertuples(index=False):
        if row.country in c_idx and row.product in p_idx:
            mat[c_idx[row.country], p_idx[row.product]] = row.rca_binary

    return mat


def compute_density(rca_binary, phi):
    phi_sum = phi.sum(axis=1)
    phi_sum = np.where(phi_sum == 0, 1, phi_sum)
    numerator = rca_binary @ phi

    return numerator / phi_sum[None, :]


def evaluate_model(density, rca_t, rca_t5, k_vals=(10, 20, 50, 100)):
    candidates = (rca_t == 0)
    gained = ((rca_t == 0) & (rca_t5 == 1))
    scores = density[candidates]
    labels = gained[candidates]

    if labels.sum() == 0:
        return None

    results = {}
    results["auc"] = roc_auc_score(labels, scores)
    results["n_entries"] = int(labels.sum())
    results["n_candidates"] = int(len(labels))
    results["entry_rate"] = labels.mean()
    order = np.argsort(-scores)

    for k in k_vals:
        topk_labels = labels[order[:k]]
        results[f"p@{k}"] = topk_labels.mean()

    return results


def phi_from_embeddings(embeddings):
    E = normalize(embeddings, norm="l2")
    phi = E @ E.T
    np.fill_diagonal(phi, 0)
    phi = np.clip(phi, 0, None)

    return phi


def main():
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))
    countries = sorted(panel["country"].unique())
    products = np.load(os.path.join(OUT_DIR, "product_list.npy"), allow_pickle=True)
    products = products.tolist()
    years = sorted(panel["year"].unique())

    phi_hidalgo_df = pd.read_parquet(os.path.join(OUT_DIR, "proximity_hildago.parquet"))
    phi_hidalgo_full = phi_hidalgo_df.values
    all_products_full = sorted(panel["product"].unique())
    full_index = {p: i for i, p in enumerate(all_products_full)}
    idx = [full_index[p] for p in products if p in full_index]
    phi_hidalgo = phi_hidalgo_full[np.ix_(idx, idx)]

    models = {"Hidalgo": phi_hidalgo}
    for d in [32, 64, 128]:
        fpath = os.path.join(OUT_DIR, f"embeddings_mf_d{d}.npy")
        if os.path.exists(fpath):
            emb = np.load(fpath)
            models[f"MF-d{d}"] = phi_from_embeddings(emb)

    fpath_n2v = os.path.join(OUT_DIR, "embeddings_node2vec_d64.npy")
    if os.path.exists(fpath_n2v):
        emb_n2v = np.load(fpath_n2v)
        models["node2vec-d64"] = phi_from_embeddings(emb_n2v)

    print(f"\nModels loaded: {list(models.keys())}")

    models["Random"] = None

    eval_windows = [(t, t+5) for t in [2000, 2005, 2010, 2015]
                    if t in years and t+5 in years]
    print(f"Evaluation windows: {eval_windows}\n")

    all_rows = []
    for model_name, phi in models.items():
        for t_start, t_end in eval_windows:
            rca_t  = make_rca_matrix(panel, t_start, countries, products)
            rca_t5 = make_rca_matrix(panel, t_end,   countries, products)

            if model_name == "Random":
                np.random.seed(42)
                density = np.random.rand(*rca_t.shape)
            else:
                density = compute_density(rca_t, phi)

            res = evaluate_model(density, rca_t, rca_t5)
            if res is None:
                continue

            row = {"model": model_name, "t_start": t_start, "t_end": t_end}
            row.update(res)
            all_rows.append(row)

            print(f"{model_name:20s} {t_start}→{t_end}: "
                  f"AUC={res['auc']:.4f}  P@10={res.get('p@10',0):.4f}  "
                  f"P@50={res.get('p@50',0):.4f}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(RESULTS_DIR, "full_comparison.csv"), index=False)

    summary = results_df.groupby("model")[["auc","p@10","p@20","p@50"]].mean().round(4)
    summary = summary.sort_values("auc", ascending=False)

    print("MEAN RESULTS ACROSS EVALUATION WINDOWS")
    print(summary.to_string())
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"))

    model_order = [m for m in ["Hidalgo","MF-d32","MF-d64","MF-d128","node2vec-d64","Random"]
                   if m in results_df["model"].unique()]
    colors_map = {
        "Hidalgo": "#1e40af", "MF-d32": "#f59e0b", "MF-d64": "#10b981",
        "MF-d128": "#6366f1", "node2vec-d64": "#ef4444", "Random": "#9ca3af"
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Product Space: Learned Embeddings vs. Hidalgo Baseline", fontsize=13, fontweight="bold")

    ax = axes[0]
    for m in model_order:
        sub = results_df[results_df["model"] == m].sort_values("t_start")
        ax.plot(sub["t_start"], sub["auc"], marker="o", label=m,
                color=colors_map.get(m, "#333"), linewidth=2)
    ax.set_xlabel("Prediction Start Year")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AUC by Prediction Window")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.01))
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    k_cols = ["p@10", "p@20", "p@50", "p@100"]
    k_cols = [c for c in k_cols if c in results_df.columns]
    mean_pk = results_df.groupby("model")[k_cols].mean()
    x = np.arange(len(k_cols))
    width = 0.8 / len(model_order)
    for i, m in enumerate(model_order):
        if m not in mean_pk.index:
            continue
        vals = mean_pk.loc[m, k_cols].values
        ax.bar(x + i*width - 0.4 + width/2, vals, width,
               label=m, color=colors_map.get(m, "#333"), alpha=0.85)
    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K")
    ax.set_title("Mean Precision@K Across Windows")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("p@","") for k in k_cols])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()