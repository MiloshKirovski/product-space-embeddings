# Reproduces Hildago et al (2007) product space baseline
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, precision_score

OUT_DIR = "data/processed"
RESULTS_DIR = "results"

def compute_proximity(rca_matrix, products):
    # Ф(p,q) = min(P(RCA_p=1|RCA_q=1), P(RCA_q=1 | RCA_p=1))
    x_p = rca_matrix.sum(axis=0)
    co = rca_matrix.T @ rca_matrix

    # Ф(p,q) = co(p,q) / max(x_p, x_q)
    # P(q|p) = co(p,q)/x_p,
    # P(p|q) = co(p,q)/x_q
    # Ф = min of the two = co / max(x_p, x_q)
    x_outer_max = np.maximum(x_p[:, None], x_p[None, :])
    x_outer_max = np.where(x_outer_max == 0, 1, x_outer_max)
    phi = co / x_outer_max

    np.fill_diagonal(phi, 0)
    return pd.DataFrame(phi, index=products, columns=products)


def compute_density(rca_binary, phi):
    # density(c, p) = SUM_p [Ф(p,q) * rca(c,q)] / SUM(Ф(p,q))
    phi_sum = phi.sum(axis=1)
    phi_sum = np.where(phi_sum == 0, 1, phi_sum)

    numerator = rca_binary @ phi
    density = numerator / phi_sum[None, :]
    return density


def make_prediction_targets(rca_t, rca_t5):
    candidates = (rca_t == 0)
    gained = ((rca_t == 0) & (rca_t5 == 1))
    return candidates, gained


def evaluate(density, rca_t, rca_t5, k_vals=(10, 20, 50)):
    # Calculates AUC and Precision @ K
    candidates = (rca_t == 0)
    gained = ((rca_t == 0) & (rca_t5 == 1))

    # Score is our calculated density in time period t and label gives us the outcome at t+x
    # We flatten because we do not care about (country, product) pairs here
    scores = density[candidates].flatten()
    labels = gained[candidates].flatten()

    if labels.sum() == 0:
        return {}

    results = {}
    results["auc"] = roc_auc_score(labels, scores)
    results["n_entries"] = int(labels.sum())
    results["n_candidates"] = int(len(labels))

    order = np.argsort(-scores)
    for k in k_vals:
        topk = order[:k]
        results[f"p@{k}"] = labels[topk].mean()

    np.random.seed(42)
    rand_scores = np.random.rand(len(labels))
    results["auc_random"] = roc_auc_score(labels, rand_scores)

    return results


def main():
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))

    years = sorted(panel["year"].unique())
    countries = sorted(panel["country"].unique())
    products = sorted(panel["product"].unique())

    def make_rca_matrix(year):
        sub = panel[panel["year"] == year][["country", "product", "rca_binary"]]
        return sub.pivot_table(
            index="country",
            columns="product",
            values="rca_binary",
            fill_value=0
        ).reindex(index=countries, columns=products, fill_value=0).values.astype(np.float32)

    train_years = [y for y in years if y <= 2005]
    print("Computing proximity from training years")
    rca_train_avg = np.mean([make_rca_matrix(y) for y in train_years], axis=0)
    phi_df = compute_proximity(rca_train_avg, products)
    phi = phi_df.values
    phi_df.to_parquet(os.path.join(OUT_DIR, "proximity_hildago.parquet"))
    print(f"mean Ф={phi.mean():.4f}, max Ф={phi.max():.4f}")

    eval_windows = [(t, t + 5) for t in [2000, 2005, 2010, 2015] if t in years and t + 5 in years]
    print("Evaluating on windows")

    all_results = []
    for t_start , t_end in eval_windows:
        rca_t = make_rca_matrix(t_start)
        rca_t5 = make_rca_matrix(t_end)
        density = compute_density(rca_t, phi)

        res = evaluate(density, rca_t, rca_t5)
        res["t_start"] = t_start
        res["t_end"] = t_end
        all_results.append(res)

        print(f"{t_start}->{t_end}: AUC={res.get('auc', 0):.4f} "
              f"(random={res.get('auc_random', 0):.4f})"
              f"P@10={res.get('p@10', 0):.4f}  "
              f"entries={res.get('n_entries', 0)}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "hildago_results.csv"), index=False)

    summary = results_df[["t_start", "t_end", "auc", "auc_random", "p@10", "p@20", "n_entries"]].to_string(index=False)
    with open(os.path.join(RESULTS_DIR, "baseline_summary.txt"), "w") as f:
        f.write("HIDALGO BASELINE RESULTS\n")
        f.write(summary + "\n\n")
        f.write(f"Mean AUC: {results_df['auc'].mean():.4f}\n")


if __name__ == "__main__":
    main()