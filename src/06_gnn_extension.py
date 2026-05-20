import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

OUT_DIR = "data/processed"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

TOP_K = 20
HIDDEN_DIM = 64
OUT_DIM = 64
DROPOUT = 0.1
N_EPOCHS = 1000
LR = 1e-3

NEG_RATIO = 5
TRAIN_WINDOW = (2000, 2005)

EVAL_WINDOWS = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_topk_graph(phi, k):
    n = phi.shape[0]
    pm = phi.copy()
    np.fill_diagonal(pm, -np.inf)
    topk_idx = np.argpartition(-pm, k, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = topk_idx.flatten()

    src_full = np.concatenate([src, dst])
    dst_full = np.concatenate([dst, src])

    edges = np.unique(np.stack([src_full, dst_full]), axis=1)

    return torch.tensor(edges, dtype=torch.long)


class ProductGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = dropout
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        return F.normalize(h, dim=1)


def make_rca_matrix(panel, year, countries, products):
    c_idx = {c: i for i, c in enumerate(countries)}
    p_idx = {p: i for i, p in enumerate(products)}
    mat = np.zeros((len(countries), len(products)), dtype=np.float32)
    sub = panel[panel["year"] == year][["country", "product", "rca_binary"]]

    for row in sub.itertuples(index=False):
        if row.country in c_idx and row.product in p_idx:
            mat[c_idx[row.country], p_idx[row.product]] = row.rca_binary

    return mat


def compute_scores(model, x, edge_index, rca_t):
    E = model(x, edge_index)
    basket = rca_t.sum(1, keepdim=True).clamp(min=1)
    centroids = F.normalize((rca_t @ E) / basket, dim=1)
    scores = (centroids @ E.T) * model.temperature

    return scores, E


def train_step(model, x, edge_index, rca_t, rca_t5, optimizer, neg_ratio):
    model.train()
    optimizer.zero_grad()
    scores, _ = compute_scores( model, x, edge_index, rca_t)
    candidate = (rca_t == 0)
    positive = candidate & (rca_t5 == 1)
    negative = candidate & (rca_t5 == 0)

    pos_scores = scores[positive]
    neg_idx = negative.nonzero(as_tuple=False)
    n_neg = min(neg_ratio * pos_scores.numel(), neg_idx.size(0))
    perm = torch.randperm(neg_idx.size(0), device=scores.device)[:n_neg]
    neg_scores = scores[neg_idx[perm, 0], neg_idx[perm, 1]]

    logits = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate_window(model, x, edge_index, rca_t, rca_t5, k_vals=(10, 20, 50, 100)):
    model.eval()
    scores, _ = compute_scores(model, x, edge_index, rca_t)

    scores = scores.cpu().numpy()

    rca_t_np = rca_t.cpu().numpy()
    rca_t5_np = rca_t5.cpu().numpy()

    cand = (rca_t_np == 0)
    gained = cand & (rca_t5_np == 1)

    s = scores[cand]
    y = gained[cand]

    res = {
        "auc": roc_auc_score(y, s),
        "n_entries": int(y.sum())
    }

    order = np.argsort(-s)
    for k in k_vals:
        res[f"p@{k}"] = float(y[order[:k]].mean())

    return res


def main():
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))
    products = np.load(os.path.join(OUT_DIR, "product_list.npy"), allow_pickle=True).tolist()
    countries = sorted(panel["country"].unique())

    mf_emb = np.load(os.path.join(OUT_DIR, "embeddings_mf_d64.npy"))
    phi_df = pd.read_parquet(os.path.join(OUT_DIR, "proximity_hildago.parquet"))

    full_products = phi_df.index.tolist()
    full_idx = {p: i for i, p in enumerate(full_products)}

    idx_sub = [full_idx[p] for p in products if p in full_idx]
    phi = phi_df.values[np.ix_(idx_sub, idx_sub)]

    print(f"Products: {len(products)}, countries: {len(countries)}")
    print(f"MF emb: {mf_emb.shape}, phi: {phi.shape}")

    edge_index = build_topk_graph(phi, k=TOP_K).to(DEVICE)

    print(f"Graph edges: {edge_index.size(1)} (top-{TOP_K} symmetrised)")

    x = torch.tensor(normalize(mf_emb, norm="l2"), dtype=torch.float32, device=DEVICE)
    rca_t = torch.tensor(make_rca_matrix(panel, TRAIN_WINDOW[0], countries, products),
                         dtype=torch.float32, device=DEVICE)
    rca_t5 = torch.tensor(make_rca_matrix(panel, TRAIN_WINDOW[1], countries, products),
                          dtype=torch.float32, device=DEVICE)

    model = ProductGNN(in_dim=mf_emb.shape[1], hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {N_EPOCHS} epochs on {TRAIN_WINDOW[0]}->{TRAIN_WINDOW[1]}")

    for epoch in range(N_EPOCHS):
        loss = train_step(model, x, edge_index, rca_t, rca_t5, optimizer, NEG_RATIO)

        if (epoch + 1) % 10 == 0:
            res = evaluate_window(model, x, edge_index, rca_t, rca_t5 )
            print( f"epoch {epoch+1:3d} loss={loss:.4f} train AUC={res['auc']:.4f} P@20={res['p@20']:.3f}")

    model.eval()
    with torch.no_grad():
        E_gnn = model(x, edge_index).cpu().numpy()

    out_path = os.path.join(OUT_DIR, "embeddings_gnn_d64.npy")
    np.save(out_path, E_gnn)
    print(f"\nRefined embeddings shape {E_gnn.shape})")

    print("\nEvaluation across all windows:")

    rows = []

    for t_start, t_end in EVAL_WINDOWS:
        rca_a = torch.tensor(make_rca_matrix(panel, t_start, countries, products),
                             dtype=torch.float32, device=DEVICE)
        rca_b = torch.tensor(make_rca_matrix(panel, t_end, countries, products), dtype=torch.float32, device=DEVICE)
        res = evaluate_window(model, x, edge_index, rca_a, rca_b)

        rows.append({
            "model": "GNN-d64",
            "t_start": t_start,
            "t_end": t_end,
            **res
        })

        print(f"  {t_start}->{t_end}: AUC={res['auc']:.4f} P@10={res['p@10']:.3f} P@20={res['p@20']:.3f} "
            f"P@50={res['p@50']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "gnn_results.csv"), index=False)
    print(f"\nMean AUC: {df['auc'].mean():.4f}    Mean P@20: {df['p@20'].mean():.3f}")


if __name__ == "__main__":
    main()