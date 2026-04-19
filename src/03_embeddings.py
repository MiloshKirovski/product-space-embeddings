import os
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_UBIQUITY = 5

def filter_products_by_ubiquity(panel, train_years, min_ubiquity):
    train = panel[panel["year"].isin(train_years)]
    ubiquity = ((train.groupby(["product", "year"]))["rca_binary"]
                .sum().groupby("product").mean())
    kept = ubiquity[ubiquity >= min_ubiquity].index.tolist()
    print(f"After ubiquity filter kept {len(kept) / len(ubiquity)}")
    print(f"Dropped {len(ubiquity)-len(kept)} with avg ubiquity < {min_ubiquity})")

    return sorted(kept)

def build_rca_matrix(panel, train_years, countries, products):
    # [countries x products] where a value of 0.8 would mean a country had RCA>=i in 8/10 training years
    c_idx = {c: i for i, c in enumerate(countries)}
    p_idx = {p: i for i, p in enumerate(products)}
    train = panel[panel["year"].isin(train_years)]

    mats = []
    for yr in train_years:
        mat = np.zeros((len(countries), len(products)), dtype=np.float32)
        yr_data = train[train["year"] == yr][["country", "product", "rca_binary"]]

        for row in yr_data.itertuples(index=False):
            if row.country in c_idx and row.product in p_idx:
                mat[c_idx[row.country], p_idx[row.product]] = row.rca_binary
        mats.append(mat)

    return np.mean(mats, axis=0)


def apply_country_weighting(rca_matrix):
    # Downweight frequent countries by 1/log(1 + basket_size).
    basket_sizes = (rca_matrix >= 0.5).sum(axis=1).astype(float)
    weights = 1.0 / np.log1p(np.maximum(basket_sizes, 1))
    weighted = rca_matrix * weights[:, None]
    return weighted


def train_mf(rca_matrix, d):
    # Products that co-occur across the same countries end up close in the space
    svd = TruncatedSVD(n_components=d, random_state=42)
    svd.fit(rca_matrix)
    embeddings = svd.components_.T # [n_products x d]
    explained = svd.explained_variance_ratio_.sum()
    print(f"d={d}: explained variance = {explained:.2f}")
    return embeddings


class BipartiteRandomWalker:
    # Walk path: country -> product -> country -> product ...
    # Then we record only product nodes, those are what we need embeddings for
    # Transition from product to country is weighted by that country's IDF weight

    def __init__(self, rca_matrix, weights, walk_length=30, n_walks = 15):
        self.rca = rca_matrix
        self.n_countries, self.n_products = rca_matrix.shape
        self.walk_length = walk_length
        self.n_walks = n_walks

        self.country_to_products = {
            c: np.where(rca_matrix[c] > 0.5)[0]
            for c in range(self.n_countries)
        }

        self.product_to_countries = {}
        self.product_transition_probs = {}
        for p in range(self.n_products):
            exporters = np.where(rca_matrix[:, p] > 0.5)[0]
            if len(exporters) == 0:
                continue
            w = weights[exporters]
            w = w / w.sum()
            self.product_to_countries[p] = exporters
            self.product_transition_probs[p] = w


    def walk_from_country(self, start_country):
        walk_products = []
        country = start_country

        for _ in range(self.walk_length):
            prods = self.country_to_products.get(country, [])
            if len(prods) == 0:
                break
            product = np.random.choice(prods)
            walk_products.append(product)
            countries = self.product_to_countries.get(product)
            if countries is None or len(countries) == 0:
                break
            probs = self.product_transition_probs[product]
            country = np.random.choice(countries, p=probs)

        return walk_products

    def generate_walks(self) -> list:
        walks = []
        for c in range(self.n_countries):
            if len(self.country_to_products.get(c, [])) == 0:
                continue
            for _ in range(self.n_walks):
                w = self.walk_from_country(c)
                if len(w) > 1:
                    walks.append(w)

        np.random.shuffle(walks)
        return walks


def train_skipgram(walks, n_products, d=64, window=5, epochs=5, lr=0.025, neg_samples=5):
    # We learn embeddings such that products appearing near each other in walks end up close in vector space
    np.random.seed(42)
    E = np.random.randn(n_products, d).astype(np.float32) * 0.01
    E_ctx = np.random.randn(n_products, d).astype(np.float32) * 0.01

    # Negative sampling distribution: freq^0.75 (standard word2vec)
    freq = np.zeros(n_products)
    for walk in walks:
        for node in walk:
            freq[node] += 1
    freq = freq ** 0.75
    freq /= freq.sum()

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))


    for epoch in range(epochs):
        np.random.shuffle(walks)
        loss_total, n_pairs = 0.0, 0
        for walk in walks:
            for i, center in enumerate(walk):
                ctx_start = max(0, i-window)
                ctx_end = min(len(walk), i + window + 1)
                for j in range(ctx_start, ctx_end):
                    if i == j:
                        continue
                    ctx = walk[j]
                    score = np.dot(E[center], E_ctx[ctx])
                    grad  = lr * (1 - sigmoid(score))
                    E[center]  += grad * E_ctx[ctx]
                    E_ctx[ctx] += grad * E[center]
                    loss_total -= np.log(sigmoid(score) + 1e-9)
                    negs = np.random.choice(n_products, neg_samples, p=freq)
                    for neg in negs:
                        score_neg = np.dot(E[center], E_ctx[neg])
                        grad_neg = lr * (-sigmoid(score_neg))
                        E[center] += grad_neg * E_ctx[neg]
                        E_ctx[neg] += grad_neg * E[center]
                        loss_total -= np.log(1 - sigmoid(score_neg) + 1e-9)
                    n_pairs += 1
        if n_pairs > 0:
            print(f"Epoch {epoch + 1}/{epochs}  loss={loss_total / n_pairs:.4f}")

    return normalize(E, norm="l2")


def train_node2vec(rca_matrix, weights, d=64):
    walker = BipartiteRandomWalker(rca_matrix, weights, walk_length=15, n_walks=30)
    walks = walker.generate_walks()
    print(f"Generated {len(walks)} walks")
    return train_skipgram(walks, rca_matrix.shape[1], d=d, epochs=10)

EMBEDDING_DIMS = [32, 64, 128]

def main():
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))
    years = sorted(panel["year"].unique())
    countries = sorted(panel["country"].unique())
    print(f"{len(countries)} countries, years {years[0]}-{years[-1]}")

    train_years = [y for y in years if y <= 2005]

    print(f"\nFiltering products (ubiquity >= {MIN_UBIQUITY})...")
    products = filter_products_by_ubiquity(panel, train_years, MIN_UBIQUITY)

    product_index = pd.DataFrame({
        "product": products,
        "idx": np.arange(len(products), dtype=np.int32)
    })
    product_index.to_csv(
        os.path.join(OUT_DIR, "product_index.csv"),
        index=False
    )
    np.save(os.path.join(OUT_DIR, "product_list.npy"), np.array(products))

    print(f"\nBuilding training RCA matrix")
    rca_train = build_rca_matrix(panel, train_years, countries, products)
    print(f"Shape: {rca_train.shape},  density: {rca_train.mean()}")

    print("\nApplying country weighting (IDF-style)")
    rca_weighted = apply_country_weighting(rca_train)
    basket_sizes = (rca_train > 0.5).sum(axis=1)
    print(f"Basket size -- mean: {basket_sizes.mean()}, "
          f"median: {np.median(basket_sizes)}, "
          f"max: {basket_sizes.max()}")

    weights = 1.0 / np.log1p(np.maximum(basket_sizes.astype(float), 1))

    for d in EMBEDDING_DIMS:
        emb = train_mf(rca_weighted, d=d)
        np.save(os.path.join(OUT_DIR, f"embeddings_mf_d{d}.npy"), emb)
        np.save(os.path.join(OUT_DIR, f"embeddings_products_d{d}.npy"), np.array(products))

    print(f"Saved MF embeddings for d={EMBEDDING_DIMS}")

    car_codes = [p for p in products if str(p).startswith("8703")]
    if car_codes:
        p_idx  = {p: i for i, p in enumerate(products)}
        emb_64 = np.load(os.path.join(OUT_DIR, "embeddings_mf_d64.npy"))
        for code in car_codes[:1]:
            sims = cosine_similarity(emb_64[p_idx[code]:p_idx[code]+1], emb_64)[0]
            top5 = np.argsort(-sims)[1:6]
            print(f"\nNearest neighbours to {code} (cars) in MF-d64 space:")
            for idx in top5:
                print(f"{products[idx]}  sim={sims[idx]:.3f}")

    print("\n[B] node2vec (IDF-weighted transitions, d=64)...")
    emb_n2v = train_node2vec(rca_train, weights, d=64)
    np.save(os.path.join(OUT_DIR, "embeddings_node2vec_d64.npy"), emb_n2v)
    np.save(os.path.join(OUT_DIR, "embeddings_node2vec_products.npy"), np.array(products))

    print("Saved node2vec embeddings")


if __name__ == "__main__":
    main()