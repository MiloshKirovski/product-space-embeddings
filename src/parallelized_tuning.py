import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from itertools import product
import time
from datetime import datetime
from multiprocessing import Pool, Manager
from functools import partial
import json
import threading

OUT_DIR = "data/processed"
RESULTS_DIR = "results/grid_search"
os.makedirs(RESULTS_DIR, exist_ok=True)

MIN_UBIQUITY = 5

GRID_PARAMS = {
    'epochs': [5, 10, 15],
    'walk_length': [20, 30, 40, 50],
    'window': [3, 5, 7, 10],
    'n_walks': [10, 15, 20, 30],
    'd': [32, 64, 128],
    'neg_samples': [5, 10, 15]
}

DEFAULT_PARAMS = {
    'lr': 0.025,
    'min_ubiquity': MIN_UBIQUITY
}


class BipartiteRandomWalker:
    def __init__(self, rca_matrix, weights, walk_length=30, n_walks=15):
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

    def generate_walks(self):
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
    np.random.seed(42)
    E = np.random.randn(n_products, d).astype(np.float32) * 0.01
    E_ctx = np.random.randn(n_products, d).astype(np.float32) * 0.01

    freq = np.zeros(n_products)
    for walk in walks:
        for node in walk:
            freq[node] += 1
    freq = freq ** 0.75
    freq = freq / (freq.sum() + 1e-8)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    for epoch in range(epochs):
        np.random.shuffle(walks)
        loss_total, n_pairs = 0.0, 0
        for walk in walks:
            for i, center in enumerate(walk):
                ctx_start = max(0, i - window)
                ctx_end = min(len(walk), i + window + 1)
                for j in range(ctx_start, ctx_end):
                    if i == j:
                        continue
                    ctx = walk[j]
                    score = np.dot(E[center], E_ctx[ctx])
                    grad = lr * (1 - sigmoid(score))
                    E[center] += grad * E_ctx[ctx]
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

    return normalize(E, norm="l2")


def filter_products_by_ubiquity(panel, train_years, min_ubiquity):
    train = panel[panel["year"].isin(train_years)]
    ubiquity = ((train.groupby(["product", "year"]))["rca_binary"]
                .sum().groupby("product").mean())
    kept = ubiquity[ubiquity >= min_ubiquity].index.tolist()
    print(f"Kept {len(kept)}/{len(ubiquity)} products (ubiquity >= {min_ubiquity})")
    return sorted(kept)


def build_rca_matrix(panel, train_years, countries, products):
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
    basket_sizes = (rca_matrix >= 0.5).sum(axis=1).astype(float)
    weights = 1.0 / np.log1p(np.maximum(basket_sizes, 1))
    weighted = rca_matrix * weights[:, None]
    return weighted, weights


def make_rca_matrix_single(panel, year, countries, products):
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


def evaluate_auc(density, rca_t, rca_t5):
    candidates = (rca_t == 0)
    gained = ((rca_t == 0) & (rca_t5 == 1))
    scores = density[candidates]
    labels = gained[candidates]
    if labels.sum() == 0:
        return None
    return roc_auc_score(labels, scores)


def phi_from_embeddings(embeddings):
    E = normalize(embeddings, norm="l2")
    phi = E @ E.T
    np.fill_diagonal(phi, 0)
    phi = np.clip(phi, 0, None)
    return phi


def train_node2vec_with_params(rca_matrix, weights, params):
    walker = BipartiteRandomWalker(
        rca_matrix, weights,
        walk_length=params['walk_length'],
        n_walks=params['n_walks']
    )
    walks = walker.generate_walks()
    print(f"Generated {len(walks)} walks")

    embeddings = train_skipgram(
        walks, rca_matrix.shape[1],
        d=params['d'],
        window=params['window'],
        epochs=params['epochs'],
        lr=DEFAULT_PARAMS['lr'],
        neg_samples=params['neg_samples']
    )
    return embeddings


def evaluate_and_save(params_tuple, idx, total_runs, rca_train, weights, panel, countries, products, eval_windows,
                      results_csv, csv_lock):
    param_names = list(GRID_PARAMS.keys())
    params = dict(zip(param_names, params_tuple))

    print(f"[{idx}/{total_runs}] Testing: {params}")
    start_time = time.time()

    try:
        embeddings = train_node2vec_with_params(rca_train, weights, params)
        phi = phi_from_embeddings(embeddings)

        auc_scores = []
        window_results = {}

        for t_start, t_end in eval_windows:
            rca_t = make_rca_matrix_single(panel, t_start, countries, products)
            rca_t5 = make_rca_matrix_single(panel, t_end, countries, products)
            density = compute_density(rca_t, phi)
            auc = evaluate_auc(density, rca_t, rca_t5)
            if auc is not None:
                auc_scores.append(auc)
                window_results[f'auc_{t_start}_{t_end}'] = auc

        if auc_scores:
            mean_auc = np.mean(auc_scores)
            std_auc = np.std(auc_scores)

            result = {
                **params,
                'mean_auc': mean_auc,
                'std_auc': std_auc,
                'min_auc': min(auc_scores),
                'max_auc': max(auc_scores),
                'n_windows': len(auc_scores),
                'time_seconds': time.time() - start_time,
                'error': None,
                **window_results
            }
            print(f"AUC={mean_auc:.4f} (±{std_auc:.4f})")

            with csv_lock:
                result_df = pd.DataFrame([result])
                if not os.path.exists(results_csv):
                    result_df.to_csv(results_csv, index=False)
                else:
                    result_df.to_csv(results_csv, mode='a', header=False, index=False)

            return result
        else:
            print(f"No valid evaluations")
            result = {
                **params,
                'mean_auc': None,
                'std_auc': None,
                'min_auc': None,
                'max_auc': None,
                'n_windows': 0,
                'time_seconds': time.time() - start_time,
                'error': "No valid evaluations",
                **{f'auc_{t_start}_{t_end}': None for t_start, t_end in eval_windows}
            }

            with csv_lock:
                result_df = pd.DataFrame([result])
                if not os.path.exists(results_csv):
                    result_df.to_csv(results_csv, index=False)
                else:
                    result_df.to_csv(results_csv, mode='a', header=False, index=False)

            return result

    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        result = {
            **params,
            'mean_auc': None,
            'std_auc': None,
            'min_auc': None,
            'max_auc': None,
            'n_windows': 0,
            'time_seconds': time.time() - start_time,
            'error': str(e)[:200],
            **{f'auc_{t_start}_{t_end}': None for t_start, t_end in eval_windows}
        }

        with csv_lock:
            result_df = pd.DataFrame([result])
            if not os.path.exists(results_csv):
                result_df.to_csv(results_csv, index=False)
            else:
                result_df.to_csv(results_csv, mode='a', header=False, index=False)

        return result


def progress_monitor(total_runs, results_csv, stop_flag):
    completed = 0
    last_count = 0

    while not stop_flag.is_set():
        if os.path.exists(results_csv):
            try:
                df = pd.read_csv(results_csv)
                completed = len(df)
                if completed > last_count:
                    print(
                        f"\n PROGRESS: {completed}/{total_runs} configurations completed ({completed / total_runs * 100:.1f}%)")
                    last_count = completed
            except:
                pass

        if completed >= total_runs:
            break

        time.sleep(5)


def main():
    print("NODE2VEC GRID SEARCH FOR AUC OPTIMIZATION - PARALLEL VERSION")

    n_cores = 8
    print(f"\nUsing {n_cores} CPU cores for parallel processing")

    print("\n1. Loading data")
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))
    years = sorted(panel["year"].unique())
    countries = sorted(panel["country"].unique())
    print(f"{len(countries)} countries, years {years[0]}-{years[-1]}")

    train_years = [y for y in years if y <= 2005]
    print(f"\n2. Preparing training data (years {train_years[0]}-{train_years[-1]})...")

    products = filter_products_by_ubiquity(panel, train_years, MIN_UBIQUITY)
    print(f"\n3. Building RCA matrix")
    rca_train = build_rca_matrix(panel, train_years, countries, products)
    print(f"Shape: {rca_train.shape}, density: {rca_train.mean():.4f}")

    print(f"\n4. Applying country weighting")
    rca_weighted, weights = apply_country_weighting(rca_train)

    eval_windows = [(t, t + 5) for t in [2000, 2005, 2010, 2015]
                    if t in years and t + 5 in years]
    print(f"\n5. Evaluation windows: {eval_windows}")

    param_names = list(GRID_PARAMS.keys())
    param_values = list(GRID_PARAMS.values())
    combinations = list(product(*param_values))
    total_runs = len(combinations)

    print(f"\n6. Grid search: {total_runs} configurations to test")
    print(f"Parameters: {param_names}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = os.path.join(RESULTS_DIR, f"grid_search_results_parallel_{timestamp}.csv")

    window_columns = [f'auc_{t_start}_{t_end}' for t_start, t_end in eval_windows]
    headers = list(param_names) + ['mean_auc', 'std_auc', 'min_auc', 'max_auc',
                                   'n_windows', 'time_seconds', 'error'] + window_columns
    headers_df = pd.DataFrame(columns=headers)
    headers_df.to_csv(results_csv, index=False)

    manager = Manager()
    csv_lock = manager.Lock()
    stop_flag = manager.Event()

    monitor_thread = threading.Thread(target=progress_monitor, args=(total_runs, results_csv, stop_flag))
    monitor_thread.daemon = True
    monitor_thread.start()

    print(f"\n7. Starting parallel processing with {n_cores} workers...")
    start_total_time = time.time()

    eval_func = partial(
        evaluate_and_save,
        rca_train=rca_train,
        weights=weights,
        panel=panel,
        countries=countries,
        products=products,
        eval_windows=eval_windows,
        results_csv=results_csv,
        csv_lock=csv_lock
    )

    args_list = [(combo, idx, total_runs) for idx, combo in enumerate(combinations, 1)]

    with Pool(processes=n_cores) as pool:
        results = pool.starmap(eval_func, args_list)

    stop_flag.set()
    monitor_thread.join(timeout=10)

    total_time = time.time() - start_total_time
    print(f"\n\nParallel processing completed in {total_time:.2f} seconds")

    results_df = pd.DataFrame(results)

    valid_results = results_df[results_df['mean_auc'].notna()].copy()
    if len(valid_results) > 0:
        best_row = valid_results.loc[valid_results['mean_auc'].idxmax()]
        best_auc = best_row['mean_auc']
        best_params = {param: best_row[param] for param in param_names}
    else:
        best_auc = -1
        best_params = None

    print("GRID SEARCH COMPLETE")

    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per configuration: {total_time / total_runs:.2f} seconds")

    if best_params:
        print(f"\nBest configuration found:")
        print(f"Mean AUC: {best_auc:.4f}")
        for param, value in best_params.items():
            print(f"{param}: {value}")

        print("\nTop 10 configurations by mean AUC:")
        if len(valid_results) > 0:
            top10 = valid_results.nlargest(10, 'mean_auc')
            display_cols = param_names + ['mean_auc', 'std_auc', 'time_seconds']
            print(top10[display_cols].to_string(index=False))

        best_params_file = os.path.join(RESULTS_DIR, f"best_params_parallel_{timestamp}.json")
        with open(best_params_file, 'w') as f:
            json.dump({
                'best_auc': float(best_auc),
                'best_params': best_params,
                'timestamp': timestamp,
                'parallel_execution': True,
                'n_cores': n_cores,
                'total_time_seconds': total_time
            }, f, indent=2)

        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        final_embeddings = train_node2vec_with_params(rca_train, weights, best_params)
        final_phi = phi_from_embeddings(final_embeddings)

        final_emb_path = os.path.join(OUT_DIR, f"embeddings_node2vec_best_parallel_{timestamp}.npy")
        np.save(final_emb_path, final_embeddings)
        print(f"Final embeddings saved to: {final_emb_path}")

        print("\nFinal evaluation with best parameters:")
        for t_start, t_end in eval_windows:
            rca_t = make_rca_matrix_single(panel, t_start, countries, products)
            rca_t5 = make_rca_matrix_single(panel, t_end, countries, products)
            density = compute_density(rca_t, final_phi)
            auc = evaluate_auc(density, rca_t, rca_t5)
            if auc is not None:
                print(f"{t_start}->{t_end}: AUC={auc:.4f}")
    else:
        print("\nNo valid results obtained!")


if __name__ == "__main__":
    main()