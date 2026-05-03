import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "data/processed"
RAW_DIR = "data/raw"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"

WB_GDP_CSV = os.path.join(RAW_DIR, "API_NY.GDP.PCAP.KD_DS2_en_csv_v2_89/API_NY.GDP.PCAP.KD_DS2_en_csv_v2_89.csv")
EVAL_WINDOWS = [(2000, 2005), (2005, 2010), (2010, 2015), (2015, 2020)]

def load_core():
    panel = pd.read_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))
    products = np.load(os.path.join(OUT_DIR, "product_list.npy"), allow_pickle=True).tolist()

    emb = np.load(os.path.join(OUT_DIR, "embeddings_mf_d64.npy"))
    emb = normalize(emb, norm="l2")

    countries = sorted(panel["country"].unique())
    years = sorted(panel["year"].unique())
    print("Embedding shape:", emb.shape)
    return panel, products, emb, countries, years


def make_rca_matrix(panel, year, countries, products):
    c_idx = {c: i for i, c in enumerate(countries)}
    p_idx = {p: i for i, p in enumerate(products)}
    mat = np.zeros((len(countries), len(products)), dtype=np.float32)
    sub = panel[panel["year"] == year][["country", "product", "rca_binary"]]

    for row in sub.itertuples(index=False):
        if row.country in c_idx and row.product in p_idx:
            mat[c_idx[row.country], p_idx[row.product]] = row.rca_binary

    return mat


def basket_centroids(rca_mat, emb):
    # Mean embedding of products where RCA>=1 [C x D]
    sizes = rca_mat.sum(axis=1, keepdims=True)
    sizes = np.where(sizes == 0, 1, sizes)
    return (rca_mat @ emb) / sizes


def load_wb_gdp(countries):
    raw = pd.read_csv(WB_GDP_CSV, skiprows=4)
    year_cols = [c for c in raw.columns if c.isdigit() and 1990 <= int(c) <= 2024]
    long = (raw[["Country Code"] + year_cols]
            .melt(id_vars="Country Code", var_name="year", value_name="gdp_pc")
            .rename(columns={"Country Code": "iso3"}))
    long["year"]   = long["year"].astype(int)
    long["gdp_pc"] = pd.to_numeric(long["gdp_pc"], errors="coerce")
    long = long.dropna(subset=["gdp_pc"])

    import pycountry
    def iso3_to_m49(iso3):
        c = pycountry.countries.get(alpha_3=iso3)
        return int(c.numeric) if c else None
    long["country"] = long["iso3"].map(iso3_to_m49)

    long = long.dropna(subset=["country"])
    long["country"] = long["country"].astype(type(countries[0]))
    mapped = long[long["country"]. isin(countries)]
    print(f"GDP: {mapped['country'].nunique()} countries matched - {len(mapped)}")

    return mapped[["country", "year", "gdp_pc"]]


def build_dataset(panel, products, emb, countries, years, gdp):
    # For every (country, window) we get the centroid_t, centroid_t5; magnitude of basket shift;
    # cos(magnitude, complexity_axis)-direction toward complexity; centroid_norm_t;
    # log(gdp_t), log(gdp_t5); log_gdp_growth=log(gdp_t5)/log(gdp_t)

    needed_years = {y for w in EVAL_WINDOWS for y in w}
    cents = {}
    for yr in sorted(years):
        if yr in years:
            rca = make_rca_matrix(panel, yr, countries, products)
            cents[yr] = basket_centroids(rca, emb)

    # Global complexity axis = first principal component of country centroids,
    # which empirically aligns with the development (income–complexity) gradient
    # (We use the first PC of all year-0 centroids as a data-driven proxy)
    all_c0 = np.vstack([cents[w[0]] for w in EVAL_WINDOWS if w[0] in cents])
    pca1 = PCA(n_components=1).fit(all_c0)
    complexity_axis = normalize(pca1.components_)               # [1 x 64]

    rows = []
    for t_start, t_end in EVAL_WINDOWS:
        if t_start not in cents or t_end not in cents:
            continue
        C_t = cents[t_start]
        C_t5 = cents[t_end]
        delta = C_t5 - C_t
        mag = np.linalg.norm(delta, axis=1)
        cos_c = (delta @ complexity_axis.T).squeeze() / (mag + 1e-9)
        norm_t = np.linalg.norm(C_t, axis=1)

        for i, country in enumerate(countries):
            row = {
                "country": country,
                "t_start": t_start,
                "t_end": t_end,
                "emb_magnitude": float(mag[i]),
                "emb_cos_complex": float(cos_c[i]),
                "emb_centroid_norm": float(norm_t[i]),
            }

            for d in range(16):
                # Because dimensions are ordered in SVD we can truncate for the regression
                row[f"c{d}"] = float(C_t[i, d])
                row[f"d{d}"] = float(delta[i, d])
            rows.append(row)

    df = pd.DataFrame(rows)

    g_t = gdp.rename(columns={"gdp_pc": "gdp_t", "year": "t_start"})
    g_t5 = gdp.rename(columns={"gdp_pc": "gdp_t5", "year": "t_end"})
    df = df.merge(g_t[["country", "t_start", "gdp_t"]], on=["country", "t_start"], how="left")
    df = df.merge(g_t5[["country", "t_end", "gdp_t5"]], on=["country", "t_end"], how="left")
    df["log_gdp_t"] = np.log(df["gdp_t"])
    df["log_gdp_growth"] = np.log(df["gdp_t5"] / df["gdp_t"])

    return df


def regress(df, feature_cols, target="log_gdp_growth", label=""):
    sub = df[feature_cols + [target]].dropna()
    if len(sub) < 30:
        print(f"[{label}] too few observations")
        return None

    X = StandardScaler().fit_transform(sub[feature_cols].values)
    y = sub[target].values
    m = Ridge(alpha=1.0).fit(X, y)
    y_hat = m.predict(X)
    r2 = r2_score(y, y_hat)
    rho, pval = spearmanr(y, y_hat)
    print(f"[{label}]     n={len(sub):4d}     R2={r2:.4f}     ρ={rho:.4f}  (p={pval:.2e})")
    return {"label": label, "n": len(sub), "r2": r2, "spearman": rho, "p": pval, "y": y, "y_hat": y_hat}


def plot_trajectories(cents_all_years, countries, years, gdp):
    plot_years = sorted(y for y in years if 2000 <= y <= 2020)
    if not plot_years:
        return
    stacked = np.vstack([cents_all_years[y] for y in plot_years])
    pca2 = PCA(n_components=2).fit(stacked)

    income_color = {}
    last_yr = 2015
    snap = gdp[gdp["year"] == last_yr].dropna().sort_values("gdp_pc")
    n = len(snap)
    low = set(snap.iloc[:n//3]["country"])
    high = set(snap.iloc[2*n//3:]["country"])
    for c in countries:
        if c in high: income_color[c] = ("#10b981", "High income")
        elif c in low: income_color[c] = ("#ef4444", "Low income")
        else: income_color[c] = ("#f59e0b", "Mid income")

    c_idx = {c: i for i, c in enumerate(countries)}
    np.random.seed(42)
    sample = np.random.choice(countries, min(60, len(countries)), replace=False)

    fig, ax = plt.subplots(figsize=(11, 8))
    drawn_labels = set()
    for country in sample:
        if country not in c_idx:
            continue
        ci = c_idx[country]
        pts = np.array([pca2.transform(cents_all_years[y][ci:ci+1])[0] for y in plot_years])
        col, lbl = income_color.get(country, ("#9ca3af", "Unknown"))
        kw = {"label": lbl} if lbl not in drawn_labels else {}
        drawn_labels.add(lbl)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=col, alpha=0.55, linewidth=1.2, markersize=3, **kw)
        ax.annotate(str(country), pts[-1], fontsize=4.5, alpha=0.6)

    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Country Trajectories in Product Embedding Space (2D projection)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "trajectories_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_magnitude_vs_growth(df):
    sub = df[["emb_magnitude", "log_gdp_growth", "t_start"]].dropna()
    if len(sub) < 10:
        return
    rho, p = spearmanr(sub["emb_magnitude"], sub["log_gdp_growth"])
    fig, ax = plt.subplots(figsize=(7, 5))
    for t, grp in sub.groupby("t_start"):
        ax.scatter(grp["emb_magnitude"], grp["log_gdp_growth"], alpha=0.4, s=15, label=str(t))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Embedding trajectory magnitude  ‖Δcentroid‖")
    ax.set_ylabel("Log GDP per-capita growth (5-year window)")
    ax.set_title(f"Basket displacement vs GDP growth (ρ = {rho:.3f}, p = {p:.3f})")
    ax.legend(title="Window start", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "magnitude_vs_growth.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_regression_scatter(results):
    results = [r for r in results if r is not None]
    if not results:
        return

    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]
    fig.suptitle("Predicted VS Actual 5-year Log GDP Growth", fontweight="bold")
    for ax, res in zip(axes, results):
        ax.scatter(res["y"], res["y_hat"], alpha=0.25, s=12, color="#3b82f6")
        lo = min(res["y"].min(), res["y_hat"].min()) - 0.05
        hi = max(res["y"].max(), res["y_hat"].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("Actual")
        ax.set_title(f"{res['label']}\nR2={res['r2']:.3f}  ρ={res['spearman']:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "regression_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    panel, products, emb, countries, years = load_core()
    gdp = load_wb_gdp(countries)

    df = build_dataset(panel, products, emb, countries, years, gdp)
    df.to_csv(os.path.join(RESULTS_DIR, "trajectories_dataset.csv"), index=False)
    print(f"Dataset: {len(df)} rows")

    # Regressions
    results = []
    if "log_gdp_growth" in df.columns:
        scalar_cols = ["emb_magnitude", "emb_cos_complex", "emb_centroid_norm"]
        r_scalar = regress(df, scalar_cols, label="Embedding scalars")
        full_cols = scalar_cols + [f"c{d}" for d in range(16)] + [f"d{d}" for d in range(16)]
        r_full = regress(df, full_cols, label="Embedding full (centroids + delta)")

        # Income controlled: regressing residuals
        # Does embedding trajectory residual growth beyond what initial income already predicts)
        sub_ctrl = df[full_cols + ["log_gdp_t", "log_gdp_growth"]].dropna()
        if len(sub_ctrl) > 30:
            Xbase = StandardScaler().fit_transform(sub_ctrl[["log_gdp_t"]].values)
            yg = sub_ctrl["log_gdp_growth"].values
            base_model = Ridge(alpha=1.0).fit(Xbase, yg)
            resid = yg - base_model.predict(Xbase)
            r2_base = r2_score(yg, base_model.predict(Xbase))
            rho_base, _ = spearmanr(yg, base_model.predict(Xbase))
            print(f"\n[Income baseline (log GDP_t -> growth)]    n={len(sub_ctrl):4d}    "
                  f"R2={r2_base:.4f}    ρ={rho_base:.4f}")

            Xemb = StandardScaler().fit_transform(sub_ctrl[full_cols].values)
            resid_model = Ridge(alpha=1.0).fit(Xemb, resid)
            resid_hat = resid_model.predict(Xemb)
            r2_resid = r2_score(resid, resid_hat)
            rho_resid, p_resid = spearmanr(resid, resid_hat)
            print(f"[Embedding on income-residualized growth]    n={len(sub_ctrl):4d}    "
                  f"R2={r2_resid:.4f}    ρ={rho_resid:.4f}  (p={p_resid:.2e})")

            r_ctrl = {"label": "Embedding (income-controlled)",
                      "n": len(sub_ctrl),
                      "r2": r2_resid, "spearman": rho_resid, "p": p_resid,
                      "y": resid, "y_hat": resid_hat}

        else:
            r_ctrl = None

        results = [r_scalar, r_full, r_ctrl]

        summary = pd.DataFrame([
            {k: v for k, v in r.items() if k not in ("y", "y_hat")}
            for r in results if r is not None
        ])
        print(summary[["label", "n", "r2", "spearman", "p"]].to_string(index=False))
        summary.to_csv(os.path.join(RESULTS_DIR, "trajectories_regression.csv"), index=False)


    needed = set(years)
    cents_all = {}
    for yr in needed:
        rca = make_rca_matrix(panel, yr, countries, products)
        cents_all[yr] = basket_centroids(rca, emb)

    plot_trajectories(cents_all, countries, years, gdp)
    if "log_gdp_growth" in df.columns:
        plot_magnitude_vs_growth(df)
        plot_regression_scatter(results)

    print("\nTop basket movers 2000->2005 (by magnitude)")
    cols_show = ["country","emb_magnitude","emb_cos_complex"]
    if "log_gdp_growth" in df.columns:
        cols_show.append("log_gdp_growth")
    top = (df[df["t_start"] == 2000]
           .nlargest(10, "emb_magnitude")[cols_show])
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
