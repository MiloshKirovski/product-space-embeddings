# Transform raw BACI trade data into RCA matrices (one per year)
import os
import glob
import pandas as pd

RAW_DIR = "data/raw/BACI_HS92_V202601"
OUT_DIR = "data/processed"

def compute_rca(df):
    # RCA(c, p) = (X_cp / X_c) / (X_wp / X_w)
    # where   X_pc - total exports of product p for country c
    # X_c - total exports of country c
    # X_wp - world exports of product p
    # X_w - total world exports

    x_c = df.groupby("country")["exports"].sum().rename("x_c")
    x_wp = df.groupby("product")["exports"].sum().rename("x_wp")
    x_w = df["exports"].sum()

    df = df.join(x_c, on="country")
    df = df.join(x_wp, on="product")
    df["rca"] = (df["exports"] / df["x_c"]) / (df["x_wp"] / x_w)
    return df.drop(columns=["x_c", "x_wp"])


def load_baci_year(filepath):
    # Loads one BACI CSV file, aggregates to country-product level
    df = pd.read_csv(filepath, dtype={"k": str})
    df = df.rename(columns={
        "i": "country",
        "k": "product",
        "v": "exports",
    })
    df = df.groupby(["country", "product"])["exports"].sum().reset_index()
    df["exports"] = pd.to_numeric(df["exports"], errors="coerce").fillna(0)
    return df


def load_synthetic(filepath):
    # Load single synthetic CSV (all years) and return list (year, df)
    df = pd.read_csv(filepath, dtype={"k": str})
    df = df.rename(columns={
        "i": "country",
        "k": "product",
        "v": "exports",
    })
    df["exports"] = pd.to_numeric(df["exports"], errors="coerce").fillna(0)
    result = []
    for yr, grp in df.groupby("year"):
        sub = grp.groupby(["country", "product"])["exports"].sum().reset_index()
        result.append((int(yr), sub))
    return result


def main():
    baci_files = sorted(glob.glob(os.path.join(RAW_DIR, "BACI_HS92_Y*.csv")))
    all_records = []

    year_data_iter = []
    for fpath in baci_files:
        fname = os.path.basename(fpath)
        year = int(fname.split("Y")[1].split("_")[0])
        year_data_iter.append((year, fpath))


    for year, fpath in year_data_iter:
        print(f"Processing {year} - {fpath}")
        df = load_baci_year(fpath)
        df = compute_rca(df)
        df["year"] = year
        df["rca_binary"] = (df["rca"] >= 1.0).astype(int)
        wide = df.pivot_table(index="country", columns="product", values="rca_binary", fill_value=0).astype(int)
        wide.to_parquet(os.path.join(OUT_DIR, f"rca_{year}.parquet"))
        all_records.append(df[["year", "country", "product", "exports", "rca", "rca_binary"]])


    panel = pd.concat(all_records, ignore_index=True)
    panel.to_parquet(os.path.join(OUT_DIR, "rca_panel.parquet"))

    print(f"Panel shape: {panel.shape}")
    print(f"Years: {sorted(panel['year'].unique())}")
    print(f"Countries: {panel["country"].nunique()}")
    print(f"Products: {panel['product'].nunique()}")
    print(f"Avg RCA>=1 per country-year: "
          f"{panel.groupby(['year', 'country'])['rca_binary'].sum().mean():.1f}")


if __name__ == "__main__":
    main()