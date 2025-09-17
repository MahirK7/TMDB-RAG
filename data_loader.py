# data_loader.py
import os
import pandas as pd
import kagglehub

def load_tmdb_data(nrows=None):
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
    # auto-detect the CSV filename
    import os as _os
    csv_files = [f for f in _os.listdir(path) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in the dataset folder.")
    csv_path = _os.path.join(path, csv_files[0])
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, nrows=nrows)
    return df

def label_data(df: pd.DataFrame):
    df = df.copy()
    df["is_popular"] = (df["vote_average"] >= 7.5).astype(int)
    df["is_long_running"] = (df["number_of_seasons"].fillna(0) >= 5).astype(int)
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = load_tmdb_data()           # tip: pass nrows=20000 to test faster
    df = label_data(df)
    out_path = "data/tmdb_labeled.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved labeled dataset to {out_path} with {len(df)} rows")
