import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# ------------------------------------
# Load Dataset
# ------------------------------------
def load_data():
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("⚠️ No CSV file found in dataset folder!")
    csv_path = os.path.join(path, csv_files[0])
    print(f"✅ Using dataset: {csv_path}")
    return pd.read_csv(csv_path)

# ------------------------------------
# Analysis Functions
# ------------------------------------
def plot_release_trend(df):
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x="first_air_date", order=df["first_air_date"].value_counts().index[:20])
    plt.xticks(rotation=90)
    plt.title("📈 Number of TV Shows by Release Year (Top 20 Years)")
    plt.tight_layout()
    plt.show()

def plot_genre_distribution(df):
    plt.figure(figsize=(10,6))
    df["genres"].dropna().str.split(",").explode().value_counts()[:15].plot(kind="bar", color="skyblue")
    plt.title("🎭 Top 15 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df["vote_average"], bins=20, kde=True, color="purple")
    plt.title("⭐ Distribution of IMDb-like Ratings (vote_average)")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_country_production(df):
    plt.figure(figsize=(12,6))
    df["origin_country"].dropna().str.split(",").explode().value_counts()[:10].plot(kind="bar", color="orange")
    plt.title("🌍 Top 10 Producing Countries")
    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# ------------------------------------
# Main
# ------------------------------------
if __name__ == "__main__":
    df = load_data()

    # Ensure key columns exist
    for col in ["first_air_date", "genres", "vote_average", "origin_country"]:
        if col not in df.columns:
            raise ValueError(f"❌ Column {col} not found in dataset!")

    # Run analyses
    plot_release_trend(df)
    plot_genre_distribution(df)
    plot_rating_distribution(df)
    plot_country_production(df)
