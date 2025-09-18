# Import libraries
import pandas as pd              # for dataset handling
import seaborn as sns            # for plotting
import matplotlib.pyplot as plt  # for visualizations
import kagglehub                 # to download dataset from Kaggle
import os                        # for file operations


# ------------------------------------
# Load Dataset
# ------------------------------------
def load_data():
    # Download dataset from Kaggle (returns local folder path)
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")

    # Find CSV files inside dataset folder
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

    # If no CSV found → raise error
    if not csv_files:
        raise FileNotFoundError("⚠️ No CSV file found in dataset folder!")

    # Take the first CSV file found
    csv_path = os.path.join(path, csv_files[0])

    # Print confirmation of which file is used
    print(f"✅ Using dataset: {csv_path}")

    # Load the CSV file into a pandas DataFrame
    return pd.read_csv(csv_path)


# ------------------------------------
# Analysis Functions
# ------------------------------------

# 1. Release year trend (Top 20 years by number of shows)
def plot_release_trend(df):
    plt.figure(figsize=(12,6))
    sns.countplot(
        data=df,
        x="first_air_date",
        order=df["first_air_date"].value_counts().index[:20]  # top 20 years
    )
    plt.xticks(rotation=90)
    plt.title("📈 Number of TV Shows by Release Year (Top 20 Years)")
    plt.tight_layout()
    plt.show()

# 2. Genre distribution (Top 15 genres)
def plot_genre_distribution(df):
    plt.figure(figsize=(10,6))
    (
        df["genres"]
        .dropna()
        .str.split(",")
        .explode()
        .value_counts()[:15]     # top 15 genres
        .plot(kind="bar", color="skyblue")
    )
    plt.title("🎭 Top 15 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 3. Rating distribution (histogram of vote_average)
def plot_rating_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df["vote_average"], bins=20, kde=True, color="purple")
    plt.title("⭐ Distribution of IMDb-like Ratings (vote_average)")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 4. Top producing countries (Top 10 origin_country values)
def plot_country_production(df):
    plt.figure(figsize=(12,6))
    (
        df["origin_country"]
        .dropna()
        .str.split(",")
        .explode()
        .value_counts()[:10]     # top 10 countries
        .plot(kind="bar", color="orange")
    )
    plt.title("🌍 Top 10 Producing Countries")
    plt.xlabel("Country")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# ------------------------------------
# Main
# ------------------------------------
if __name__ == "__main__":
    # Load dataset into DataFrame
    df = load_data()

    # Safety check: ensure required columns exist
    for col in ["first_air_date", "genres", "vote_average", "origin_country"]:
        if col not in df.columns:
            raise ValueError(f"❌ Column {col} not found in dataset!")

    # Run all analyses one after another
    plot_release_trend(df)
    plot_genre_distribution(df)
    plot_rating_distribution(df)
    plot_country_production(df)
