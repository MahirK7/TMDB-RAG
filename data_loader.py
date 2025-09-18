# data_loader.py  -> This is the filename of the script

# Import the 'os' module for interacting with the operating system (like working with files/directories)
import os

# Import the pandas library for handling and analyzing tabular data (CSV, Parquet, etc.)
import pandas as pd

# Import kagglehub, a library to download datasets directly from Kaggle
import kagglehub


# Define a function to load the TMDB dataset
def load_tmdb_data(nrows=None):
    # Download the dataset from Kaggle (returns the local folder path where it's saved)
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
    
    # Re-import 'os' with an alias '_os' (local scope, just to avoid shadowing issues)
    import os as _os
    
    # Find all files in the dataset folder that end with '.csv' (case-insensitive)
    csv_files = [f for f in _os.listdir(path) if f.lower().endswith(".csv")]
    
    # If no CSV file is found, raise an error so the user knows something went wrong
    if not csv_files:
        raise FileNotFoundError("No CSV found in the dataset folder.")
    
    # Construct the full file path to the first CSV file found
    csv_path = _os.path.join(path, csv_files[0])
    
    # Print which CSV file is being loaded
    print(f"Loading: {csv_path}")
    
    # Read the CSV file into a pandas DataFrame
    # If 'nrows' is provided, only read that many rows (useful for faster testing)
    df = pd.read_csv(csv_path, nrows=nrows)
    
    # Return the DataFrame containing the dataset
    return df


# Define a function to add label columns to the dataset
def label_data(df: pd.DataFrame):
    # Make a copy of the DataFrame to avoid modifying the original input
    df = df.copy()
    
    # Create a new binary column 'is_popular'
    # 1 if vote_average >= 7.5, else 0
    df["is_popular"] = (df["vote_average"] >= 7.5).astype(int)
    
    # Create a new binary column 'is_long_running'
    # 1 if number_of_seasons >= 5, else 0
    # fillna(0) replaces missing values with 0 before comparison
    df["is_long_running"] = (df["number_of_seasons"].fillna(0) >= 5).astype(int)
    
    # Return the updated DataFrame with the new labels
    return df


# This block runs only if the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Create a folder named 'data' if it doesn’t already exist
    os.makedirs("data", exist_ok=True)
    
    # Load the TMDB dataset
    # (Tip: pass nrows=20000 if you want a quicker test instead of loading the whole dataset)
    df = load_tmdb_data()
    
    # Add the label columns ('is_popular' and 'is_long_running') to the dataset
    df = label_data(df)
    
    # Define the output path for saving the labeled dataset in Parquet format
    out_path = "data/tmdb_labeled.parquet"
    
    # Save the DataFrame as a Parquet file (efficient binary format for big data)
    # index=False means don’t save the row index column
    df.to_parquet(out_path, index=False)
    
    # Print confirmation that the dataset has been saved, including the number of rows
    print(f"✅ Saved labeled dataset to {out_path} with {len(df)} rows")
