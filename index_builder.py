# index_builder.py -> This script builds a FAISS vector index for the TMDB dataset

# Import core libraries
import os                          # for file/directory handling
import pandas as pd                # for loading and processing tabular data
from tqdm import tqdm              # for progress bars in loops

# Import LangChain components for text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter  # splits text into chunks
from langchain_huggingface import HuggingFaceEmbeddings             # HuggingFace model wrapper for embeddings
from langchain_community.vectorstores import FAISS                  # FAISS vector database integration


# Define constants for paths and model to use
SOURCE_PATH = "data/tmdb_labeled.parquet"          # location of the labeled TMDB dataset
INDEX_DIR = "artifacts/tmdb_index"                 # where the FAISS index will be saved
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # pretrained embedding model to use


# Convert a single DataFrame row into a text block
def row_to_block(row: pd.Series) -> str:
    # Select which columns we want to include in the text representation
    cols = [
        'id','name','number_of_seasons','number_of_episodes','original_language',
        'vote_count','vote_average','overview','adult','backdrop_path','first_air_date',
        'last_air_date','homepage','in_production','original_name','popularity',
        'poster_path','type','status','tagline','genres','created_by','languages',
        'networks','origin_country','spoken_languages','production_companies',
        'production_countries','episode_run_time','is_popular','is_long_running'
    ]
    
    # Create a list of lines, one for each column's key:value
    lines = []
    for c in cols:
        if c in row:  # ensure the column exists in the row
            lines.append(f"{c}: {row[c]}")
    
    # Join all lines into one text block separated by newlines
    return "\n".join(lines)


# Function to build (or reuse) the FAISS index
def build_index(force_rebuild=False):
    # Ensure "artifacts" directory exists
    os.makedirs("artifacts", exist_ok=True)

    # If the index already exists and we're not forcing rebuild → just use it
    if os.path.exists(INDEX_DIR) and not force_rebuild:
        print(f"✅ Using existing FAISS index at {INDEX_DIR}")
        return

    # Otherwise, rebuild from scratch
    print("⚙️ Building new FAISS index...")

    # Load the labeled TMDB dataset (saved as parquet earlier)
    df = pd.read_parquet(SOURCE_PATH)

    # Convert each row into a text block (with a progress bar)
    docs_text = [row_to_block(r) for _, r in tqdm(df.iterrows(), total=len(df), desc="📄 Converting rows")]

    # Split text blocks into smaller chunks (800 characters each, with 100 char overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(docs_text)

    # Generate embeddings for all document chunks
    print("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Build FAISS index from documents (with a progress bar on embedding)
    vs = FAISS.from_documents(tqdm(docs, desc="⚡ Embedding docs"), embeddings)

    # Save the FAISS index locally
    vs.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to {INDEX_DIR} (chunks: {len(docs)})")


# Run the script directly (not when imported)
if __name__ == "__main__":
    import sys
    # Check if user passed '--rebuild' flag in command line to force rebuild
    force = "--rebuild" in sys.argv
    # Build the index accordingly
    build_index(force_rebuild=force)
