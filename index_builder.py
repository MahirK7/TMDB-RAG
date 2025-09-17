# index_builder.py
import os
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

SOURCE_PATH = "data/tmdb_labeled.parquet"
INDEX_DIR = "artifacts/tmdb_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def row_to_block(row: pd.Series) -> str:
    cols = [
        'id','name','number_of_seasons','number_of_episodes','original_language',
        'vote_count','vote_average','overview','adult','backdrop_path','first_air_date',
        'last_air_date','homepage','in_production','original_name','popularity',
        'poster_path','type','status','tagline','genres','created_by','languages',
        'networks','origin_country','spoken_languages','production_companies',
        'production_countries','episode_run_time','is_popular','is_long_running'
    ]
    lines = []
    for c in cols:
        if c in row:
            lines.append(f"{c}: {row[c]}")
    return "\n".join(lines)

def build_index(force_rebuild=False):
    os.makedirs("artifacts", exist_ok=True)

    if os.path.exists(INDEX_DIR) and not force_rebuild:
        print(f"✅ Using existing FAISS index at {INDEX_DIR}")
        return

    print("⚙️ Building new FAISS index...")
    df = pd.read_parquet(SOURCE_PATH)

    # Progress bar for document creation
    docs_text = [row_to_block(r) for _, r in tqdm(df.iterrows(), total=len(df), desc="📄 Converting rows")]

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents(docs_text)

    print("🔢 Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Progress wrapper for embeddings
    vs = FAISS.from_documents(tqdm(docs, desc="⚡ Embedding docs"), embeddings)

    vs.save_local(INDEX_DIR)
    print(f"✅ FAISS index saved to {INDEX_DIR} (chunks: {len(docs)})")

if __name__ == "__main__":
    import sys
    force = "--rebuild" in sys.argv
    build_index(force_rebuild=force)
