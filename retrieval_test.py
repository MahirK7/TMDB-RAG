# retrieval_test.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = "artifacts/tmdb_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    query = "Top-rated Spanish dramas still in production with more than 3 seasons"
    docs = vs.similarity_search(query, k=5)
    print("🔎 Query:", query)
    for i, d in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(d.page_content[:600], "...")
