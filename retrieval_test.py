# retrieval_test.py -> Script to test FAISS retrieval without involving an LLM

# Import HuggingFace embeddings wrapper (needed to query FAISS index)
from langchain_huggingface import HuggingFaceEmbeddings

# Import FAISS vectorstore integration from LangChain community module
from langchain_community.vectorstores import FAISS


# ---------------------------
# Configurations
# ---------------------------

INDEX_DIR = "artifacts/tmdb_index"   # Path where your FAISS index is stored
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model (must match what you used for indexing)


# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    # Load the same embedding model that was used to build the index
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Load the FAISS index from disk
    # 'allow_dangerous_deserialization=True' is required because FAISS stores pickled metadata
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # Define a natural language query for retrieval
    query = "Top-rated Spanish dramas still in production with more than 3 seasons"
    
    # Run similarity search on the vectorstore
    # k=5 → retrieve top 5 most relevant chunks
    docs = vs.similarity_search(query, k=5)

    # Print query for clarity
    print("🔎 Query:", query)
    
    # Print the retrieved documents (first 600 characters only to keep output readable)
    for i, d in enumerate(docs, 1):
        print(f"\n--- Result {i} ---")
        print(d.page_content[:600], "...")

