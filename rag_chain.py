# Import FAISS vectorstore for efficient similarity search
from langchain_community.vectorstores import FAISS

# Import HuggingFace embeddings wrapper (to embed text chunks)
from langchain_huggingface import HuggingFaceEmbeddings

# Import RetrievalQA chain (LangChain utility that connects retriever + LLM into a Q&A system)
from langchain.chains import RetrievalQA

# Import supported LLM backends
from langchain_ollama import OllamaLLM       # Local models served by Ollama
from langchain_openai import ChatOpenAI      # OpenAI models (GPT family)
from langchain_anthropic import ChatAnthropic # Anthropic models (Claude family)


# ---------------------------
# Configurations
# ---------------------------

INDEX_DIR = "artifacts/tmdb_index"   # Path where FAISS index is stored
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model


# ---------------------------
# Function: build_qa_chain
# ---------------------------
def build_qa_chain(model_name: str):
    # Load HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    
    # Load FAISS vectorstore from disk (with embeddings for compatibility check)
    # allow_dangerous_deserialization=True is needed when loading pickled FAISS index
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    
    # Create retriever from FAISS (returns top-k similar chunks when queried)
    retriever = vs.as_retriever(search_kwargs={"k": 5})  # k=5 → return top 5 matches


    # 🔑 Choose which LLM backend to use based on the string prefix
    if model_name.startswith("ollama:"):
        # Extract actual model name (after the prefix) → e.g. "ollama: llama3:3b" → "llama3:3b"
        model_name = model_name.split(":", 1)[1]
        llm = OllamaLLM(model=model_name)  # Initialize Ollama LLM

    elif model_name.startswith("openai:"):
        # Extract model name → e.g. "openai: gpt-4o-mini" → "gpt-4o-mini"
        model_name = model_name.split(":", 1)[1]
        # Initialize OpenAI chat model (temperature=0 → deterministic responses)
        llm = ChatOpenAI(model=model_name, temperature=0)

    elif model_name.startswith("anthropic:"):
        # Extract model name → e.g. "anthropic: claude-3-sonnet" → "claude-3-sonnet"
        model_name = model_name.split(":", 1)[1]
        # Initialize Anthropic chat model (temperature=0)
        llm = ChatAnthropic(model=model_name, temperature=0)

    else:
        # If prefix not recognized → raise error
        raise ValueError(f"Unknown model choice: {model_name}")


    # Build RetrievalQA chain:
    #  - Takes in a user query
    #  - Uses retriever to fetch top docs
    #  - Feeds them into LLM
    #  - Returns answer + sources
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True  # include the original chunks in the output
    )

    return qa  # Return the full QA pipeline
