from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

INDEX_DIR = "artifacts/tmdb_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_qa_chain(model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    # 🔑 Choose model based on dropdown
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[1]
        llm = OllamaLLM(model=model_name)

    elif model_name.startswith("openai:"):
        model_name = model_name.split(":", 1)[1]
        llm = ChatOpenAI(model=model_name, temperature=0)

    elif model_name.startswith("anthropic:"):
        model_name = model_name.split(":", 1)[1]
        llm = ChatAnthropic(model=model_name, temperature=0)

    else:
        raise ValueError(f"Unknown model choice: {model_name}")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa
