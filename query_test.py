from rag_chain import build_qa_chain

qa = build_qa_chain()
query = "List popular Indian TV shows released after 2020"
result = qa.invoke(query)

print("\n📝 Answer:", result["result"])
print("\n📂 Sources:")
for doc in result["source_documents"]:
    print("-", doc.page_content[:200], "...")
