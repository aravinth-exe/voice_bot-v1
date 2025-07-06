# app/assistant_chain.py

from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma(persist_directory="chroma_store", embedding_function=embedding)
llm = OllamaLLM(model="llama3")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

def query_agent(query: str) -> str:
    try:
        print(f"[RAG] Query: {query}")
        response = qa_chain.invoke({"query": query})
        answer = response.get("result") or response.get("answer") or ""
        if not answer.strip() or "not sure" in answer.lower():
            print("[RAG] No answer from vector DB. Using LLM fallback...")
            answer = llm.invoke(query)
        return answer
    except Exception as e:
        print(f"[ERROR] {e}")
        return "I'm sorry, something went wrong."
