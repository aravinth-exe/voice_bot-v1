import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.assistant_chain import qa_chain
from langchain_ollama import OllamaLLM

query = "How does delivery work?"
response = qa_chain.invoke({"query": query})
answer = response.get("answer")

# Fallback to LLM if no RAG result
if not answer:
    print("🟡 No answer from vector DB. Using LLM fallback...")
    llm = OllamaLLM(model="llama3")
    answer = llm.invoke(query)

print("✅ Final Answer:", answer)
