# rag_ingest.py

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def ingest_docs(folder):
    chroma_path = "chroma_store"
    if os.path.exists(chroma_path):
        import shutil
        shutil.rmtree(chroma_path)

    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not file.endswith((".pdf", ".docx", ".txt")):
            continue
        print(f"📥 Loading: {file}")
        loader = UnstructuredFileLoader(path)
        docs.extend(loader.load())

    print(f"✅ Documents loaded: {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Chunks created: {len(chunks)}")

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=chroma_path)
    print("✅ Chroma DB persisted at", chroma_path)
