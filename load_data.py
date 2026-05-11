import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

data_path = "data"
documents = []

for file in os.listdir(data_path):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join(data_path, file))
        documents.extend(loader.load())

print(f"Loaded {len(documents)} documents")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(docs, embeddings)

db.save_local("db")

print("Vector DB created successfully!")