from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import shutil


loader = CSVLoader("yfinance_method_details..csv")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50, separators=["Method:"])
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

# Define the persist directory
persist_directory = None#"./chroma_db"

"""
# Delete previous Chroma DB if it exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Removed existing Chroma DB at: {persist_directory}")
"""

print("creating new chroma vector store")
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory

)
vectorstore.persist()
print("vector_store_len",len(vectorstore))