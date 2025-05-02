from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import time
import  os

from langchain_pinecone import PineconeVectorStore



pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)

index_name = "langchain-test-index"  # change if desired
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

"""
# Delete previous Chroma DB if it exists
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print(f"Removed existing Chroma DB at: {persist_directory}")
"""



def initialise_pinecone_db():
    # Load documents
    loader = CSVLoader("yfinance_method_details..csv")  # fixed double-dot typo
    docs = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50, separators=["Method:"])
    split_docs = splitter.split_documents(docs)

    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # Add documents
    vector_store.add_documents(split_docs)

    return vector_store

# Call the function
vectorstore = initialise_pinecone_db()
