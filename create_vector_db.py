from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def create_vector_db(chunks):
    print("--- Step 3: Creating Embeddings and Vector Database ---")
    
    # 1. Initialize the embedding model
    print("Loading embedding model (this might take a few seconds the first time to download)...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Create and store the vector database locally
    persist_directory = "./chroma_db"
    print(f"Creating Chroma DB at {persist_directory}...")
    
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_model, 
        persist_directory=persist_directory
    )
    print("Database created and saved successfully!\n")
    
    return db