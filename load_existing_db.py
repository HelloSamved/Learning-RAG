from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def loadexistingdb():
    print("--- Loading Existing Vector Database ---")
    # Must use the exact same embedding model used during creation
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )
    print("Database loaded successfully!\n")
    return db