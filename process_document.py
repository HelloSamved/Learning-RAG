from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_document(file_path):
    print("--- Step 2: Ingesting and Chunking Data ---")
    
    # 1. Load the text file
    loader = TextLoader(file_path)
    document = loader.load()
    print(f"Successfully loaded document: {file_path}")
    
    # 2. Configure the text splitter
    # Chunk size controls the length of each piece; overlap preserves context between boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    
    # 3. Split the document into chunks
    chunks = text_splitter.split_documents(document)
    print(f"Split the document into {len(chunks)} individual chunks.\n")
    
    # Let's inspect the first two chunks to see what they look like
    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i+1} ---")
        print(chunk.page_content)
        print("-" * 20)
        
    return chunks