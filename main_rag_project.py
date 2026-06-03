from process_document import processdocument
from create_vector_db import createvectordb
from load_existing_db import loadexistingdb
from generate_answer import generateanswer
import os

if __name__ == "__main__":

    key_file= os.path.join("ragvenv", "api_key.txt")
    try:
        with open(key_file, "r") as f:
            api_key = f.read().strip()
            os.environ["GOOGLE_API_KEY"] = api_key
            print("API key loaded successfully!\n")
    except FileNotFoundError:
        print("API key file not found.")

    # Test our function
    chunks = processdocument("sample_text.txt")

    #Run to create the vector database
    db = createvectordb(chunks)

    #Run to load the existing vector database
    db = loadexistingdb()

    #Run to generate an answer
    question= input("Enter your question: ")
    generateanswer(db, question)