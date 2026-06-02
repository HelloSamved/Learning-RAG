from process_document import processdocument
from create_vector_db import createvectordb

if __name__ == "__main__":
    # Test our function
    chunks = processdocument("sample_text.txt")

    #Run Step-3
    db = createvectordb(chunks)