from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


def generateanswer(db, question):
    print("--- Step 4: Retrieving and Generating Answer ---")
    
    # 1. Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7
    )
    
    # 2. Set up the Retriever to fetch the top 3 most relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 3. Create the Prompt Template
    system_prompt = """
    You are a helpful assistant. Answer the user's question using ONLY the provided context.
    If you don't know the answer based on the context, just say that you don't know.
    
    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # 4. Build the RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # 5. Run the query
    print(f"Question: {question}\n")
    response = rag_chain.invoke({"input": question})
    
    print("Answer:")
    print(response["answer"])