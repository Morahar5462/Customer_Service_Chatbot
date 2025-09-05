from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env

# ✅ Use Gemini (replace with gemini-1.5-pro or gemini-1.5-flash if needed)
llm = GoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
)

# ✅ Use local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path=r"C:\Users\morah\OneDrive\Desktop\Customer_Service _Chatbot\dataset\dataset.csv", source_column="prompt")
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    llm = GoogleGenerativeAI(model="gemini-1.5-flash-latest")

    # Create a better prompt
    prompt_template = """
    You are a helpful customer support assistant.
    Use the following context (if relevant) to answer the user’s question.
    If the context is empty, just answer naturally using your own knowledge.
    
    Context: {context}
    Question: {question}
    
    Give a polite, clear, and enhanced answer — not just a copy.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Retrieval + LLM Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return chain


if __name__ == "__main__":
    
    create_vector_db()
    chain = get_qa_chain()
    result = chain.invoke({"query": "hello?"})
    print(result)
