
# import os
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import fitz  # PyMuPDF

# # Load environment variables from .env
# load_dotenv()

# # Load Gemini API Key from .env
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# # Load Hugging Face Embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Function to read and process PDF
# def process_pdf(pdf_path):
#     text = ""
#     doc = fitz.open(pdf_path)
#     for page in doc:
#         text += page.get_text("text") + "\n"

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = text_splitter.split_text(text)

#     vector_db = FAISS.from_texts(chunks, embedding=embeddings)
#     return vector_db.as_retriever()

# # Function to query Gemini API
# def query_gemini(question, context):
#     prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
#     return response.text.strip() if response else "I'm not sure."

# # Function to answer questions from PDF
# def answer_question(pdf_path, question):
#     retriever = process_pdf(pdf_path)
#     docs = retriever.get_relevant_documents(question)

#     context = "\n".join([doc.page_content for doc in docs])
#     answer = query_gemini(question, context)
#     return answer

# # Main Execution
# if __name__ == "__main__":
#     pdf_path = "ml.pdf"  # Replace with your actual PDF
#     question = input("Ask a question: ")
#     answer = answer_question(pdf_path, question)
#     print("\nAnswer:", answer)


import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF

# Load environment variables from .env
load_dotenv()

# Load Gemini API Key from .env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Function to read and process PDF
def process_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_db.as_retriever()

# Function to query Gemini API
def query_gemini(question, context):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text.strip() if response else "I'm not sure."

# Function to answer questions from PDF
def answer_question(pdf_path, question):
    retriever = process_pdf(pdf_path)
    docs = retriever.get_relevant_documents(question)

    context = "\n".join([doc.page_content for doc in docs])
    answer = query_gemini(question, context)
    return answer

# Main Execution
if __name__ == "__main__":
    pdf_path = "ml.pdf"  # Replace with your actual PDF
    retriever = process_pdf(pdf_path)  # Load once to avoid reloading for every question

    print("\n📄 PDF loaded! You can now ask questions.")
    print("💡 Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            print("👋 Exiting... Have a great day!")
            break

        answer = answer_question(pdf_path, question)
        print("\n📝 Answer:", answer)
