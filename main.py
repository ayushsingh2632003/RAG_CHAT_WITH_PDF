import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
#from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks



# def get_vector_store(marathi_text_chunks):
#     """
#     दिलेल्या मराठी मजकुरासाठी वेक्टर स्टोअर तयार करते.

#     Args:
#         marathi_text_chunks: मराठी मजकुराचे तुकडे असलेली सूची.

#     Returns:
#         FAISS वेक्टर स्टोअर.
#     """
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     vector_store = FAISS.from_texts(marathi_text_chunks, embedding=embeddings)
#     vector_store.save_local("marathi_faiss_index") # marathi_faiss_index नावाची directory तयार करते.
#     #return vector_store



def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings, allow_dangerous_deserialization=True)
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # vector_store.save_local("faiss_index")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")




def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not
    in the provided context just say, "Answer is not available in the context", dont provide the wrong answers \n\n

    Contxet: \n {context}?\n
    Question: \n{question}\n

    Answer:

"""

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff")
    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #new_db = FAISS.load_local("marathi_faiss_index", embeddings, allow_dangerous_deserialization=True)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat With Multiple PDF")
    st.markdown("## **💬 Chat With Your PDF Files**")
    st.image("img.jpg", width=500) 

    # User question input and button
    user_question = st.text_area("Ask a question from pdf files", height=100)
    if st.button("Ask Question"):
        if user_question.strip() != "":
            user_input(user_question)
        else:
            st.warning("Please enter a question before clicking Ask Question.")

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files and Click on the Submit button and process files",
            type=["pdf"],
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

