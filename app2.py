import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import time

# Load environment variables from .env file
load_dotenv()

# Load API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit title
st.title("Medical Chatbot - Symptom Analysis Assistant")

# Load the medical model from Groq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Chat prompt to encourage specific responses based on symptoms provided
prompt = ChatPromptTemplate.from_template(
    """
    You are a medical physician doctor's assistant providing information from a trusted medical reference.
    Focus on analyzing the symptoms the user shares and suggesting possible conditions.
    Be specific about potential causes, conditions, or common diagnoses associated with the symptoms.
    <context>
    {context}
    </context>
    User's Question: {input}
    """
)

# Function to load and embed the Gale Encyclopedia of Medicine
def vector_embedding():
    if "vectors" not in st.session_state:
        # Load the Gale Encyclopedia of Medicine PDF
        pdf_path = "/Users/sans22/Documents/Llama2-Medical-Chatbot-main/data/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(pdf_path)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

        try:
            # Create a vector database for the loaded documents
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.success("Medical Reference Vector Store is ready.")
        except Exception as e:
            st.error(f"Error creating vector store: {e}")

# Call the vector embedding function to load and process the document
vector_embedding()

# Initialize chat history if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to display chat messages
def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Display existing chat messages
display_chat_messages()

# User input for conversational chat
user_input = st.chat_input("Describe your symptoms or ask about a medical condition:")

# If there's user input and the vector database is ready, retrieve answers
if user_input and "vectors" in st.session_state:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Retrieve and summarize relevant information based on symptoms
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Analyzing symptoms..."):
        response = retrieval_chain.invoke({'input': user_input})

    # Display bot response with improved specificity
    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    display_chat_messages()

    # Optionally, show relevant document chunks as additional reference
    with st.expander("Reference Information"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
