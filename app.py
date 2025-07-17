import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
import platform

# Langchain imports for document processing, embeddings, and LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS # Changed from Chroma to FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables from .env file
load_dotenv()

# Set up the Google API Key
# Ensure you have GOOGLE_API_KEY set in your environment variables or a .env file
# Example: GOOGLE_API_KEY="YOUR_API_KEY_HERE"
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY environment variable not found. Please set it.")
    st.stop()

# --- Fix for RuntimeError: There is no current event loop ---
# This helps with issues where libraries might implicitly try to manage asyncio event loops
# especially on Windows.
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Initialize Langchain Components ---

@st.cache_resource
def get_text_splitter():
    """Initializes and returns a text splitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

@st.cache_resource
def get_embeddings_model():
    """
    Initializes and returns the Google Generative AI Embeddings model.
    Includes asyncio loop handling to prevent RuntimeError in certain environments.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_llm():
    """
    Initializes and returns the Google Generative AI Chat model.
    Using 'gemini-1.5-flash' as it's generally more broadly available.
    """
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

def get_vector_store(text_chunks, embeddings_model):
    """
    Creates and returns a FAISS vector store from text chunks and an embeddings model.
    FAISS is used instead of Chroma to avoid sqlite3 version compatibility issues
    in deployment environments like Streamlit Cloud. The vector store is created
    in memory for this session.
    """
    return FAISS.from_documents(text_chunks, embeddings_model) # Changed from Chroma to FAISS

def get_conversation_chain(vector_store, llm):
    """
    Creates and returns a conversational retrieval chain.
    This chain takes a question, retrieves relevant documents, and generates a response
    while maintaining chat history.
    """
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- Streamlit UI ---

st.set_page_config(page_title="PDF Chatbot with Gemini & FAISS", layout="wide")
st.title("📄 PDF Chatbot powered by Gemini & FAISS") # Updated title
st.markdown("Upload a PDF document and start asking questions about its content!")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload your PDF")
    pdf_docs = st.file_uploader(
        "Upload your PDF files here and click 'Process'", accept_multiple_files=True, type="pdf"
    )
    if st.button("Process PDF"):
        if pdf_docs:
            with st.spinner("Processing PDF..."):
                raw_text = ""
                for pdf in pdf_docs:
                    # Save the uploaded PDF to a temporary file to allow PyPDFLoader to read it
                    with open(f"temp_{pdf.name}", "wb") as f:
                        f.write(pdf.getbuffer())
                    loader = PyPDFLoader(f"temp_{pdf.name}")
                    raw_text += " ".join([page.page_content for page in loader.load_and_split()])
                    os.remove(f"temp_{pdf.name}") # Clean up the temporary file

                # Get text chunks
                text_splitter = get_text_splitter()
                text_chunks = text_splitter.create_documents([raw_text])

                # Get embeddings model
                embeddings_model = get_embeddings_model()

                # Create vector store (now FAISS)
                vector_store = get_vector_store(text_chunks, embeddings_model)
                st.session_state.vector_store = vector_store # Store in session state

                # Initialize conversation chain
                llm = get_llm()
                st.session_state.conversation = get_conversation_chain(vector_store, llm)

                st.success("PDF processed successfully! You can now ask questions.")
        else:
            st.warning("Please upload at least one PDF file.")

# Main chat interface
if "conversation" not in st.session_state:
    st.info("Please upload and process a PDF to start chatting.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0: # User message
            with st.chat_message("user"):
                st.write(message.content)
        else: # AI message
            with st.chat_message("assistant"):
                st.write(message.content)

    # User input for questions
    user_question = st.chat_input("Ask a question about the PDF...")
    if user_question:
        if st.session_state.conversation:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history.append(response['chat_history'][-2]) # User message
                st.session_state.chat_history.append(response['chat_history'][-1]) # AI message
                # Re-run to display the new messages
                st.rerun()
        else:
            st.warning("Please process a PDF first to start the conversation.")
