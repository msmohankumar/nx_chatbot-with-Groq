import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
VECTOR_STORES_PATH = "vector_stores"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AVAILABLE_LANGUAGES = ["python", "cpp", "vb", "docs"] # Add other languages like 'csharp' if you have data

# --- CORE FUNCTIONS ---

@st.cache_resource
def load_llm():
    """Loads the Language Model."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env file.")
    return ChatGroq(model_name="llama3-8b-8192", temperature=0.2)

# This function is now cached per language
@st.cache_resource
def load_retriever(language: str):
    """Loads the vector store for a specific language."""
    store_path = os.path.join(VECTOR_STORES_PATH, language)
    if not os.path.exists(store_path):
        raise FileNotFoundError(f"Vector store for '{language}' not found. Please run 'python ingest_data.py'.")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
    vector_store = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={'k': 5})

def create_rag_chain(_retriever, _llm, language: str):
    """Creates the RAG chain with a language-specific prompt."""
    lang_map = {
        "python": "Python", "cpp": "C++", "vb": "VB.NET", "csharp": "C#", "docs": "general documentation"
    }
    lang_name = lang_map.get(language, "the specified language")

    prompt_template = f"""
    You are an expert programmer specializing in Siemens NX CAD automation in {lang_name}.
    Your primary goal is to provide complete, correct, and runnable code solutions based ONLY on the provided context.

    **CRITICAL INSTRUCTIONS:**
    1.  Generate the answer strictly in {lang_name}.
    2.  If the user asks for a code example, it **MUST** be a complete, self-contained script or function for that language.
    3.  Follow the best practices for the language, including necessary imports, main function/entry points, and error handling.
    4.  If the context is from general documentation (docs), summarize it clearly.
    5.  If the context does not contain enough information, clearly state: "The provided documents do not contain enough information to answer this question." Do not invent code.

    **Context:**
    {{context}}

    **Question:**
    {{input}}

    **Complete {lang_name} Answer:**
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(_llm, prompt)
    retrieval_chain = create_retrieval_chain(_retriever, document_chain)
    return retrieval_chain

# --- STREAMLIT UI ---

st.set_page_config(page_title="NX CAD Customization Bot", layout="wide")
st.title("🤖 NX CAD Multi-Language Assistant")
st.caption("Powered by Local NX Data and Llama 3")

# Sidebar for language selection
st.sidebar.title("Configuration")
selected_language = st.sidebar.selectbox(
    "Select Programming Language",
    options=AVAILABLE_LANGUAGES,
    index=0 # Default to Python
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Ask me anything about NX Open in {selected_language.upper()}!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chatbot logic
try:
    llm = load_llm()
    retriever = load_retriever(selected_language)
    rag_chain = create_rag_chain(retriever, llm, selected_language)

    if prompt := st.chat_input(f"Ask about NX Open in {selected_language.upper()}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Searching {selected_language.upper()} knowledge base..."):
                response = rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, I encountered an error.")
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"**An error occurred:** {e}")
    st.info("Please ensure your '.env' file is correct and you have run 'python ingest_data.py' successfully.")

