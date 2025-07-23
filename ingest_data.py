import os
import glob
import shutil
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# --- CONFIGURATION ---
SOURCE_DOCS_PATH = "source_documents"
VECTOR_STORES_PATH = "vector_stores"

# Mapping from file extensions to language and loader
LANGUAGE_MAPPING = {
    ".py": {"language": "python", "loader": TextLoader},
    ".cs": {"language": "csharp", "loader": TextLoader},
    ".cpp": {"language": "cpp", "loader": TextLoader},
    ".h": {"language": "cpp", "loader": TextLoader},
    ".vb": {"language": "vb", "loader": TextLoader},
    ".html": {"language": "docs", "loader": UnstructuredHTMLLoader},
    ".pdf": {"language": "docs", "loader": PyPDFLoader},
    ".txt": {"language": "docs", "loader": TextLoader}
}

def load_and_process_documents():
    """Loads all documents, sorts them by language, and returns a dictionary of documents."""
    language_docs = {}
    all_files = glob.glob(os.path.join(SOURCE_DOCS_PATH, "**", "*"), recursive=True)
    
    for file_path in all_files:
        if not os.path.isfile(file_path):
            continue

        file_ext = os.path.splitext(file_path)[1].lower()
        mapping = LANGUAGE_MAPPING.get(file_ext)

        if not mapping:
            continue
        
        language = mapping["language"]
        loader_class = mapping["loader"]
        print(f" > Found {language} file: {file_path}")

        try:
            loader_args = {'file_path': file_path}
            if loader_class == TextLoader:
                loader_args['encoding'] = 'utf-8'
            
            loader = loader_class(**loader_args)
            docs = loader.load()
            
            if language not in language_docs:
                language_docs[language] = []
            language_docs[language].extend(docs)
        except Exception as e:
            print(f"   - Failed to load {file_path}: {e}")
            
    return language_docs

def create_vector_stores():
    """Creates a separate FAISS vector store for each detected language."""
    if os.path.exists(VECTOR_STORES_PATH):
        print(f"Removing existing vector stores at '{VECTOR_STORES_PATH}'...")
        shutil.rmtree(VECTOR_STORES_PATH)
    os.makedirs(VECTOR_STORES_PATH)
    
    language_documents = load_and_process_documents()
    
    if not language_documents:
        print("\nNo documents were loaded. Please check your 'source_documents' folder.")
        return

    print("\nSplitting documents and creating embeddings...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    for language, docs in language_documents.items():
        print(f"\n--- Processing language: {language.upper()} ---")
        if not docs:
            print("No documents to process.")
            continue
            
        texts = text_splitter.split_documents(docs)
        print(f"Split into {len(texts)} chunks.")

        print("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(texts, embeddings)
        
        save_path = os.path.join(VECTOR_STORES_PATH, language)
        vector_store.save_local(save_path)
        print(f"Vector store for {language.upper()} saved at {save_path}")

if __name__ == "__main__":
    create_vector_stores()
