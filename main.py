import os
from typing import List
import streamlit as st
import chromadb
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import BertTokenizer

# CONSTANTS =====================================================
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1

# This is the maximum chunk size allowed by the chosen embedding model. You can choose a smaller size.
CHUNK_SIZE = 8192

DOCUMENT_DIR = "E:\\test\\chat-with-pdf"  # The directory where the PDF files should be placed
VECTOR_STORE_DIR = "./vectorstore/"  # The directory where the vectors are stored
COLLECTION_NAME = "collection1"  # ChromaDB collection name
# ===============================================================

# Define your Jina API key directly in the script
JINA_API_KEY = 'jina_268f16cdd7f6410c850adbe32de20171ha3URkzQHnwlpDmy8-yhBXACVzXV'

@st.cache_data
def load_documents() -> List[Document]:
    """Loads the PDF files within the DOCUMENT_DIR constant."""
    try:
        st.write("[+] Loading documents...")

        documents = DirectoryLoader(
            os.path.join(DOCUMENT_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader
        ).load()
        st.success(f"[+] Document loaded, total pages: {len(documents)}")

        return documents
    except Exception as e:
        st.error(f"[-] Error loading the document: {str(e)}")
        return []

@st.cache_data
def chunk_document(_documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", cache_dir=os.environ.get("HF_HOME")
    )
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE // 50,
    )

    st.write(f"[+] Splitting documents...")
    chunks = text_splitter.split_documents(_documents)
    st.success(f"[+] Document splitting done, {len(chunks)} chunks total.")

    return chunks

@st.cache_resource
def create_and_store_embeddings(_embedding_model, _chunks: List[Document]) -> Chroma:
    """Calculates the embeddings and stores them in a chroma vectorstore."""
    try:
        vectorstore = Chroma.from_documents(
            _chunks,
            embedding=_embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        )
        st.success("[+] Vectorstore created.")
        return vectorstore
    except Exception as e:
        st.error(f"[-] Error creating and storing embeddings: {str(e)}")
        raise

@st.cache_resource
def get_vectorstore_retriever(_embedding_model) -> VectorStoreRetriever:
    """Returns the vectorstore."""
    db = chromadb.PersistentClient(VECTOR_STORE_DIR)
    try:
        # Check for the existence of the vectorstore specified by the COLLECTION_NAME
        db.get_collection(COLLECTION_NAME)
        retriever = Chroma(
            embedding_function=_embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        ).as_retriever(search_kwargs={"k": 3})
    except ValueError:
        # The vectorstore doesn't exist, so create it.
        pdf = load_documents()
        if not pdf:
            raise ValueError("No documents were loaded.")
        chunks = chunk_document(pdf)
        retriever = create_and_store_embeddings(_embedding_model, chunks).as_retriever(
            search_kwargs={"k": 3}
        )
    return retriever

def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates the RAG chain."""
    template = """Answer the question based only on the following context.
    Think step by step before providing a detailed answer. I will give you
    $500 if the user finds the response useful.
    <context>
    {context}
    </context>

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = get_vectorstore_retriever(embedding_model)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

def run_chain(chain: Runnable, query: str) -> str:
    """Run the RAG chain with the user query."""
    try:
        response = chain.invoke({"input": query})

        context_output = ""
        for doc in response["context"]:
            context_output += f"[+] {doc.metadata} | content: {doc.page_content[:20]}...\n"

        return context_output + "\n" + response["answer"]
    except Exception as e:
        st.error(f"[-] Error running the chain: {str(e)}")
        return ""

def main():
    st.title("PDF Chat with RAG Chain")

    # Initialize models
    try:
        embedding_model = JinaEmbeddings(
            jina_api_key=JINA_API_KEY,
            model_name=EMBED_MODEL_NAME,
        )
    except Exception as e:
        st.error(f"[-] Failed to initialize JinaEmbeddings: {str(e)}")
        return

    try:
        llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)
    except Exception as e:
        st.error(f"[-] Failed to initialize ChatGroq: {str(e)}")
        return

    # Create RAG chain
    try:
        chain = create_rag_chain(embedding_model=embedding_model, llm=llm)
    except Exception as e:
        st.error(f"[-] Failed to create RAG chain: {str(e)}")
        return

    # User input
    query = st.text_input("Enter a prompt:", "")
    if query:
        with st.spinner("Processing..."):
            response = run_chain(chain, query)
            st.write(response)

if __name__ == "__main__":
    main()
