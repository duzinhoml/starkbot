import streamlit as st
import pinecone
import uuid
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from dotenv import load_dotenv
import google.generativeai as genai
# import json
# import subprocess
import hashlib
from serpapi.google_search import GoogleSearch

load_dotenv()

# Streamlit UI elements
st.title("StarkBot")

# Pinecone configuration
pinecone_index_name = "starkbot"

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
serpapi_key = os.getenv("SERPAPI_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Index Settings
pinecone_dimension = 768
pinecone_metric = "cosine"
pinecone_cloud = "aws"
pinecone_region = "us-east-1"

from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Data chunking function
def chunk_data(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Sidebar elements
with st.sidebar:
    # Gemini model selection
    gemini_model = st.selectbox(
        "Select Gemini Model",
        ("gemini-2.0-flash-exp", "gemini-1.5-pro-latest"),
    )

    # File upload
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])

    # AI Model Parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        max_tokens = st.slider("Max Tokens", min_value=1, max_value=8192, value=1096)
    with col2:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with col3:
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.01)

    # System Prompt
    system_prompt = st.text_area("System Prompt", value="You are a helpful and harmless AI assistant. Please answer questions to the best of your ability, even if they are complex or controversial. Avoid generating responses that are based on copyrighted material. Avoid saying: 'According to the text', 'Based on the provided text', 'Based on the excerpts'. Simply provide the answer.")

    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)
    generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=max_tokens, temperature=temperature, top_p=top_p)
    gemini_llm = genai.GenerativeModel(model_name=gemini_model, generation_config=generation_config)

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.user_question = ""

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
# Chat interface
user_question = st.text_area("Ask a question:", key="user_question", value=st.session_state.get("user_question", ""))
ask_button = st.button("Ask", key="ask_button")

# Upsert data to Pinecone
if uploaded_file is not None:
    # Generate a unique ID for the document
    doc_id = hashlib.md5(uploaded_file.getbuffer()).hexdigest()

    index = pinecone.Index(pinecone_index_name)

    # Check if the document ID already exists in the index
    if doc_id not in [v[0] for v in index.fetch(ids=[doc_id]).vectors.values()]:
        # Save uploaded file to a temporary location
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
        file_path = uploaded_file.name
        file_type = uploaded_file.type.split("/")[1]  # Extract file extension
    
        chunks = chunk_data(file_path, file_type)
    
        for chunk in chunks:
            chunk_id = uuid.uuid4().hex
            #vector = gemini_embeddings.embed_query(chunk.page_content)
            #index.upsert([(chunk_id, vector, {"text": chunk.page_content})])
            embeddings = genai.embed_content(
                model="models/embedding-001",
                content=chunk.page_content,
                task_type="retrieval_document",
                title="Embedding of the document"
            )
            index.upsert(vectors=[(chunk_id, embeddings['embedding'], {"text": chunk.page_content})])
    
        st.success("Data uploaded to Pinecone!")
    else:
        st.info("File already exists in Pinecone. Skipping upsert.")

if ask_button:
    # RAG pipeline implementation
    index = pinecone.Index(pinecone_index_name)
    # Fetch relevant chunks
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]

    prompt_with_context = f"""{system_prompt}
    Context:
    {chr(10).join(contexts)}
    Question: {user_question}"""

    try:
        response = gemini_llm.generate_content(prompt_with_context)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.write(f"An error occurred: {e}")
