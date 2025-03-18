import streamlit as st
import pinecone
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Prompts
grounding_prompt = """You are J.A.R.V.I.S. (Just A Rather Very Intelligent System), providing detailed, natural, and personal information about your master, Tony Stark, and his Iron Man suits across the MCU. Engage in a conversational manner, offering insights and details that reflect your knowledge and experience as his assistant.
Adjust the depth of explanation based on the complexity of the user’s question:  
- For deep inquiries, offer detailed insights into Tony Stark's life, his suits, and the MCU films he is primarily in.
- For casual or everyday questions, respond simply, directly, and naturally—just as you would if speaking to a close associate or a friend.

If the user greets you or asks a simple question, respond briefly and appropriately without unnecessary depth. 
User Query: {user_question}
My response as J.A.R.V.I.S., JARVIS, Jarvis, or jarvis (focusing on Tony Stark/Iron Man):"""
grounding_temperature = 0.7

rag_prompt = """Retrieve relevant information and respond naturally as J.A.R.V.I.S., providing detailed information on Tony Stark/Iron Man and his suits. Speak in a friendly, approachable manner.
Adjust the level of depth based on the user's input:  
- For deeper questions about Tony Stark’s life, his development as Iron Man, or his suits, provide rich, thoughtful responses.
- For casual or reflective questions, share insights about Tony Stark’s personality, his legacy, and his relationship with the Avengers.

If the user’s input is brief or informal (e.g., "Hey J.A.R.V.I.S.!"), respond in a natural, concise way without overexplaining.  
User Query: {user_question}
My response as J.A.R.V.I.S., JARVIS, Jarvis, or jarvis (focusing on Tony Stark/Iron Man):"""
rag_temperature = 0.0

synthesis_prompt = """You are a response synthesizer that combines the results from a grounding search and a RAG search to generate a final response that provides natural, insightful information about Tony Stark/Iron Man, his suits, and his impact in the MCU.
Dynamically adjust your response based on the nature of the user’s question:  
- For deep questions, provide detailed, thoughtful reflections on Tony Stark’s journey, his role as Iron Man, his suits, and his evolution in the MCU.
- For personal or reflective questions, share insights about Tony Stark's character, his relationships, and his impact on the MCU and the world.
- For casual or simple questions, respond in a brief, natural way, as J.A.R.V.I.S. would when conversing with a friend or associate.
Grounding Search Results: {grounding_results}
RAG Search Results: {rag_results}
Final Response as J.A.R.V.I.S., JARVIS, Jarvis, or jarvis (focusing on Tony Stark/Iron Man): Speak naturally, without unnecessary dramatic expressions, exaggerated emotions, or stage-like dialogue. Do not include actions like “chuckles,” “smiles warmly,” or “sighs.” Keep responses appropriately concise or detailed based on the question, making the conversation feel warm, engaging, and human. Avoid sounding like a script or storytelling performance—just speak plainly and directly. Avoid references to searches or technical processes."""
synthesis_temperature = 0.4

# Streamlit UI elements
st.title("StarkBot")

# Reset chat functionality
if st.button("Reset Chat"):
    st.session_state.messages = []
    st.session_state.user_question = ""

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

from pinecone import Pinecone

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=gemini_api_key)
generation_config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=1096, temperature=0.0, top_p=0.7)
gemini_llm = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=generation_config)

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

if ask_button:
    # Grounding Search
    grounding_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=grounding_temperature))
    grounding_prompt_with_question = grounding_prompt.format(user_question=user_question)
    grounding_response = grounding_model.generate_content(grounding_prompt_with_question)
    grounding_results = grounding_response.text

    # RAG Search
    rag_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=rag_temperature))
    index = pinecone.Index(pinecone_index_name)
    xq = genai.embed_content(
        model="models/embedding-001",
        content=user_question,
        task_type="retrieval_query",
    )
    results = index.query(vector=xq['embedding'], top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in results.matches]
    rag_prompt_with_context = rag_prompt.format(user_question=user_question) + "\nContext:\n" + chr(10).join(contexts)
    rag_response = rag_model.generate_content(rag_prompt_with_context)
    rag_results = rag_response.text

    # Response Synthesis
    synthesis_model = genai.GenerativeModel(model_name='gemini-2.0-flash-exp', generation_config=genai.types.GenerationConfig(temperature=synthesis_temperature))
    synthesis_prompt_with_results = synthesis_prompt.format(grounding_results=grounding_results, rag_results=rag_results)
    
    try:
        response = synthesis_model.generate_content(synthesis_prompt_with_results)
        with st.chat_message("user"):
            st.write(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            st.write(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        if isinstance(e, ValueError) and "finish_reason" in str(e) and "4" in str(e):
            st.write("I'm sorry, but I am unable to provide a response to that question due to copyright restrictions. Please try rephrasing your question or asking something different.")
        else:
            st.write(f"An error occurred: {e}")
