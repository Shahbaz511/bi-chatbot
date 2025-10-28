import streamlit as st
import os
import glob
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="BI Broup  Chatbot", page_icon="🤖")
st.title("🤖 Business International Bahrain ")
st.write("Hello! I'm your AI assistant. How can I help you with our Xerox products and services today?")

@st.cache_resource
def load_knowledge_base():
    knowledge_content = ""
    txt_files = glob.glob("knowledge_base/*.txt")
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                knowledge_content += f"\n\n--- {os.path.basename(txt_file)} ---\n{file.read()}"
        except Exception as e:
            st.error(f"Error loading {txt_file}: {e}")
    return knowledge_content

@st.cache_resource
def initialize_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API key not configured")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def get_response(question, model, knowledge_base):
    prompt = f"""You are a helpful customer support assistant for Business Institute Bahrain. Use this business information:

{knowledge_base}

Question: {question}

Answer based ONLY on the information above. If you don't know, suggest contacting info@bi-bh.com."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize
knowledge_base = load_knowledge_base()
model = initialize_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about Xerox products, services, or support..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt, model, knowledge_base)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})