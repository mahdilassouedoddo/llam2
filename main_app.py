#importing necessary packages
import os 
import base64 
import gc 
import random 
import tempfile 
import time 
import uuid
import sqlite3 
from huggingface_hub import configure_http_backend 
import requests 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core import Settings 
from llama_index.core import VectorStoreIndex, ServiceContext, load_index_from_storage, StorageContext, SimpleDirectoryReader 
from llama_index.core import PromptTemplate 
from llama_index.llms.openai import OpenAI 
from llama_index.llms.llama_cpp import LlamaCPP 
from llama_cpp import Llama 
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt, 
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from PIL import Image
import streamlit as st
from huggingface_hub import configure_http_backend
import requests

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session
configure_http_backend(backend_factory=backend_factory)
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.loaded_history = []
    st.session_state.selected_history = None
session_id = st.session_state.id
client = None
 
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    st.session_state.loaded_history = []
    st.session_state.selected_history = None
    gc.collect()
# Loading the data
documents = SimpleDirectoryReader(
    "C:\\Users\\MLASSOUED\\Documents\\llam2\\data"
).load_data()
 #Loading the LLM
llm = LlamaCPP(
    model_path="C:\Users\MLASSOUED\OneDrive - ODDO BHF\Bureau\llam2\projects\Phi-3-medium-128k-instruct-Q5_K_M.gguf",
    temperature=0,
    max_new_tokens=500,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 0},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
image = Image.open('logo.png')
st.image(image, width=200)
# Loading the embedding model 
embed_model = HuggingFaceEmbedding(
 
    model_name="BAAI/bge-large-en-v1.5"
 
)
 
Settings.llm = llm
 
Settings.embed_model = embed_model
# Indexing the documents and loading them
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
index.storage_context.persist(persist_dir='./storage')
index.set_index_id("vector_index")
index.storage_context.persist("./storage")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)

# Chat history database creation 
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (session_id TEXT, role TEXT, content TEXT)''')
    conn.commit()
    conn.close()
 
init_db()
 
def save_to_db(session_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)", (str(session_id), role, content))
    conn.commit()
    conn.close()
 
def load_chat_history():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT session_id FROM chat_history")
    session_ids = c.fetchall()
    session_histories = {}
    for session_id in session_ids:
 
        c.execute("SELECT role, content FROM chat_history WHERE session_id=?", (session_id[0],))
 
        session_histories[session_id[0]] = c.fetchall()
    conn.close()
 
    return session_histories
# Initialisation
qa_prompt_tmpl_str = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information above I want you to think step by step to answer the query in a detailed manner, in case you don't know the answer say 'I don't know. Provide answer only in French'.\n"
    "Query: {query_str}\n"
    "Answer: "
 
)

qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
query_engine = index.as_query_engine(similarity_top_k=2, llm=llm)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt_tmpl})
#  Rendering and the web page
st.sidebar.header("Chat History") 
chat_histories = load_chat_history()
for session_id, history in chat_histories.items():
    content = history[0]
    with st.sidebar:
        # for role, content in history:
        #     st.write(f"{role}: {content}"
          if st.button(f"{content[1]}",key=session_id):
 
            st.session_state.selected_history = history
 

 
col1, col2 = st.columns([6, 1])
 
with col1:
    st.header("Chat with Wiki")
with col2:
    st.button("Clear ↺", on_click=reset_chat)
if "messages" not in st.session_state:
    reset_chat()
# Load selected chat history into the main chat area
if st.session_state.selected_history:
    for role, content in st.session_state.selected_history:
        st.session_state.messages.append({"role": role, "content": content})
    st.session_state.selected_history = None  # Clear after loading
 
# Render all chat messages, including the loaded history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = query_engine.query(prompt)
        response_1 = response.response
        message_placeholder.markdown(response_1)

        for i, node in enumerate(response.source_nodes):
            with st.expander(f"Nom du fichier : {node.metadata['file_name']}"):
                st.markdown(
                    f"""
                    <div style="font-size:16px; font-family:sans-serif;">
                        Contenu : {node.text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    st.session_state.messages.append({"role": "assistant", "content": response_1})
    save_to_db(session_id, "user", prompt)
    save_to_db(session_id, "assistant", response_1)
 