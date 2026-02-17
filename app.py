import streamlit as st

from groq import Groq

import os

from dotenv import load_dotenv



# RAG imports

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS



import tempfile



# ---------------------------------------------------

# LOAD ENV

# ---------------------------------------------------



load_dotenv()



api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")



if not api_key:

    st.error("🚨 GROQ API Key missing!")

    st.stop()



client = Groq(api_key=api_key)



# ---------------------------------------------------

# PAGE CONFIG

# ---------------------------------------------------



st.set_page_config(

    page_title="NeuraBot AI",

    page_icon="🤖",

    layout="wide"

)



# ---------------------------------------------------

# PREMIUM CSS (Looks like funded startup)

# ---------------------------------------------------



st.markdown("""

<style>



[data-testid="stAppViewContainer"]{

    background: radial-gradient(circle at top, #0b1220, #020617);

}



.big-title{

    font-size:42px;

    font-weight:700;

    background: linear-gradient(90deg,#60a5fa,#a78bfa);

    -webkit-background-clip:text;

    -webkit-text-fill-color:transparent;

}



.chat-bubble{

    background: rgba(255,255,255,0.04);

    padding:18px;

    border-radius:14px;

    border:1px solid rgba(255,255,255,0.06);

    line-height:1.7;

}



</style>

""", unsafe_allow_html=True)



# ---------------------------------------------------

# SIDEBAR

# ---------------------------------------------------



with st.sidebar:



    st.markdown("## 🚀 NeuraBot")

    st.caption("Next-Gen AI Copilot")



    if st.button("➕ New Chat"):

        st.session_state.messages = []



    st.divider()



    uploaded_file = st.file_uploader(

        "📄 Upload PDF for AI Analysis",

        type=["pdf"]

    )



    st.divider()



    st.success("⚡ Powered by Groq")

    st.caption("Built by Aakash 🚀")



# ---------------------------------------------------

# HEADER

# ---------------------------------------------------



st.markdown('<div class="big-title">NeuraBot AI</div>', unsafe_allow_html=True)

st.caption("Enterprise-Grade Intelligent Assistant")



# ---------------------------------------------------

# CHAT MEMORY

# ---------------------------------------------------



if "messages" not in st.session_state:

    st.session_state.messages = []



if "vector_store" not in st.session_state:

    st.session_state.vector_store = None



# ---------------------------------------------------

# PROCESS PDF (RAG MAGIC)

# ---------------------------------------------------



if uploaded_file:



    with st.spinner("🔬 Processing document..."):



        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded_file.read())

            tmp_path = tmp.name



        loader = PyPDFLoader(tmp_path)

        pages = loader.load()



        splitter = RecursiveCharacterTextSplitter(

            chunk_size=1000,

            chunk_overlap=150

        )



        chunks = splitter.split_documents(pages)



        embeddings = HuggingFaceEmbeddings(

            model_name="sentence-transformers/all-MiniLM-L6-v2"

        )



        vector_db = FAISS.from_documents(chunks, embeddings)



        st.session_state.vector_store = vector_db



    st.success("✅ Document ready! Ask questions about it.")



# ---------------------------------------------------

# DISPLAY CHAT

# ---------------------------------------------------



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(

            f'<div class="chat-bubble">{msg["content"]}</div>',

            unsafe_allow_html=True

        )



# ---------------------------------------------------

# INPUT

# ---------------------------------------------------



prompt = st.chat_input("Message NeuraBot...")



# ---------------------------------------------------

# STREAM FUNCTION (Typing Effect)

# ---------------------------------------------------



def stream_text(text):

    placeholder = st.empty()

    full = ""



    for char in text:

        full += char

        placeholder.markdown(

            f'<div class="chat-bubble">{full}▌</div>',

            unsafe_allow_html=True

        )

    placeholder.markdown(

        f'<div class="chat-bubble">{full}</div>',

        unsafe_allow_html=True

    )



# ---------------------------------------------------

# RESPONSE LOGIC

# ---------------------------------------------------



if prompt:



    st.session_state.messages.append(

        {"role": "user", "content": prompt}

    )



    with st.chat_message("user"):

        st.markdown(

            f'<div class="chat-bubble">{prompt}</div>',

            unsafe_allow_html=True

        )



    # ⭐ If PDF exists → RAG

    if st.session_state.vector_store:



        docs = st.session_state.vector_store.similarity_search(

            prompt,

            k=3

        )



        context = "\n\n".join(

            d.page_content for d in docs

        )



        final_prompt = f"""

        Answer using ONLY this context:



        {context}



        User question:

        {prompt}

        """



        messages = [

            {"role":"system",

             "content":"You are a helpful AI that answers from documents."},

            {"role":"user","content":final_prompt}

        ]



    else:

        messages = [

            {"role":"system",

             "content":"You are NeuraBot, an elite AI assistant."},

            {"role":"user","content":prompt}

        ]



    with st.chat_message("assistant"):

        completion = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=messages,

            temperature=0.7

        )



        response = completion.choices[0].message.content



        stream_text(response)



    st.session_state.messages.append(

        {"role":"assistant","content":response}

    )

