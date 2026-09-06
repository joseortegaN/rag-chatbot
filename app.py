import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

DB_DIR = "chroma_db"
MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Chatbot con tus documentos", page_icon="📄")
st.title("📄 Chatbot con tus documentos")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection("documentos", embedding_function=embed_fn)


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Falta GROQ_API_KEY. Copia .env.example a .env y agrega tu key gratuita de https://console.groq.com/keys")
        st.stop()
    return Groq(api_key=api_key)


collection = get_collection()
if collection.count() == 0:
    st.warning("No hay documentos indexados todavía. Pon archivos en 'docs/' y corre 'python ingest.py' primero.")
    st.stop()

client = get_groq_client()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Pregunta algo sobre tus documentos...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = collection.query(query_texts=[question], n_results=4)
    context = "\n\n---\n\n".join(results["documents"][0])
    sources = sorted({m["source"] for m in results["metadatas"][0]})

    system_prompt = (
        "Responde la pregunta del usuario usando SOLO el siguiente contexto. "
        "Si la respuesta no está en el contexto, di que no lo sabes.\n\n"
        f"Contexto:\n{context}"
    )

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content
        st.markdown(answer)
        st.caption(f"Fuentes: {', '.join(sources)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
