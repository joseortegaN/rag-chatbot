# Chatbot con tus documentos (RAG)

Chatbot que responde preguntas basándose en tus propios documentos (PDF, TXT, MD),
usando un pipeline RAG (Retrieval-Augmented Generation). 100% gratuito.

## Stack

- **LLM**: [Groq](https://console.groq.com) (GPT-OSS 120B) — capa gratuita, sin tarjeta de crédito
- **Embeddings**: `sentence-transformers` — corre localmente, sin costo ni API
- **Base vectorial**: ChromaDB — local, sin servidor externo
- **Interfaz**: Streamlit

## Instalación

1. Clona el repo e instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Consigue una API key gratuita de Groq en https://console.groq.com/keys

3. Copia `.env.example` a `.env` y pega tu key:

   ```bash
   cp .env.example .env
   ```

## Uso

1. Pon tus archivos (`.pdf`, `.txt`, `.md`) en la carpeta `docs/` (ya incluye un ejemplo).

2. Indexa los documentos:

   ```bash
   python ingest.py
   ```

3. Corre la app:

   ```bash
   streamlit run app.py
   ```

4. Abre el navegador en `http://localhost:8501` y pregunta sobre tus documentos.

## Deploy gratis

1. Sube este repo a GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io), conecta tu cuenta de GitHub y selecciona el repo.
3. En "Advanced settings" agrega tu `GROQ_API_KEY` como secreto.
4. Deploy. Listo, tu chatbot queda público con una URL.

## Cómo funciona

1. `ingest.py` divide tus documentos en fragmentos, genera embeddings localmente y los
   guarda en una base vectorial (`chroma_db/`).
2. `app.py` recibe tu pregunta, busca los fragmentos más relevantes en la base vectorial,
   y se los pasa como contexto al LLM (Groq) para que genere la respuesta.
