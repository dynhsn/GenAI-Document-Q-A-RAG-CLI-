# GenAI Document Q&A (RAG CLI)

Command-line tool that answers questions over your own documents using
Retrieval-Augmented Generation (RAG) + the Claude API.

## How it works
1. **Load** — reads every `.txt`/`.md` file in a folder.
2. **Chunk** — splits each document into overlapping ~800-character chunks.
3. **Retrieve** — vectorizes chunks with TF-IDF, finds the top-k chunks most
   relevant to your question via cosine similarity.
4. **Generate** — stuffs the retrieved chunks into a prompt and asks Claude
   to answer, grounded in that context, with source citations.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Usage
```bash
# One-shot question
python app.py --docs ./sample_docs --ask "What does the DevOps course cover in the Kubernetes module?"

# Interactive mode
python app.py --docs ./sample_docs
```

Sample docs included are three course syllabi (Data Science, DevOps,
Python for Cybersecurity) — swap in your own `.txt`/`.md` files to index
anything else.

## Why this design
- **No vector DB dependency** — TF-IDF + cosine similarity is enough to
  demonstrate the retrieval pattern without extra infra, and it's fast to
  read/explain in an interview.
- **Grounded answers** — the prompt explicitly forces Claude to answer only
  from retrieved context and cite sources, reducing hallucination.
- **Swappable retriever** — `DocumentStore.retrieve()` is isolated, so it's
  a one-line swap to a real embedding model (e.g. Voyage AI, OpenAI
  embeddings) or a vector DB (Pinecone, Chroma) later.

## Possible extensions
- Swap TF-IDF for semantic embeddings
- Add PDF/docx ingestion
- Stream responses
- Add a simple Flask API wrapper (`POST /ask`) to expose it as a service

## Resume bullet
> Built a RAG-based document Q&A CLI in Python using the Anthropic Claude
> API — implemented chunking, TF-IDF retrieval, and grounded prompt
> construction to answer questions over custom document sets with source
> citations.
