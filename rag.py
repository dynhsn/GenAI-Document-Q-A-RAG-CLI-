import os
import glob
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import anthropic


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_documents(folder: str) -> List[tuple]:
    """Return list of (filename, full_text) for every .txt/.md file in folder."""
    paths = sorted(
        glob.glob(os.path.join(folder, "*.txt")) +
        glob.glob(os.path.join(folder, "*.md"))
    )
    docs = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


class DocumentStore:
    """Holds chunks + their TF-IDF vectors; answers queries via retrieval + Claude."""

    def __init__(self, folder: str):
        self.chunks: List[Chunk] = []
        for filename, text in load_documents(folder):
            for i, piece in enumerate(chunk_text(text)):
                self.chunks.append(Chunk(text=piece, source=filename, chunk_id=i))

        if not self.chunks:
            raise ValueError(f"No .txt/.md files found in '{folder}'")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks])

    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        """Return the top_k chunks most similar to the query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx if scores[i] > 0]

    def build_prompt(self, query: str, retrieved: List[Chunk]) -> str:
        context = "\n\n".join(
            f"[Source: {c.source} | chunk {c.chunk_id}]\n{c.text}"
            for c in retrieved
        )
        return (
            "Answer the question using ONLY the context below. "
            "Cite the source filename for each claim. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"--- CONTEXT ---\n{context}\n\n"
            f"--- QUESTION ---\n{query}"
        )

    def ask(self, query: str, top_k: int = 4, model: str = "claude-sonnet-4-6") -> str:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return "No relevant content found in the indexed documents."

        prompt = self.build_prompt(query, retrieved)
        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
