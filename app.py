"""
app.py — CLI for the GenAI Document Q&A tool.

Usage:
    python app.py --docs ./sample_docs --ask "What is covered in Phase 3?"
    python app.py --docs ./sample_docs            # interactive mode
"""

import argparse
import sys

from rag import DocumentStore


def main():
    parser = argparse.ArgumentParser(description="Ask questions over a folder of documents using Claude + RAG.")
    parser.add_argument("--docs", required=True, help="Folder containing .txt/.md files to index")
    parser.add_argument("--ask", help="A single question to ask (non-interactive mode)")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve per query")
    args = parser.parse_args()

    print(f"Indexing documents in '{args.docs}'...")
    try:
        store = DocumentStore(args.docs)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    print(f"Indexed {len(store.chunks)} chunks.\n")

    if args.ask:
        answer = store.ask(args.ask, top_k=args.top_k)
        print(f"Q: {args.ask}\n\nA: {answer}")
        return

    print("Interactive mode. Type a question, or 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit"):
            break
        if not query:
            continue
        answer = store.ask(query, top_k=args.top_k)
        print(f"\nClaude: {answer}\n")


if __name__ == "__main__":
    main()
