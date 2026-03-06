"""
Build the FAISS vector index from knowledge base files.
Run this script once to create the vector store.
"""
import sys
import os
# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

from rag.retriever import RAGRetriever
from config import Config


def main():
    """Build the vector index."""
    print("=" * 60)
    print("Math Mentor - Building Knowledge Base Index")
    print("=" * 60)

    # Validate API key
    if not Config.ANTHROPIC_API_KEY:
        print("\n[WARNING] ANTHROPIC_API_KEY not set in .env file")
        print("   The RAG index will be built, but the app won't work without it.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    retriever = RAGRetriever()

    print("\n[*] Building index from knowledge base files...")
    print(f"   Knowledge base directory: {Config.KNOWLEDGE_BASE_DIR}")
    print(f"   Vector store directory: {Config.VECTOR_STORE_DIR}")

    try:
        stats = retriever.build_index()

        print("\n[SUCCESS] Index built successfully!")
        print(f"\n[STATISTICS]:")
        print(f"   Files processed: {stats['files_processed']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Topics: {', '.join(stats['topics'])}")
        print(f"\n[INFO] Vector store saved to: {Config.VECTOR_STORE_DIR}")
        print("\nYou can now run the app with: streamlit run app.py")

    except Exception as e:
        print(f"\n[ERROR] Error building index: {e}")
        print("\nMake sure:")
        print("  1. Knowledge base files exist in knowledge_base/")
        print("  2. Required packages are installed (faiss-cpu, sentence-transformers)")
        sys.exit(1)


if __name__ == "__main__":
    main()
