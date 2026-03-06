"""
RAG Retriever module for Math Mentor.
Handles vector store creation and similarity search retrieval.
"""
import os
from typing import List, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config


class RAGRetriever:
    """
    Retrieval-Augmented Generation retriever using FAISS vector store.
    Provides semantic search over the mathematics knowledge base.
    """
    
    def __init__(self, vector_store_dir: str = None):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store_dir: Directory to store/load FAISS index
        """
        self.vector_store_dir = vector_store_dir or Config.VECTOR_STORE_DIR
        self.embeddings = None
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        if self.embeddings is None:
            # Use a simpler embedding model that works with Python 3.13
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Use a lightweight model that doesn't require heavy dependencies
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def build_index(self, knowledge_base_dir: str = None) -> Dict[str, Any]:
        """
        Build the vector index from knowledge base files.
        
        Args:
            knowledge_base_dir: Directory containing .txt knowledge files
            
        Returns:
            Dictionary with build statistics
        """
        kb_dir = knowledge_base_dir or Config.KNOWLEDGE_BASE_DIR
        self._initialize_embeddings()
        
        documents = []
        stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'topics': []
        }
        
        # Load all knowledge base files
        for filename in os.listdir(kb_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(kb_dir, filename)
                topic = filename.replace('.txt', '')
                stats['topics'].append(topic)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = self.text_splitter.split_text(content)
                
                # Create documents with metadata
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            'source': filename,
                            'topic': topic,
                            'chunk_id': i
                        }
                    )
                    documents.append(doc)
                
                stats['files_processed'] += 1
                stats['total_chunks'] += len(chunks)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save to disk
        os.makedirs(self.vector_store_dir, exist_ok=True)
        self.vector_store.save_local(self.vector_store_dir)
        
        return stats
    
    def load_index(self) -> bool:
        """
        Load existing vector index from disk.
        
        Returns:
            True if successful, False if index not found
        """
        self._initialize_embeddings()
        
        if not os.path.exists(self.vector_store_dir):
            return False
        
        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of documents with content and metadata
        """
        if self.vector_store is None:
            if not self.load_index():
                raise ValueError("Vector store not loaded. Run build_index.py first.")
        
        k = top_k or Config.TOP_K_RETRIEVAL
        
        # Perform similarity search
        docs = self.vector_store.similarity_search(query, k=k)
        
        # Also get scores using similarity_search_with_score
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': 1 - score  # Convert distance to similarity
            })
        
        return results
    
    def retrieve_with_context(self, query: str, topic: str = None) -> str:
        """
        Retrieve and format relevant context for a query.
        
        Args:
            query: The search query
            topic: Optional topic filter
            
        Returns:
            Formatted context string
        """
        results = self.retrieve(query)
        
        if topic:
            results = [r for r in results if r['metadata'].get('topic') == topic]
        
        if not results:
            return "No relevant context found."
        
        # Format results
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            topic = result['metadata'].get('topic', 'Unknown')
            score = result['similarity_score']
            
            context_parts.append(
                f"[Source {i}] {source} (Topic: {topic}, Relevance: {score:.2f})\n"
                f"{result['content']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics in the knowledge base."""
        topics = set()
        kb_dir = Config.KNOWLEDGE_BASE_DIR
        for filename in os.listdir(kb_dir):
            if filename.endswith('.txt'):
                topics.add(filename.replace('.txt', ''))
        return sorted(list(topics))


# Singleton instance
_retriever_instance = None


def get_retriever() -> RAGRetriever:
    """Get or create the RAG retriever singleton."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance
