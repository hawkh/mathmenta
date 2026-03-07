"""
Cross-Encoder Reranking for Mathematical RAG

This module provides intelligent reranking of retrieved documents
using cross-encoder models optimized for mathematical content.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Cross-encoder reranking disabled.")
    logger.warning("Install with: pip install sentence-transformers")


@dataclass
class RerankingResult:
    """Container for reranking results."""
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    combined_score: float
    rank: int


class MathematicalReranker:
    """
    Rerank retrieved mathematical documents using cross-encoder.
    
    The reranker improves retrieval quality by:
    1. Using cross-encoder for query-document relevance scoring
    2. Boosting documents with mathematical formulas when appropriate
    3. Considering topic matching and difficulty level
    4. Combining FAISS similarity with cross-encoder scores
    
    Examples:
        >>> reranker = MathematicalReranker()
        >>> docs = [
        ...     {'content': 'Quadratic formula: x = (-b ± √(b²-4ac))/2a', 'metadata': {'topic': 'algebra'}},
        ...     {'content': 'Derivative of x² is 2x', 'metadata': {'topic': 'calculus'}}
        ... ]
        >>> results = reranker.rerank("solve x² + 2x + 1 = 0", docs, top_k=1)
        >>> results[0]['content']  # Should boost quadratic formula
    """
    
    # Available cross-encoder models ranked by quality/speed tradeoff
    MODELS = {
        'fast': 'cross-encoder/ms-marco-MiniLM-L-6-v2',  # Fast, good for production
        'balanced': 'cross-encoder/ms-marco-TinyBERT-L-2-v2',  # Very fast
        'accurate': 'cross-encoder/stsb-roberta-base',  # More accurate, slower
    }
    
    def __init__(self, model_name: str = 'fast', use_gpu: bool = False):
        """
        Initialize the reranker.
        
        Args:
            model_name: Model preset ('fast', 'balanced', 'accurate') or custom model name
            use_gpu: Whether to use GPU acceleration
        """
        self.model_name = self.MODELS.get(model_name, model_name)
        self.use_gpu = use_gpu
        self.model = None
        self.initialized = False
        
        # Scoring weights
        self.faiss_weight = 0.3  # Weight for original FAISS similarity
        self.rerank_weight = 0.7  # Weight for cross-encoder score
        
        # Topic boost factors
        self.topic_boosts = {
            'formula': 1.2,
            'theorem': 1.1,
            'example': 1.0,
            'definition': 1.05,
        }
        
        logger.info(f"MathematicalReranker initialized with model: {self.model_name}")
    
    def _initialize_model(self):
        """Lazy initialization of cross-encoder model."""
        if not self.initialized and CROSS_ENCODER_AVAILABLE:
            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self.model = CrossEncoder(self.model_name, device='cuda' if self.use_gpu else 'cpu')
                self.initialized = True
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                self.initialized = False
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
        use_topic_boost: bool = True,
        query_topic: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of retrieved documents with 'content' and 'metadata'
            top_k: Number of documents to return after reranking
            use_topic_boost: Whether to apply topic-based boosting
            query_topic: Optional topic from query for boosting
            
        Returns:
            List of reranked documents with scores
            
        Examples:
            >>> reranker.rerank("quadratic formula", docs, top_k=3)
            [{'content': '...', 'rerank_score': 0.95, 'rank': 1}, ...]
        """
        if not documents:
            return []
        
        # Initialize model if needed
        if not self.initialized:
            self._initialize_model()
        
        # If cross-encoder not available, return original order with scores
        if not self.initialized or not CROSS_ENCODER_AVAILABLE:
            logger.warning("Cross-encoder not available, returning original ranking")
            return self._fallback_ranking(documents, top_k)
        
        try:
            start_time = time.time()
            
            # Prepare query-document pairs for cross-encoder
            pairs = [[query, doc['content']] for doc in documents]
            
            # Get cross-encoder scores
            rerank_scores = self.model.predict(pairs)
            
            # Combine scores and apply boosts
            scored_docs = []
            for i, (doc, rerank_score) in enumerate(zip(documents, rerank_scores)):
                # Get original FAISS score if available
                faiss_score = doc.get('similarity_score', 0.5)
                
                # Combined score
                combined = (
                    self.faiss_weight * faiss_score +
                    self.rerank_weight * float(rerank_score)
                )
                
                # Apply topic boost if enabled
                if use_topic_boost:
                    boost = self._get_topic_boost(doc, query_topic)
                    combined *= boost
                
                scored_docs.append({
                    'content': doc['content'],
                    'metadata': doc.get('metadata', {}),
                    'original_score': faiss_score,
                    'rerank_score': float(rerank_score),
                    'combined_score': combined,
                    'index': i
                })
            
            # Sort by combined score (descending)
            scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Assign ranks and take top_k
            results = []
            for rank, doc in enumerate(scored_docs[:top_k], 1):
                doc['rank'] = rank
                results.append(doc)
            
            elapsed = time.time() - start_time
            logger.info(f"Reranked {len(documents)} documents in {elapsed:.3f}s, top_k={top_k}")
            
            return results
            
        except Exception as e:
            logger.exception(f"Error during reranking: {e}")
            return self._fallback_ranking(documents, top_k)
    
    def _get_topic_boost(self, doc: Dict[str, Any], query_topic: Optional[str]) -> float:
        """
        Get boost factor based on document type and topic matching.
        
        Args:
            doc: Document with metadata
            query_topic: Topic from query
            
        Returns:
            Boost factor (>= 1.0)
        """
        boost = 1.0
        metadata = doc.get('metadata', {})
        
        # Boost by document type
        doc_type = metadata.get('type', 'example')
        if doc_type in self.topic_boosts:
            boost *= self.topic_boosts[doc_type]
        
        # Boost if topic matches
        if query_topic:
            doc_topic = metadata.get('topic', '')
            if doc_topic and doc_topic.lower() == query_topic.lower():
                boost *= 1.3  # Strong boost for topic match
        
        # Boost documents containing formulas
        content = doc.get('content', '')
        if any(pattern in content for pattern in ['=', '\\', '√', '∫', '∑']):
            boost *= 1.1  # Slight boost for formula-heavy content
        
        return boost
    
    def _fallback_ranking(
        self,
        documents: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback ranking when cross-encoder is unavailable.
        
        Uses FAISS similarity scores with simple heuristics.
        """
        scored_docs = []
        for i, doc in enumerate(documents):
            faiss_score = doc.get('similarity_score', 0.5)
            boost = self._get_topic_boost(doc, None)
            
            scored_docs.append({
                'content': doc['content'],
                'metadata': doc.get('metadata', {}),
                'original_score': faiss_score,
                'rerank_score': faiss_score,  # Use FAISS score as proxy
                'combined_score': faiss_score * boost,
                'rank': 0
            })
        
        # Sort and assign ranks
        scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)
        for rank, doc in enumerate(scored_docs[:top_k], 1):
            doc['rank'] = rank
        
        return scored_docs[:top_k]
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_batch: List[List[Dict[str, Any]]],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank documents for multiple queries in batch.
        
        Args:
            queries: List of queries
            documents_batch: List of document lists (one per query)
            top_k: Number of documents to return per query
            
        Returns:
            List of reranked document lists
        """
        if len(queries) != len(documents_batch):
            raise ValueError("Number of queries must match number of document lists")
        
        results = []
        for query, docs in zip(queries, documents_batch):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        
        return results
    
    def get_score_explanation(self, doc: Dict[str, Any]) -> str:
        """
        Get human-readable explanation of document score.
        
        Args:
            doc: Reranked document
            
        Returns:
            Explanation string
        """
        parts = []
        
        if 'rerank_score' in doc:
            parts.append(f"Relevance: {doc['rerank_score']:.2f}")
        
        if 'original_score' in doc:
            parts.append(f"Similarity: {doc['original_score']:.2f}")
        
        metadata = doc.get('metadata', {})
        if 'topic' in metadata:
            parts.append(f"Topic: {metadata['topic']}")
        
        if 'type' in metadata:
            parts.append(f"Type: {metadata['type']}")
        
        return ", ".join(parts)


class HybridRetriever:
    """
    Hybrid retrieval combining FAISS + reranking.
    
    This wrapper around the existing Retriever adds reranking
    to the retrieval pipeline.
    
    Examples:
        >>> hybrid = HybridRetriever(existing_retriever)
        >>> results = hybrid.retrieve_with_rerank("quadratic formula", top_k=5)
    """
    
    def __init__(
        self,
        base_retriever: Any,
        reranker: Optional[MathematicalReranker] = None,
        initial_k: int = 20,
        final_k: int = 5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            base_retriever: Existing Retriever instance
            reranker: MathematicalReranker instance (created if None)
            initial_k: Number of documents to retrieve initially
            final_k: Number of documents to return after reranking
        """
        self.base_retriever = base_retriever
        self.reranker = reranker or MathematicalReranker()
        self.initial_k = initial_k
        self.final_k = final_k
        
        logger.info(f"HybridRetriever initialized: initial_k={initial_k}, final_k={final_k}")
    
    def retrieve_with_rerank(
        self,
        query: str,
        topic: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with reranking.
        
        Args:
            query: Search query
            topic: Optional topic filter
            top_k: Override final_k for this query
            
        Returns:
            Reranked list of documents
        """
        final_k = top_k or self.final_k
        
        # Initial retrieval (get more candidates)
        initial_docs = self.base_retriever.retrieve(query, top_k=self.initial_k)
        
        if not initial_docs:
            return []
        
        # Apply topic filter if specified
        if topic:
            initial_docs = [
                doc for doc in initial_docs
                if doc.get('metadata', {}).get('topic', '').lower() == topic.lower()
            ]
        
        # Rerank
        reranked = self.reranker.rerank(
            query,
            initial_docs,
            top_k=final_k,
            query_topic=topic
        )
        
        return reranked
    
    def retrieve_with_context(
        self,
        query: str,
        topic: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> str:
        """
        Retrieve and format context with reranking.
        
        Args:
            query: Search query
            topic: Optional topic filter
            top_k: Override final_k for this query
            
        Returns:
            Formatted context string
        """
        docs = self.retrieve_with_rerank(query, topic, top_k)
        
        if not docs:
            return "No relevant context found."
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc['content']
            source = doc.get('metadata', {}).get('source', 'Unknown')
            context_parts.append(f"[{i}] From {source}:\n{content}")
        
        return "\n\n".join(context_parts)


def get_reranker() -> MathematicalReranker:
    """Get a configured MathematicalReranker instance."""
    return MathematicalReranker()


def create_hybrid_retriever(
    base_retriever: Any,
    initial_k: int = 20,
    final_k: int = 5
) -> HybridRetriever:
    """Create a HybridRetriever with default settings."""
    return HybridRetriever(
        base_retriever,
        initial_k=initial_k,
        final_k=final_k
    )
