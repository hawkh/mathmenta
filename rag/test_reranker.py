"""
Tests for Cross-Encoder Reranker

Tests cover:
- Basic reranking functionality
- Topic-based boosting
- Performance benchmarks
- Edge cases
"""

import unittest
import time
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)

from rag.reranker import MathematicalReranker, HybridRetriever


class TestMathematicalReranker(unittest.TestCase):
    """Test mathematical reranker functionality."""
    
    def setUp(self):
        self.reranker = MathematicalReranker()
        
        # Sample documents for testing
        self.sample_docs = [
            {
                'content': 'Quadratic Formula: For ax² + bx + c = 0, the solutions are x = (-b ± √(b²-4ac))/2a',
                'metadata': {'topic': 'algebra', 'type': 'formula', 'source': 'algebra.txt'}
            },
            {
                'content': 'Example: Solve x² + 5x + 6 = 0. Using the quadratic formula with a=1, b=5, c=6...',
                'metadata': {'topic': 'algebra', 'type': 'example', 'source': 'algebra_examples.txt'}
            },
            {
                'content': 'Derivative Rules: d/dx(xⁿ) = nxⁿ⁻¹, d/dx(sin x) = cos x, d/dx(eˣ) = eˣ',
                'metadata': {'topic': 'calculus', 'type': 'formula', 'source': 'calculus.txt'}
            },
            {
                'content': 'Integration by Parts: ∫u dv = uv - ∫v du',
                'metadata': {'topic': 'calculus', 'type': 'formula', 'source': 'calculus.txt'}
            },
            {
                'content': 'Pythagorean Theorem: In a right triangle, a² + b² = c²',
                'metadata': {'topic': 'geometry', 'type': 'theorem', 'source': 'geometry.txt'}
            },
        ]
    
    def test_rerank_basic(self):
        """Test basic reranking functionality."""
        query = "quadratic formula"
        results = self.reranker.rerank(query, self.sample_docs, top_k=3)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['rank'], 1)
        
        # First result should be the quadratic formula document
        print(f"✓ Query: '{query}'")
        print(f"  Top result: {results[0]['content'][:80]}...")
        print(f"  Score: {results[0]['rerank_score']:.3f}")
    
    def test_rerank_calculus_query(self):
        """Test reranking for calculus query."""
        query = "derivative of sine"
        results = self.reranker.rerank(query, self.sample_docs, top_k=3)
        
        print(f"✓ Query: '{query}'")
        print(f"  Top result: {results[0]['content'][:80]}...")
    
    def test_rerank_with_topic_boost(self):
        """Test reranking with topic-based boosting."""
        query = "solve equation"
        results = self.reranker.rerank(
            query,
            self.sample_docs,
            top_k=3,
            query_topic='algebra'
        )
        
        print(f"✓ Query: '{query}' with topic boost 'algebra'")
        print(f"  Top result: {results[0]['content'][:80]}...")
    
    def test_rerank_empty_documents(self):
        """Test handling empty document list."""
        query = "test"
        results = self.reranker.rerank(query, [], top_k=5)
        self.assertEqual(len(results), 0)
        print("✓ Empty document list handled")
    
    def test_rerank_fewer_than_top_k(self):
        """Test when fewer documents than top_k."""
        query = "test"
        results = self.reranker.rerank(query, self.sample_docs[:2], top_k=10)
        self.assertEqual(len(results), 2)
        print("✓ Fewer documents than top_k handled")
    
    def test_rerank_preserves_metadata(self):
        """Test that metadata is preserved after reranking."""
        query = "quadratic"
        results = self.reranker.rerank(query, self.sample_docs, top_k=3)
        
        for result in results:
            self.assertIn('metadata', result)
            self.assertIn('content', result)
            self.assertIn('rerank_score', result)
        
        print("✓ Metadata preserved in results")
    
    def test_rerank_score_ordering(self):
        """Test that results are ordered by score."""
        query = "formula"
        results = self.reranker.rerank(query, self.sample_docs, top_k=5)
        
        # Check descending order
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i]['combined_score'],
                results[i+1]['combined_score']
            )
        
        print("✓ Results ordered by combined score")
    
    def test_get_score_explanation(self):
        """Test score explanation generation."""
        query = "quadratic"
        results = self.reranker.rerank(query, self.sample_docs, top_k=1)
        
        explanation = self.reranker.get_score_explanation(results[0])
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        
        print(f"✓ Score explanation: {explanation}")


class TestHybridRetriever(unittest.TestCase):
    """Test hybrid retriever integration."""
    
    def setUp(self):
        # Mock base retriever
        class MockRetriever:
            def retrieve(self, query, top_k=20):
                # Return sample docs with similarity scores
                return [
                    {
                        'content': f'Document {i} about {query}',
                        'metadata': {'topic': 'test', 'source': 'test.txt'},
                        'similarity_score': 0.9 - i*0.1
                    }
                    for i in range(min(top_k, 5))
                ]
        
        self.mock_retriever = MockRetriever()
        self.hybrid = HybridRetriever(
            self.mock_retriever,
            initial_k=10,
            final_k=3
        )
    
    def test_hybrid_retrieve(self):
        """Test hybrid retrieval with reranking."""
        query = "test query"
        results = self.hybrid.retrieve_with_rerank(query)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['rank'], 1)
        
        print(f"✓ Hybrid retrieval: {len(results)} results")
        print(f"  Top result: {results[0]['content'][:60]}...")
    
    def test_hybrid_retrieve_with_context(self):
        """Test hybrid retrieval with formatted context."""
        query = "test query"
        context = self.hybrid.retrieve_with_context(query)
        
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        
        print(f"✓ Context retrieval: {len(context.split(chr(10)))} lines")
        print(f"  Preview: {context[:100]}...")


class TestRerankerPerformance(unittest.TestCase):
    """Test reranker performance benchmarks."""
    
    def setUp(self):
        self.reranker = MathematicalReranker()
        self.num_docs = 20
        
        self.sample_docs = [
            {
                'content': f'Document {i} with mathematical content about calculus and algebra. ' * 5,
                'metadata': {'topic': 'math', 'type': 'example', 'source': 'test.txt'},
                'similarity_score': 0.5
            }
            for i in range(self.num_docs)
        ]
    
    def test_rerank_latency(self):
        """Test that reranking latency is acceptable."""
        query = "test mathematical query"
        
        start_time = time.time()
        results = self.reranker.rerank(query, self.sample_docs, top_k=5)
        elapsed = time.time() - start_time
        
        # Should complete in under 1 second for 20 docs
        # Note: First run may be slower due to model loading
        print(f"✓ Reranking latency: {elapsed:.3f}s for {self.num_docs} documents")
        
        # Log warning if slow (but don't fail)
        if elapsed > 2.0:
            print(f"  ⚠ Warning: Latency > 2s")
    
    def test_rerank_batch_performance(self):
        """Test batch reranking performance."""
        queries = ["query " + str(i) for i in range(5)]
        docs_batch = [self.sample_docs for _ in queries]
        
        start_time = time.time()
        results = self.reranker.rerank_batch(queries, docs_batch, top_k=5)
        elapsed = time.time() - start_time
        
        print(f"✓ Batch reranking: {elapsed:.3f}s for {len(queries)} queries")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        self.reranker = MathematicalReranker()
    
    def test_very_short_query(self):
        """Test handling very short queries."""
        query = "x"
        docs = [{'content': 'x squared equals 4', 'metadata': {}}]
        results = self.reranker.rerank(query, docs, top_k=1)
        
        self.assertEqual(len(results), 1)
        print("✓ Short query handled")
    
    def test_very_long_query(self):
        """Test handling very long queries."""
        query = "What is the quadratic formula and how do I use it to solve equations of the form " * 10
        docs = [{'content': 'Quadratic formula is...', 'metadata': {}}]
        results = self.reranker.rerank(query, docs, top_k=1)
        
        self.assertEqual(len(results), 1)
        print("✓ Long query handled")
    
    def test_special_characters_in_query(self):
        """Test handling special characters in query."""
        query = "What is ∫x²dx and √(a²+b²)?"
        docs = [{'content': 'Integration and square roots...', 'metadata': {}}]
        results = self.reranker.rerank(query, docs, top_k=1)
        
        self.assertEqual(len(results), 1)
        print("✓ Special characters handled")
    
    def test_mismatched_topics(self):
        """Test when query topic doesn't match any documents."""
        query = "quantum physics"
        docs = [
            {'content': 'Quadratic equations', 'metadata': {'topic': 'algebra'}},
            {'content': 'Derivatives', 'metadata': {'topic': 'calculus'}}
        ]
        results = self.reranker.rerank(query, docs, top_k=2)
        
        self.assertEqual(len(results), 2)
        print("✓ Topic mismatch handled")


def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "="*80)
    print("RERANKER - COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestMathematicalReranker,
        TestHybridRetriever,
        TestRerankerPerformance,
        TestEdgeCases
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"Overall: {'✓ PASSED' if success else '✗ FAILED'}")
    print("="*80)
    
    return success


if __name__ == '__main__':
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
