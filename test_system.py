"""
Test script for Math Mentor multi-agent system.
Tests core functionality without requiring API keys.
"""
import sys
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from config import Config
        print("  [OK] Config imported")
        
        from rag.retriever import get_retriever
        print("  [OK] RAG retriever imported")
        
        from memory.store import get_memory_store
        print("  [OK] Memory store imported")
        
        from graph import get_agent_graph
        print("  [OK] Agent graph imported")
        
        from agents.nodes import get_agent_nodes
        print("  [OK] Agent nodes imported")
        
        from input_processing import get_ocr_processor, get_asr_processor
        print("  [OK] Input processors imported")
        
        from utils import safe_calculate
        print("  [OK] Utils imported")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Import failed: {e}")
        return False


def test_vector_store():
    """Test vector store loading and retrieval."""
    print("\nTesting vector store...")
    
    try:
        from rag.retriever import get_retriever
        
        retriever = get_retriever()
        loaded = retriever.load_index()
        
        if loaded:
            print("  [OK] Vector index loaded")
            
            # Test retrieval
            results = retriever.retrieve("probability", top_k=2)
            if len(results) > 0:
                print(f"  [OK] Retrieved {len(results)} results")
            else:
                print("  [FAIL] No results retrieved")
                return False
        else:
            print("  [FAIL] Failed to load vector index")
            return False
        
        return True
    except Exception as e:
        print(f"  [FAIL] Vector store test failed: {e}")
        return False


def test_memory_store():
    """Test memory store operations."""
    print("\nTesting memory store...")
    
    try:
        from memory.store import get_memory_store
        
        memory = get_memory_store()
        
        # Test saving a session
        test_session = {
            'input': 'Test problem',
            'input_type': 'text',
            'topic': 'algebra',
            'solution': 'Test solution',
            'final_answer': '42',
            'success': True
        }
        
        session_id = memory.save_session(test_session)
        if session_id:
            print(f"  [OK] Session saved with ID: {session_id}")
        else:
            print("  [FAIL] Failed to save session")
            return False
        
        # Test retrieving the session
        retrieved = memory.get_session(session_id)
        if retrieved and retrieved.get('input') == 'Test problem':
            print("  [OK] Session retrieved successfully")
        else:
            print("  [FAIL] Failed to retrieve session")
            return False
        
        # Test statistics
        stats = memory.get_statistics()
        if stats['total_sessions'] > 0:
            print(f"  [OK] Statistics: {stats['total_sessions']} sessions")
        else:
            print("  [FAIL] Statistics incorrect")
            return False
        
        return True
    except Exception as e:
        print(f"  [FAIL] Memory store test failed: {e}")
        return False


def test_calculator():
    """Test safe calculator."""
    print("\nTesting calculator...")
    
    try:
        from utils import safe_calculate
        
        # Test basic operations
        result = safe_calculate("2 + 2")
        if result == 4:
            print("  [OK] Addition works")
        else:
            print(f"  [FAIL] Addition failed: {result}")
            return False
        
        result = safe_calculate("10 / 2")
        if result == 5.0:
            print("  [OK] Division works")
        else:
            print(f"  [FAIL] Division failed: {result}")
            return False
        
        result = safe_calculate("2 ** 3")
        if result == 8:
            print("  [OK] Exponentiation works")
        else:
            print(f"  [FAIL] Exponentiation failed: {result}")
            return False
        
        result = safe_calculate("abs(-5)")
        if result == 5:
            print("  [OK] Function calls work")
        else:
            print(f"  [FAIL] Function call failed: {result}")
            return False
        
        return True
    except Exception as e:
        print(f"  [FAIL] Calculator test failed: {e}")
        return False


def test_agent_graph_structure():
    """Test that agent graph can be initialized."""
    print("\nTesting agent graph structure...")
    
    try:
        from graph import get_agent_graph
        
        graph = get_agent_graph()
        print("  [OK] Agent graph initialized")
        
        # Check that graph has the expected nodes
        if hasattr(graph, 'nodes'):
            print("  [OK] Graph has nodes attribute")
        else:
            print("  [FAIL] Graph missing nodes attribute")
            return False
        
        return True
    except Exception as e:
        print(f"  [FAIL] Agent graph test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Math Mentor - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Vector Store", test_vector_store),
        ("Memory Store", test_memory_store),
        ("Calculator", test_calculator),
        ("Agent Graph", test_agent_graph_structure),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
