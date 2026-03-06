"""
Memory management module for Math Mentor.
Stores and retrieves past problem-solving sessions for context and learning.
"""
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from config import Config


class MemoryStore:
    """
    Persistent memory store for problem-solving sessions.
    Saves interactions to JSON file and provides retrieval capabilities.
    """
    
    def __init__(self, store_file: str = None):
        """
        Initialize the memory store.
        
        Args:
            store_file: Path to the JSON store file
        """
        self.store_file = store_file or Config.MEMORY_STORE_FILE
        self._ensure_store_exists()
    
    def _ensure_store_exists(self):
        """Create the store file if it doesn't exist."""
        # Ensure directory exists
        store_dir = os.path.dirname(self.store_file)
        if store_dir and not os.path.exists(store_dir):
            os.makedirs(store_dir, exist_ok=True)
        
        if not os.path.exists(self.store_file):
            self._write_data([])
    
    def _read_data(self) -> List[Dict[str, Any]]:
        """Read all data from the store."""
        try:
            with open(self.store_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _write_data(self, data: List[Dict[str, Any]]):
        """Write data to the store."""
        with open(self.store_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def save_session(self, session: Dict[str, Any]) -> str:
        """
        Save a problem-solving session to memory.
        
        Args:
            session: Session data containing input, output, and metadata
            
        Returns:
            Session ID
        """
        sessions = self._read_data()
        
        # Create session record
        session_record = {
            'id': f"session_{len(sessions) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'input': session.get('input', ''),
            'input_type': session.get('input_type', 'text'),
            'topic': session.get('topic', 'unknown'),
            'solution': session.get('solution', ''),
            'explanation': session.get('explanation', ''),
            'retrieved_context': session.get('retrieved_context', []),
            'agent_trace': session.get('agent_trace', []),
            'confidence_scores': session.get('confidence_scores', {}),
            'human_feedback': session.get('human_feedback'),
            'user_rating': session.get('user_rating'),
            'success': session.get('success', True)
        }
        
        sessions.append(session_record)
        self._write_data(sessions)
        
        return session_record['id']
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific session by ID.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Session data or None if not found
        """
        sessions = self._read_data()
        for session in sessions:
            if session.get('id') == session_id:
                return session
        return None
    
    def get_similar_problems(self, query: str, topic: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar problems from memory.
        
        Args:
            query: Search query (problem description)
            topic: Optional topic filter
            limit: Maximum number of results
            
        Returns:
            List of similar problem sessions
        """
        sessions = self._read_data()
        
        # Filter by topic if specified
        if topic:
            sessions = [s for s in sessions if s.get('topic') == topic]
        
        # Filter to only successful sessions
        sessions = [s for s in sessions if s.get('success', True)]
        
        # Simple keyword-based similarity
        query_words = set(query.lower().split())
        
        scored_sessions = []
        for session in sessions:
            input_text = session.get('input', '').lower()
            solution_text = session.get('solution', '').lower()
            
            # Count word overlaps
            input_words = set(input_text.split())
            solution_words = set(solution_text.split())
            
            overlap = len(query_words & input_words) + len(query_words & solution_words) * 0.5
            
            if overlap > 0:
                scored_sessions.append((overlap, session))
        
        # Sort by score and return top results
        scored_sessions.sort(reverse=True, key=lambda x: x[0])
        
        return [session for _, session in scored_sessions[:limit]]
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of recent sessions
        """
        sessions = self._read_data()
        # Sort by timestamp descending
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return sessions[:limit]
    
    def get_sessions_by_topic(self, topic: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get sessions for a specific topic.
        
        Args:
            topic: Topic to filter by
            limit: Maximum number of sessions to return
            
        Returns:
            List of sessions for the topic
        """
        sessions = self._read_data()
        topic_sessions = [s for s in sessions if s.get('topic') == topic]
        topic_sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return topic_sessions[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions.
        
        Returns:
            Dictionary with statistics
        """
        sessions = self._read_data()
        
        if not sessions:
            return {
                'total_sessions': 0,
                'topics': {},
                'average_confidence': 0,
                'success_rate': 0
            }
        
        # Count by topic
        topics = {}
        for session in sessions:
            topic = session.get('topic', 'unknown')
            topics[topic] = topics.get(topic, 0) + 1
        
        # Calculate average confidence
        confidences = []
        for session in sessions:
            conf_scores = session.get('confidence_scores', {})
            if conf_scores:
                confidences.extend(conf_scores.values())
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Calculate success rate
        successful = sum(1 for s in sessions if s.get('success', True))
        success_rate = successful / len(sessions) if sessions else 0
        
        return {
            'total_sessions': len(sessions),
            'topics': topics,
            'average_confidence': round(avg_confidence, 3),
            'success_rate': round(success_rate, 3)
        }
    
    def update_session_feedback(self, session_id: str, rating: int, feedback: str = None):
        """
        Update a session with user feedback.
        
        Args:
            session_id: The session ID to update
            rating: User rating (1-5)
            feedback: Optional feedback text
        """
        sessions = self._read_data()
        
        for session in sessions:
            if session.get('id') == session_id:
                session['user_rating'] = rating
                session['user_feedback'] = feedback
                session['feedback_timestamp'] = datetime.now().isoformat()
                break
        
        self._write_data(sessions)
    
    def clear_all(self):
        """Clear all sessions from memory."""
        self._write_data([])
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a specific session.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        sessions = self._read_data()
        
        original_length = len(sessions)
        sessions = [s for s in sessions if s.get('id') != session_id]
        
        if len(sessions) < original_length:
            self._write_data(sessions)
            return True
        
        return False


# Singleton instance
_memory_instance = None


def get_memory_store() -> MemoryStore:
    """Get or create the memory store singleton."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryStore()
    return _memory_instance
