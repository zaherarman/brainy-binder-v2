from datetime import datetime, timedelta, timezone
from typing import Any

import threading

class ConversationManager:
    """Thread-safe in-memory conversation history manager."""

    def __init__(self, session_timeout_minutes: int = 60, cache_timeout_minutes: int = 30) -> None:
        self.conversations: dict[str, dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.cache_timeout = timedelta(minutes=cache_timeout_minutes)
        self.lock = threading.Lock()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """
        Get conversation history for a session."""
        
        with self.lock:
            if session_id in self.conversations:
                self.conversations[session_id]["last_activity"] = datetime.now(timezone.utc)
                return self.conversations[session_id]["messages"].copy()
        return []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """
        Add a message to the conversation history."""
        
        message = {"role": role, "content": content}
        
        with self.lock:
            if session_id not in self.conversations:
                self.conversations[session_id] = {
                    "messages": [],
                    "last_activity": datetime.now(timezone.utc),
                }
                
            self.conversations[session_id]["messages"].append(message)
            self.conversations[session_id]["last_activity"] = datetime.now(timezone.utc)

    def clear_session(self, session_id: str) -> None:
        """
        Remove conversation history for a session."""
        
        with self.lock:
            if session_id in self.conversations:
                del self.conversations[session_id]

    def get_session_count(self) -> int:
        """Return the number of active sessions in cache."""
        
        with self.lock:
            return len(self.conversations)

    def cleanup_old_sessions(self) -> int:
        """Evict sessions older than cache_timeout from memory."""
        
        with self.lock:
            now = datetime.now(timezone.utc)
            to_remove = [
                sid
                for sid, data in self.conversations.items()
                if now - data["last_activity"] > self.cache_timeout
            ]
            
            for sid in to_remove:
                del self.conversations[sid]
                
            return len(to_remove)

conversation_manager = ConversationManager(
    session_timeout_minutes=60,
    cache_timeout_minutes=30,
)
