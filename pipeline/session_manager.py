"""Minimal session tracking for pipeline runs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Session:
    session_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Manage pipeline sessions and their artifacts."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    def start(self, metadata: Optional[Dict[str, Any]] = None) -> Session:
        sid = str(uuid.uuid4())
        sess = Session(session_id=sid, metadata=metadata or {})
        self._sessions[sid] = sess
        return sess

    def store_results(self, session_id: str, key: str, value: Any) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].results[key] = value

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)
