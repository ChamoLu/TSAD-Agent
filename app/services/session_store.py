from __future__ import annotations

from threading import Lock
from typing import Dict, List, Optional


class SessionStore:
    def __init__(self):
        self._lock = Lock()
        self._results: Dict[str, dict] = {}
        self._histories: Dict[str, List[dict]] = {}

    def save_result(self, record: dict) -> dict:
        with self._lock:
            self._results[record['id']] = record
            self._histories.setdefault(record['id'], [])
        return record

    def get_result(self, result_id: str) -> Optional[dict]:
        with self._lock:
            return self._results.get(result_id)

    def append_message(self, result_id: str, role: str, content: str) -> None:
        with self._lock:
            self._histories.setdefault(result_id, []).append({
                'role': role,
                'content': content,
            })

    def get_history(self, result_id: str, limit: int = 8) -> List[dict]:
        with self._lock:
            return list(self._histories.get(result_id, [])[-limit:])
