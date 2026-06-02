"""Token revocation store (logout / forced invalidation).

In-memory by default (single process). For multi-replica production, back this
with Redis keyed by jti with a TTL = token lifetime; the interface stays the same.
See plan §8.1.
"""

from __future__ import annotations

from threading import Lock


class RevocationStore:
    def __init__(self):
        self._lock = Lock()
        self._revoked: set[str] = set()

    def revoke(self, jti: str | None) -> None:
        if not jti:
            return
        with self._lock:
            self._revoked.add(jti)

    def is_revoked(self, jti: str | None) -> bool:
        if not jti:
            return False
        with self._lock:
            return jti in self._revoked


_store: RevocationStore | None = None


def get_revocation_store() -> RevocationStore:
    global _store
    if _store is None:
        _store = RevocationStore()
    return _store
