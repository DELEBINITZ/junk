"""Token revocation (logout, rotation). In-memory by default; Redis-backed when
``cache_backend=redis`` so revocation survives across replicas."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class RevocationStore(Protocol):
    def revoke(self, jti: str, expires_at: int) -> None: ...
    def is_revoked(self, jti: str) -> bool: ...


class InMemoryRevocationStore:
    def __init__(self) -> None:
        self._revoked: dict[str, int] = {}
        self._lock = threading.Lock()

    def revoke(self, jti: str, expires_at: int) -> None:
        with self._lock:
            self._revoked[jti] = expires_at
            self._prune()

    def is_revoked(self, jti: str) -> bool:
        with self._lock:
            exp = self._revoked.get(jti)
            if exp is None:
                return False
            if exp < int(time.time()):
                self._revoked.pop(jti, None)
                return False
            return True

    def _prune(self) -> None:
        now = int(time.time())
        for k in [k for k, v in self._revoked.items() if v < now]:
            self._revoked.pop(k, None)


class RedisRevocationStore:
    """Lazy Redis-backed revocation; keys auto-expire via TTL."""

    def __init__(self, redis_url: str) -> None:
        import redis  # lazy

        self._r = redis.Redis.from_url(redis_url)

    def revoke(self, jti: str, expires_at: int) -> None:
        ttl = max(1, expires_at - int(time.time()))
        self._r.setex(f"revoked:{jti}", ttl, "1")

    def is_revoked(self, jti: str) -> bool:
        return bool(self._r.exists(f"revoked:{jti}"))


_default: RevocationStore | None = None


def get_default_revocation_store() -> RevocationStore:
    global _default
    if _default is None:
        _default = InMemoryRevocationStore()
    return _default


def build_revocation_store(settings) -> RevocationStore:
    if settings.cache_backend == "redis" and settings.redis_url:
        return RedisRevocationStore(settings.redis_url)
    return get_default_revocation_store()


__all__ = [
    "RevocationStore",
    "InMemoryRevocationStore",
    "RedisRevocationStore",
    "get_default_revocation_store",
    "build_revocation_store",
]
