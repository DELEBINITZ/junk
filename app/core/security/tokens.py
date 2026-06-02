"""Token revocation (logout, rotation).

================================ THE PROBLEM ==============================
JWTs are *stateless*: once signed they are valid until they expire, and the
server doesn't "remember" them. That is great for scale but creates one gap —
what happens when a user logs out, or we rotate a refresh token, BEFORE it
naturally expires? The signature is still valid, so the old token would keep
working. A REVOCATION LIST closes that gap: a deny-list of ``jti`` (token ids)
that must be rejected even though their signatures verify.

The auth dependency (deps.py) checks every access token's ``jti`` against this
store and rejects revoked ones. Each revoked entry only needs to live until the
token would have expired anyway (after ``exp`` the signature fails on its own),
so entries are pruned/TTL'd at that point — the list never grows unbounded.

In-memory by default (fine for a single process); Redis-backed when
``cache_backend=redis`` so a logout on one replica is honored by ALL replicas
(a deny-list only works if every server sees the same list).
===========================================================================
"""

from __future__ import annotations

import threading
import time
from typing import Protocol


class RevocationStore(Protocol):
    """The pluggable contract: record a revoked token id, and answer "is this
    token id revoked?". A Protocol (duck-typed) so the in-memory and Redis
    implementations are interchangeable without a shared base class."""
    def revoke(self, jti: str, expires_at: int) -> None: ...
    def is_revoked(self, jti: str) -> bool: ...


class InMemoryRevocationStore:
    """Single-process deny-list: ``jti -> expiry``. Good for dev / one replica.
    Does NOT survive a restart or share across processes — use Redis for that."""

    def __init__(self) -> None:
        self._revoked: dict[str, int] = {}     # jti -> the token's own exp time
        # A lock because FastAPI serves requests on multiple threads and a plain
        # dict mutated concurrently could corrupt or race.
        self._lock = threading.Lock()

    def revoke(self, jti: str, expires_at: int) -> None:
        # Remember this token id as revoked until it would have expired anyway,
        # then opportunistically drop any already-expired entries.
        with self._lock:
            self._revoked[jti] = expires_at
            self._prune()

    def is_revoked(self, jti: str) -> bool:
        with self._lock:
            exp = self._revoked.get(jti)
            if exp is None:
                return False                   # never revoked
            if exp < int(time.time()):
                # Past its expiry: the signature check already rejects it, so we
                # no longer need to track it — clean up and treat as not-revoked.
                self._revoked.pop(jti, None)
                return False
            return True                        # actively revoked and still within its lifetime

    def _prune(self) -> None:
        # Drop entries whose tokens have expired; keeps the deny-list bounded.
        now = int(time.time())
        for k in [k for k, v in self._revoked.items() if v < now]:
            self._revoked.pop(k, None)


class RedisRevocationStore:
    """Lazy Redis-backed revocation; keys auto-expire via TTL.

    SHARED across replicas — this is the production choice. Because the deny-list
    lives in Redis (not process memory), a logout handled by one app instance is
    seen by every instance. Redis' own key TTL does the pruning for us."""

    def __init__(self, redis_url: str) -> None:
        import redis  # lazy

        self._r = redis.Redis.from_url(redis_url)

    def revoke(self, jti: str, expires_at: int) -> None:
        # Store the revoked id with a TTL equal to the token's remaining lifetime
        # (at least 1s). After that, Redis evicts the key automatically — once the
        # token is expired the signature check rejects it, so we needn't remember.
        ttl = max(1, expires_at - int(time.time()))
        self._r.setex(f"revoked:{jti}", ttl, "1")

    def is_revoked(self, jti: str) -> bool:
        # Presence of the key == revoked. (Absent => either never revoked, or its
        # TTL lapsed, which means the token is also expired.)
        return bool(self._r.exists(f"revoked:{jti}"))


# Process-wide fallback store, shared so revocations aren't split across copies.
_default: RevocationStore | None = None


def get_default_revocation_store() -> RevocationStore:
    """The in-memory singleton, used when no Redis is configured."""
    global _default
    if _default is None:
        _default = InMemoryRevocationStore()
    return _default


def build_revocation_store(settings) -> RevocationStore:
    """Pick the store from config: Redis when a cache backend + URL are set (so
    revocation is cluster-wide), otherwise the in-memory default."""
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
