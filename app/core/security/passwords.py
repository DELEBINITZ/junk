"""Password hashing for the local auth provider (dev / on-prem without OIDC).

Uses ``bcrypt`` directly (passlib's bcrypt backend has a version-probe bug with
bcrypt 4.x). bcrypt's 72-byte input limit is handled by byte-truncation.
"""

from __future__ import annotations

import bcrypt


def _b(plain: str) -> bytes:
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_b(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_b(plain), hashed.encode("utf-8"))
    except Exception:
        return False


__all__ = ["hash_password", "verify_password"]
