"""Portable password hashing for demo users."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os


PBKDF2_ITERATIONS = 120_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Return a portable PBKDF2 password hash.

    This avoids demo fragility from native bcrypt wheels while still keeping seed
    credentials out of plaintext storage.
    """
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password hash in constant time."""

    try:
        algorithm, iterations, salt_b64, digest_b64 = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False
