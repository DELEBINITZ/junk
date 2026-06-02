"""Password hashing for the local auth provider (dev / on-prem without OIDC).

================================ WHY HASH AT ALL ==========================
We must NEVER store passwords as plain text — a database leak would then hand an
attacker every user's actual password. Instead we store a one-way HASH: a value
you can verify a password against but cannot reverse back into the password.

bcrypt is a deliberately SLOW, SALTED password hash:
  * SALTED — a random salt is mixed in and stored alongside the hash, so two
    users with the same password get different hashes and precomputed "rainbow
    table" attacks don't work. (``gensalt()`` generates that per-password salt.)
  * SLOW   — bcrypt is intentionally expensive to compute, which throttles
    brute-force guessing. (Fast hashes like plain SHA-256 are the wrong tool
    for passwords precisely because they're cheap to guess against.)

Uses ``bcrypt`` directly (passlib's bcrypt backend has a version-probe bug with
bcrypt 4.x). bcrypt's 72-byte input limit is handled by byte-truncation.
===========================================================================
"""

from __future__ import annotations

import bcrypt


def _b(plain: str) -> bytes:
    # bcrypt only considers the FIRST 72 BYTES of input — bytes beyond that are
    # ignored by the algorithm. We truncate explicitly (after UTF-8 encoding, so
    # the boundary is byte-correct for multibyte characters) so hash and verify
    # always feed bcrypt the identical input and agree.
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    # Generate a fresh salt and produce the salted hash. The returned string
    # embeds the salt + cost, so verify_password needs nothing else to check it.
    return bcrypt.hashpw(_b(plain), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """True iff ``plain`` matches the stored ``hashed``. bcrypt re-derives the
    hash using the salt embedded in ``hashed`` and compares — we never decrypt
    anything (there is nothing to decrypt; the hash is one-way)."""
    try:
        return bcrypt.checkpw(_b(plain), hashed.encode("utf-8"))
    except Exception:
        # A malformed/garbage stored hash must read as "wrong password", never as
        # a crash that could be probed or that 500s the login path.
        return False


__all__ = ["hash_password", "verify_password"]
