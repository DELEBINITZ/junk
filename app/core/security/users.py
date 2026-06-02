"""User directory for the local auth provider.

This is the "who can log in" table for LOCAL auth — the source the login endpoint
checks an email+password against before minting a JWT (see jwt.py). It exists so
the platform has a working identity story with zero external infrastructure.

In ``local`` mode the platform authenticates against this store. In ``oidc`` mode
it is unused (identity comes from the IdP). The default store is seeded with two
orgs so multi-tenant isolation is exercisable out of the box; override by
implementing :class:`UserStore` against your own user table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.core.security.passwords import hash_password


@dataclass
class User:
    """One user record. Crucially the user is BOUND to an ``org_id`` and a set of
    ``roles`` HERE — these become the JWT's ``org_id``/``roles`` claims at login,
    which is the origin of the trusted tenant+role identity the rest of the system
    relies on. We store only the password HASH, never the password."""
    user_id: str
    email: str
    org_id: str                  # the tenant this user belongs to
    roles: tuple[str, ...]       # the user's RBAC roles (viewer/analyst/admin)
    password_hash: str           # bcrypt hash (see passwords.py); never plaintext
    display_name: str = ""
    disabled: bool = False       # a disabled user must not be allowed to authenticate


class UserStore(Protocol):
    """The lookup contract the auth layer needs: by email (for login) and by id
    (e.g. when refreshing a token to re-fetch current roles). A Protocol so a real
    deployment can back it with a database without changing callers."""
    def get_by_email(self, email: str) -> User | None: ...
    def get_by_id(self, user_id: str) -> User | None: ...


@dataclass
class InMemoryUserStore:
    """Dict-backed store (two indexes for the two lookups). Fine for dev/tests;
    a production deployment swaps in a DB-backed UserStore."""
    _by_email: dict[str, User] = field(default_factory=dict)
    _by_id: dict[str, User] = field(default_factory=dict)

    def add(self, user: User) -> None:
        # Index by lowercased email so logins are case-insensitive, and by id.
        self._by_email[user.email.lower()] = user
        self._by_id[user.user_id] = user

    def get_by_email(self, email: str) -> User | None:
        return self._by_email.get(email.lower())     # case-insensitive match

    def get_by_id(self, user_id: str) -> User | None:
        return self._by_id.get(user_id)


# Demo seed — DEV ONLY. TWO orgs (acme, globex), each with the three role levels,
# so you can actually exercise both RBAC (viewer/analyst/admin) AND tenant
# isolation (acme must never see globex's data) right out of the box. These are
# fake accounts with a shared throwaway password; never use them in production.
_SEED = [
    ("u-alice", "alice@acme.test", "org_acme", ("admin",), "Alice (Acme admin)"),
    ("u-bob", "bob@acme.test", "org_acme", ("analyst",), "Bob (Acme analyst)"),
    ("u-carol", "carol@acme.test", "org_acme", ("viewer",), "Carol (Acme viewer)"),
    ("u-dave", "dave@globex.test", "org_globex", ("admin",), "Dave (Globex admin)"),
    ("u-erin", "erin@globex.test", "org_globex", ("analyst",), "Erin (Globex analyst)"),
]
DEFAULT_DEV_PASSWORD = "password"  # noqa: S105 - dev seed only


def build_default_user_store(password: str = DEFAULT_DEV_PASSWORD) -> InMemoryUserStore:
    """Construct the seeded dev store. The password is HASHED once and shared by
    all seed users — convenient for demos, deliberately useless for real auth."""
    store = InMemoryUserStore()
    pw = hash_password(password)                     # hash once; store only the hash
    for uid, email, org, roles, name in _SEED:
        store.add(User(user_id=uid, email=email, org_id=org, roles=roles,
                       password_hash=pw, display_name=name))
    return store


__all__ = ["User", "UserStore", "InMemoryUserStore", "build_default_user_store",
           "DEFAULT_DEV_PASSWORD"]
