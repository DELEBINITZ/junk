"""User directory for the local auth provider.

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
    user_id: str
    email: str
    org_id: str
    roles: tuple[str, ...]
    password_hash: str
    display_name: str = ""
    disabled: bool = False


class UserStore(Protocol):
    def get_by_email(self, email: str) -> User | None: ...
    def get_by_id(self, user_id: str) -> User | None: ...


@dataclass
class InMemoryUserStore:
    _by_email: dict[str, User] = field(default_factory=dict)
    _by_id: dict[str, User] = field(default_factory=dict)

    def add(self, user: User) -> None:
        self._by_email[user.email.lower()] = user
        self._by_id[user.user_id] = user

    def get_by_email(self, email: str) -> User | None:
        return self._by_email.get(email.lower())

    def get_by_id(self, user_id: str) -> User | None:
        return self._by_id.get(user_id)


# Demo seed — DEV ONLY. Two orgs, distinct roles. Default password: "password".
_SEED = [
    ("u-alice", "alice@acme.test", "org_acme", ("admin",), "Alice (Acme admin)"),
    ("u-bob", "bob@acme.test", "org_acme", ("analyst",), "Bob (Acme analyst)"),
    ("u-carol", "carol@acme.test", "org_acme", ("viewer",), "Carol (Acme viewer)"),
    ("u-dave", "dave@globex.test", "org_globex", ("admin",), "Dave (Globex admin)"),
    ("u-erin", "erin@globex.test", "org_globex", ("analyst",), "Erin (Globex analyst)"),
]
DEFAULT_DEV_PASSWORD = "password"  # noqa: S105 - dev seed only


def build_default_user_store(password: str = DEFAULT_DEV_PASSWORD) -> InMemoryUserStore:
    store = InMemoryUserStore()
    pw = hash_password(password)
    for uid, email, org, roles, name in _SEED:
        store.add(User(user_id=uid, email=email, org_id=org, roles=roles,
                       password_hash=pw, display_name=name))
    return store


__all__ = ["User", "UserStore", "InMemoryUserStore", "build_default_user_store",
           "DEFAULT_DEV_PASSWORD"]
