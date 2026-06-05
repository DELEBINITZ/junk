"""PostgreSQL access with Row-Level-Security tenant isolation.

================================ MENTAL MODEL =============================
This module owns two things: a Postgres CONNECTION POOL (reuse a handful of open
connections instead of paying TCP+auth on every query) and — the important part —
the ONE primitive that enforces multi-tenant isolation at the database itself.

How RLS isolation works here, end to end:
  1. Each tenant-scoped table has a Row-Level Security (RLS) POLICY that says, in
     effect, "a row is visible only if its org_id equals the current org setting".
  2. That "current org setting" is a per-transaction GUC (Postgres's name for a
     runtime config variable), here ``app.organization_id``.
  3. ``org_transaction(org_id)`` opens a transaction and SETS that GUC for the
     transaction's lifetime, BEFORE running any of the caller's SQL.
  => Therefore every query inside that transaction is transparently filtered to
     the one tenant. A query that forgot a ``WHERE org_id=...`` STILL cannot see
     another org's rows — the database enforces it, not application code. This is
     defense-in-depth: the strongest place to put tenant isolation.

Critically, ``org_id`` is supplied by the caller from the VERIFIED token
(SecurityContext) — never from a request body — so it can't be spoofed. psycopg
is lazy-imported so the default in-memory store needs no Postgres driver at all.
===========================================================================
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings


class PostgresDatabase:
    """Thin wrapper around an async psycopg connection pool that hands out
    tenant-scoped transactions. One instance is shared process-wide (see the
    module-level singleton below)."""

    def __init__(self, dsn: str, *, rls_setting: str = "app.organization_id",
                 min_size: int = 1, max_size: int = 10) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is required for store_backend=postgres")
        self.dsn = dsn
        self.rls_setting = rls_setting          # the GUC name the RLS policies read
        self._min, self._max = min_size, max_size   # pool sizing bounds
        self._pool = None                       # created lazily on first use (see open())

    async def open(self) -> None:
        """Create and open the connection pool once (idempotent). Called at startup
        or lazily on the first transaction. ``open=False`` then ``open(wait=True)``
        ensures the pool is fully warmed (connections established) before we use it."""
        if self._pool is not None:
            return
        from psycopg_pool import AsyncConnectionPool  # lazy import keeps the driver optional

        self._pool = AsyncConnectionPool(
            conninfo=self.dsn, min_size=self._min, max_size=self._max, open=False
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        """Drain and close the pool on shutdown so connections are released cleanly."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def org_transaction(self, org_id: str) -> AsyncIterator:
        """Transaction scoped to one tenant. Sets the RLS GUC for its lifetime.

        This is the isolation primitive every PG-backed store call goes through.
        ``set_config(name, value, true)`` sets the GUC with ``is_local=true`` so it
        applies only to THIS transaction (and is rolled back when the transaction
        ends) — there's no risk of it leaking to the next caller that borrows the
        same pooled connection. The caller then runs its SQL against the yielded
        connection, fully scoped to ``org_id`` by the RLS policies."""
        if not org_id:                           # refuse to run un-scoped: no org => no transaction
            raise ValueError("org_transaction requires org_id")
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:    # borrow a pooled connection
            async with conn.transaction():              # BEGIN ... COMMIT/ROLLBACK
                await conn.execute(
                    "SELECT set_config(%s, %s, true)", (self.rls_setting, str(org_id))
                )                                       # arm RLS for this transaction
                yield conn

    async def verify_rls_isolation(self) -> None:
        """Boot-time PROOF that tenant isolation actually holds at the database.

        ``FORCE ROW LEVEL SECURITY`` does NOT apply to a Postgres SUPERUSER or a role
        with the BYPASSRLS attribute — and the store SELECTs deliberately omit
        ``WHERE org_id`` (they rely entirely on RLS). So pointing ``DATABASE_URL`` at
        such a role silently returns EVERY tenant's rows with no error and no log.
        This self-test turns that misconfiguration into a loud failure. Three checks:

          1. the connecting role is neither SUPERUSER nor BYPASSRLS;
          2. RLS is ENABLED and FORCED on each tenant table (migrations actually ran);
          3. a LIVE probe: a row written under org 'A' is INVISIBLE under org 'B' —
             run inside a transaction that is ALWAYS rolled back, so the probe row
             never persists.

        Raises ``ConfigError`` on any failure. The caller decides whether that is
        fatal (prod: refuse to boot) or a warning (dev)."""
        import uuid

        from app.core.errors import ConfigError

        if self._pool is None:
            await self.open()
        tenant_tables = ("chat_sessions", "chat_messages", "audit_log")

        async with self._pool.connection() as conn:
            # 1. The connecting role must not bypass RLS.
            cur = await conn.execute(
                "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
            )
            role = await cur.fetchone()
            if role and (role[0] or role[1]):
                raise ConfigError(
                    "DATABASE_URL connects as a SUPERUSER or BYPASSRLS role — Row-Level "
                    "Security (tenant isolation) does NOT apply to such roles, so every "
                    "tenant's rows would be returned. Create a dedicated app role with "
                    "NOSUPERUSER NOBYPASSRLS and connect as that (see PRODUCTION_SETUP.txt §2.1)."
                )
            # 2. RLS enabled + forced on every tenant table.
            cur = await conn.execute(
                "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relname = ANY(%s)", (list(tenant_tables),)
            )
            state = {r[0]: (r[1], r[2]) for r in await cur.fetchall()}
            for t in tenant_tables:
                enabled, forced = state.get(t, (False, False))
                if not (enabled and forced):
                    raise ConfigError(
                        f"Row-Level Security is not ENABLED+FORCED on '{t}'. Run the "
                        f"migrations (asi-migrate) before serving — tenant isolation "
                        f"depends on it."
                    )

        # 3. Live cross-org probe, always rolled back so nothing persists.
        probe_id = f"__rls_probe_{uuid.uuid4().hex}"
        org_a, org_b = "__rls_probe_org_a", "__rls_probe_org_b"
        leaked = False

        class _Rollback(Exception):
            pass

        async with self._pool.connection() as conn:
            try:
                async with conn.transaction():
                    await conn.execute("SELECT set_config(%s, %s, true)", (self.rls_setting, org_a))
                    await conn.execute(
                        "INSERT INTO chat_sessions (id, org_id, user_id) VALUES (%s, %s, %s)",
                        (probe_id, org_a, "__rls_probe_user"),
                    )
                    # Switch tenant WITHIN the same transaction; org B must not see A's row.
                    await conn.execute("SELECT set_config(%s, %s, true)", (self.rls_setting, org_b))
                    cur = await conn.execute(
                        "SELECT count(*) FROM chat_sessions WHERE id=%s", (probe_id,)
                    )
                    (visible,) = await cur.fetchone()
                    leaked = bool(visible)
                    raise _Rollback()      # discard the probe row no matter what
            except _Rollback:
                pass
        if leaked:
            raise ConfigError(
                "RLS self-test FAILED: a row written under one org was visible under "
                "another — tenant isolation is NOT effective. Refusing to trust it. Check "
                "the DB role (no SUPERUSER/BYPASSRLS) and that the org_isolation policies "
                "from the migrations are present."
            )

    @asynccontextmanager
    async def privileged_transaction(self) -> AsyncIterator:
        """Cross-tenant transaction for admin/audit reads. Use sparingly; the
        connecting role should still be RLS-bound unless it is a trusted admin.

        Deliberately does NOT set the org GUC, so it is NOT scoped to one tenant —
        that is its whole purpose (e.g. a platform-wide audit). Because it bypasses
        the per-tenant filter, it must only ever be used for genuinely cross-tenant
        admin work, never on the user-facing chat path."""
        if self._pool is None:
            await self.open()
        async with self._pool.connection() as conn:
            async with conn.transaction():
                yield conn


# Module-level singleton: one shared pool per process. Building a second pool would
# waste connections, so get_database memoizes the instance here.
_db: PostgresDatabase | None = None


def get_database(settings: Settings) -> PostgresDatabase:
    """Return the process-wide PostgresDatabase, constructing it (with pool sizing
    and the RLS GUC name from settings) on first call. Subsequent calls reuse it."""
    global _db
    if _db is None:
        _db = PostgresDatabase(
            settings.database_url, rls_setting=settings.rls_setting_name,
            min_size=settings.db_pool_min, max_size=settings.db_pool_max,
        )
    return _db


def reset_database() -> None:
    """Drop the singleton so the next get_database builds a fresh one. Used by
    tests (and reconfiguration) to avoid leaking a pool across instances."""
    global _db
    _db = None


__all__ = ["PostgresDatabase", "get_database", "reset_database"]
