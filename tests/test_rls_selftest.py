"""B2 regression: the RLS boot self-test refuses configs where tenant isolation
would silently not hold (superuser/BYPASSRLS role, RLS not forced, or a live
cross-org leak). Driven by a fake psycopg pool so it needs no live Postgres.
"""

from __future__ import annotations

import pytest

from app.core.db.postgres import PostgresDatabase
from app.core.errors import ConfigError

_HEALTHY_TABLES = {
    "chat_sessions": (True, True),
    "chat_messages": (True, True),
    "audit_log": (True, True),
}


class _FakeCursor:
    def __init__(self, one=None, all_=None):
        self._one, self._all = one, all_

    async def fetchone(self):
        return self._one

    async def fetchall(self):
        return self._all or []


class _FakeTx:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False    # never suppress -> the probe's _Rollback propagates out


class _FakeConn:
    def __init__(self, *, role, tables, probe_visible):
        self._role, self._tables, self._probe_visible = role, tables, probe_visible

    async def execute(self, sql, params=None):
        s = sql.lower()
        if "rolsuper" in s:
            return _FakeCursor(one=self._role)
        if "relrowsecurity" in s:
            rows = [(name, en, fo) for name, (en, fo) in self._tables.items()]
            return _FakeCursor(all_=rows)
        if "count(*)" in s:
            return _FakeCursor(one=(self._probe_visible,))
        return _FakeCursor()        # set_config / INSERT

    def transaction(self):
        return _FakeTx()


class _FakeConnCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def connection(self):
        return _FakeConnCtx(self._conn)


def _db(*, role=(False, False), tables=None, probe_visible=0) -> PostgresDatabase:
    db = PostgresDatabase("postgresql://app:pw@h:5432/db")
    db._pool = _FakePool(_FakeConn(
        role=role, tables=tables or _HEALTHY_TABLES, probe_visible=probe_visible))
    return db


@pytest.mark.asyncio
async def test_healthy_config_passes():
    await _db().verify_rls_isolation()   # no raise


@pytest.mark.asyncio
async def test_superuser_role_rejected():
    with pytest.raises(ConfigError, match="SUPERUSER or BYPASSRLS"):
        await _db(role=(True, False)).verify_rls_isolation()


@pytest.mark.asyncio
async def test_bypassrls_role_rejected():
    with pytest.raises(ConfigError, match="SUPERUSER or BYPASSRLS"):
        await _db(role=(False, True)).verify_rls_isolation()


@pytest.mark.asyncio
async def test_rls_not_forced_rejected():
    tables = {**_HEALTHY_TABLES, "chat_messages": (True, False)}   # enabled but NOT forced
    with pytest.raises(ConfigError, match="ENABLED\\+FORCED"):
        await _db(tables=tables).verify_rls_isolation()


@pytest.mark.asyncio
async def test_missing_table_rejected():
    tables = {"chat_sessions": (True, True)}   # other tenant tables absent
    with pytest.raises(ConfigError, match="ENABLED\\+FORCED"):
        await _db(tables=tables).verify_rls_isolation()


@pytest.mark.asyncio
async def test_live_cross_org_leak_rejected():
    # role/relrowsecurity look fine, but the probe row written under org A is
    # visible under org B -> isolation not effective -> refuse.
    with pytest.raises(ConfigError, match="self-test FAILED"):
        await _db(probe_visible=1).verify_rls_isolation()
