"""Shared test fixtures. All on the zero-infra deterministic path."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

# enable every module so contract/routing tests cover all of them
os.environ.setdefault("CAP_BRAND_ENABLED", "true")
os.environ.setdefault("CAP_ACI_ENABLED", "true")

from app.config import reload_settings  # noqa: E402
from app.core.bootstrap import build_services, seed_demo  # noqa: E402
from app.core.contracts import ToolContext  # noqa: E402
from app.core.security.context import SecurityContext  # noqa: E402


def make_sc(org="org_acme", user="u-alice", roles=("analyst",)) -> SecurityContext:
    return SecurityContext(org_id=org, user_id=user, roles=roles, email=f"{user}@test")


@pytest_asyncio.fixture
async def services():
    svc = build_services(reload_settings())
    await seed_demo(svc)
    yield svc
    await svc.aclose()


@pytest.fixture
def acme() -> SecurityContext:
    return make_sc()


@pytest.fixture
def globex() -> SecurityContext:
    return make_sc(org="org_globex", user="u-dave")


def tool_ctx(services, sc: SecurityContext) -> ToolContext:
    return ToolContext(org_id=sc.org_id, user_id=sc.user_id, roles=sc.roles,
                       trace_id="t", request_id="r", deps=services.deps)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c


def login(client, email="alice@acme.test", password="password") -> str:
    r = client.post("/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]
