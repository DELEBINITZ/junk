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


_DEV_API_KEY = "dev-api-key-change-me"     # the default gateway key (config.api_keys)

# Seed identities (formerly the dev user store). Auth now = API key + a JWT we mint
# directly here; these map a test email to the (user_id, org_id, roles) the JWT carries.
_SEED_IDENTITIES = {
    "alice@acme.test": ("u-alice", "org_acme", ("admin",)),
    "bob@acme.test": ("u-bob", "org_acme", ("analyst",)),
    "carol@acme.test": ("u-carol", "org_acme", ("viewer",)),
    "dave@globex.test": ("u-dave", "org_globex", ("admin",)),
    "erin@globex.test": ("u-erin", "org_globex", ("analyst",)),
}


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    # The gateway API key rides on EVERY request by default; individual tests add the
    # ``Authorization: Bearer <jwt>`` header via login() below.
    with TestClient(app, headers={"x-api-key": _DEV_API_KEY}) as c:
        yield c


def login(client, email="alice@acme.test", password="password") -> str:
    """Mint a valid access JWT for a seed identity. There is no login endpoint
    anymore — auth is API key (sent by the client fixture) + a JWT verified per
    request — so tests mint the JWT directly with the same secret the app verifies."""
    from app.config import get_settings
    from app.core.security.jwt import create_access_token

    user_id, org_id, roles = _SEED_IDENTITIES.get(email, _SEED_IDENTITIES["alice@acme.test"])
    return create_access_token(get_settings(), sub=user_id, org_id=org_id,
                               roles=roles, email=email).token
