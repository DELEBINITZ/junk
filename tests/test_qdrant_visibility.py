"""B3 regression: shared-corpus visibility is configurable and explicit.

Default (shared_untagged_public=True, the public-knowledge-base model): a shared
doc is visible to an org if it is tagged for the org, explicitly public, OR has no
customer_tags at all (untagged => public to everyone). Strict mode
(shared_untagged_public=False, for corpora that mix customer-scoped data): the
untagged-is-public clause is dropped, so a missing tag cannot leak.

qdrant-client isn't installed in the test env, so we stub it (the store lazy-imports
it) and assert the filter STRUCTURE directly.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest


def _install_fake_qdrant():
    """Minimal qdrant_client stub: model constructors that just capture kwargs, plus
    no-op clients, so the store builds and we can introspect its filter."""
    class _Rec:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    models = types.SimpleNamespace(
        Filter=_Rec, FieldCondition=_Rec, MatchValue=_Rec, MatchAny=_Rec,
        DatetimeRange=_Rec, IsEmptyCondition=_Rec, PayloadField=_Rec,
    )
    mod = types.ModuleType("qdrant_client")
    mod.models = models
    mod.QdrantClient = type("QdrantClient", (), {"__init__": lambda self, *a, **k: None})
    mod.AsyncQdrantClient = type("AsyncQdrantClient", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["qdrant_client"] = mod


def _store(*, untagged_public: bool):
    _install_fake_qdrant()
    from app.core.rag.qdrant_backend import QdrantVectorStore

    return QdrantVectorStore("http://x:6333", shared_untagged_public=untagged_public)


def test_shared_default_untagged_is_public():
    # DEFAULT policy: tagged-for-org OR public OR untagged(IsEmpty) are all visible.
    should = _store(untagged_public=True)._filter("org_acme", None, "shared").must[0].should
    match_keys = {c.key for c in should if hasattr(c, "match")}
    assert match_keys == {"customer_tags", "public"}
    assert any(hasattr(c, "is_empty") for c in should)     # untagged => public clause present
    pub = next(c for c in should if getattr(c, "key", None) == "public")
    assert pub.match.value is True


def test_shared_strict_mode_drops_untagged_public():
    # STRICT policy: NO untagged-is-public clause -> a missing tag is invisible to all.
    should = _store(untagged_public=False)._filter("org_acme", None, "shared").must[0].should
    assert not any(hasattr(c, "is_empty") for c in should)
    match_keys = {c.key for c in should if hasattr(c, "match")}
    assert match_keys == {"customer_tags", "public"}


def test_tenant_filter_still_hard_scopes_org_id():
    flt = _store(untagged_public=True)._filter("org1", None, "tenant")
    cond = flt.must[0]
    assert cond.key == "org_id" and cond.match.value == "org1"


def test_cron_is_public_policy():
    # The ingest-side rule that stamps the explicit `public` flag (TLP-driven).
    _install_fake_qdrant()
    sys.modules.setdefault("requests", types.ModuleType("requests"))
    sys.path.insert(0, "scripts")
    sys.modules.pop("index_reports_cron", None)
    cron = importlib.import_module("index_reports_cron")

    assert cron._is_public({"public": True}) is True
    assert cron._is_public({"tlp": "CLEAR"}) is True
    assert cron._is_public({"tlp": "TLP:WHITE"}) is True
    assert cron._is_public({"tlp": "AMBER"}) is False
    assert cron._is_public({}) is False
