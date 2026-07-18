"""Tenant-isolation regression tests for the Qdrant access filter.

WHY THIS EXISTS: the access filter (`_build_access_filter`) is the single control
that stops org A from retrieving org B's private data. For a multitenant security
product a cross-tenant leak is existential, so this invariant must be pinned by a
test that runs in CI with NO external services.

Two layers:
  1. Structural — assert the filter is built as `must(is_deleted=false)` +
     `min_should(min_count=1)` over {customer_tags∋org, public}. This locks the
     hardening in place (a regression back to a bare `should` fails here).
  2. Semantic — a small pure-Python matcher mirrors Qdrant's must/min_should
     semantics and PROVES, against the real Filter object, that a private point
     tagged for another org is never visible. Deterministic, no Qdrant needed.

A live-Qdrant integration test (auto-skipped when no reachable QDRANT_URL) proves
the same invariant end-to-end against a real server.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny  # noqa: E402

from security_intel.tools.qdrant_search import _build_access_filter  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure-Python evaluator of a Qdrant Filter for the simple conditions we use.
# Mirrors Qdrant semantics: ALL `must` match, NONE of `must_not`, at least one
# `should` (if present), and at least `min_should.min_count` of its conditions.
# --------------------------------------------------------------------------- #
def _cond_matches(cond: FieldCondition, payload: dict) -> bool:
    val = payload.get(cond.key)
    m = cond.match
    if isinstance(m, MatchValue):
        return val == m.value
    if isinstance(m, MatchAny):
        if isinstance(val, list):
            return any(v in m.any for v in val)
        return val in m.any
    raise AssertionError(f"unhandled match type in test evaluator: {type(m)}")


def _matches(filt: Filter, payload: dict) -> bool:
    if filt.must:
        if not all(_cond_matches(c, payload) for c in filt.must):
            return False
    if filt.must_not:
        if any(_cond_matches(c, payload) for c in filt.must_not):
            return False
    if filt.should:
        if not any(_cond_matches(c, payload) for c in filt.should):
            return False
    if filt.min_should:
        matched = sum(1 for c in filt.min_should.conditions if _cond_matches(c, payload))
        if matched < filt.min_should.min_count:
            return False
    return True


# Sample corpus points, one per visibility class.
def _pt(tags, public, deleted=False):
    return {"customer_tags": tags, "public": public, "is_deleted": deleted}


A_PRIVATE = _pt(["orgA"], False)                 # A's private doc
B_PRIVATE = _pt(["orgB"], False)                 # B's private doc
PUBLIC = _pt([], True)                            # shared/public doc
SHARED_AB = _pt(["orgA", "orgB"], False)          # doc scoped to both
DELETED_A = _pt(["orgA"], False, deleted=True)    # A's doc, soft-deleted
UNTAGGED_PRIVATE = _pt([], False)                 # no tags, not public


# ------------------------------ structural ------------------------------ #
def test_filter_uses_min_should_not_bare_should():
    """The org/public OR-group MUST be a hard requirement (min_should, min_count=1),
    never a bare `should` that a Qdrant version could treat as optional."""
    f = _build_access_filter("orgA")
    assert f.min_should is not None, "access filter must use min_should"
    assert f.min_should.min_count == 1
    assert len(f.min_should.conditions) == 2
    # A bare top-level `should` would be the leak-prone form — must not be used.
    assert not f.should, "access filter must not rely on a bare top-level `should`"
    # is_deleted=false is always required.
    assert any(
        isinstance(c, FieldCondition) and c.key == "is_deleted"
        and isinstance(c.match, MatchValue) and c.match.value is False
        for c in (f.must or [])
    ), "is_deleted=false must be a `must` condition"


def test_extra_must_is_appended():
    extra = FieldCondition(key="published_at", match=MatchValue(value="2026-01-01"))
    f = _build_access_filter("orgA", extra_must=[extra])
    assert extra in (f.must or [])
    assert f.min_should is not None  # hardening preserved with extra_must


# ------------------------------ semantic (isolation) ------------------------------ #
def test_org_cannot_see_another_orgs_private_point():
    """THE isolation invariant: orgA's filter never matches orgB's private point."""
    fa = _build_access_filter("orgA")
    fb = _build_access_filter("orgB")
    assert _matches(fa, A_PRIVATE) is True
    assert _matches(fa, B_PRIVATE) is False   # <-- no cross-tenant leak
    assert _matches(fb, A_PRIVATE) is False   # <-- and the reverse
    assert _matches(fb, B_PRIVATE) is True


def test_public_visible_to_all_orgs():
    assert _matches(_build_access_filter("orgA"), PUBLIC) is True
    assert _matches(_build_access_filter("orgZ"), PUBLIC) is True


def test_shared_point_visible_only_to_tagged_orgs():
    assert _matches(_build_access_filter("orgA"), SHARED_AB) is True
    assert _matches(_build_access_filter("orgB"), SHARED_AB) is True
    assert _matches(_build_access_filter("orgC"), SHARED_AB) is False


def test_deleted_point_never_visible():
    assert _matches(_build_access_filter("orgA"), DELETED_A) is False


def test_untagged_non_public_point_is_excluded():
    """Documents current STRICT behavior: a point with empty customer_tags AND
    public=false is visible to nobody. (The module docstring's "empty tags = visible
    to all" intent is only realized when such docs are also marked public=true — flag
    for the ingestion contract, but strict-deny is the safe default for isolation.)"""
    assert _matches(_build_access_filter("orgA"), UNTAGGED_PRIVATE) is False


# ------------------------------ live integration (guarded) ------------------------------ #
async def test_live_qdrant_isolation():
    """End-to-end proof against a real Qdrant. Auto-skips unless QDRANT_URL is set
    and reachable — seeds org A + org B private points and asserts a scroll with
    org A's access filter never returns org B's point. No TEI/embeddings needed."""
    url = os.getenv("QDRANT_URL")
    if not url:
        pytest.skip("QDRANT_URL not set — skipping live isolation test")

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except Exception as e:  # pragma: no cover
        pytest.skip(f"qdrant client unavailable: {e}")

    client = AsyncQdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=10)
    coll = "test_tenant_isolation_tmp"
    dim = 8
    try:
        if await client.collection_exists(coll):
            await client.delete_collection(coll)
        await client.create_collection(coll, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        await client.upsert(coll, points=[
            PointStruct(id=1, vector=[0.1] * dim,
                        payload={"customer_tags": ["orgA"], "public": False, "is_deleted": False, "text": "A-private"}),
            PointStruct(id=2, vector=[0.1] * dim,
                        payload={"customer_tags": ["orgB"], "public": False, "is_deleted": False, "text": "B-private"}),
        ])
        points, _ = await client.scroll(coll, scroll_filter=_build_access_filter("orgA"),
                                        limit=10, with_payload=True)
        texts = {(p.payload or {}).get("text") for p in points}
        assert "A-private" in texts
        assert "B-private" not in texts, "CROSS-TENANT LEAK: orgA saw orgB's private point"
    except AssertionError:
        raise
    except Exception as e:
        pytest.skip(f"Qdrant unreachable/failed ({e}) — skipping live isolation test")
    finally:
        try:
            await client.delete_collection(coll)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
