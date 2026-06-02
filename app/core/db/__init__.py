"""Durable storage + the org-isolation backbone.

`postgres.py` provides the per-org RLS transaction primitive; `migrations/` holds
the forward-only schema; `migrate.py` applies them. The org GUC
(`app.organization_id`) makes row-level security enforce tenant isolation even if
an application-layer WHERE clause is forgotten — defense in depth (plan §8.2).
"""
