"""Forward-only SQL migration runner.

Applies every `migrations/*.sql` not yet recorded in `schema_migrations`, in
filename order, each in its own transaction. Run with:

    .venv/bin/python -m app.core.db.migrate

Requires the psycopg driver (the 'prod' extra) and a reachable DATABASE_URL.
Statements are split on ';' (the migration SQL contains no semicolons inside
string literals).
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings


logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def _statements(sql: str) -> list[str]:
    return [stmt.strip() for stmt in sql.split(";") if stmt.strip()]


def run_migrations(database_url: str | None = None) -> list[str]:
    import psycopg

    url = database_url or settings.database_url
    applied: list[str] = []
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            version = path.name
            if version in done:
                continue
            with conn.transaction():
                for statement in _statements(path.read_text()):
                    conn.execute(statement)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            applied.append(version)
            logger.info("migration.applied", extra={"version": version})
    return applied


if __name__ == "__main__":  # pragma: no cover
    result = run_migrations()
    print("applied:", ", ".join(result) if result else "none (up to date)")
