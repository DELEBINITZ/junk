"""Migration runner (``asi-migrate`` console script).

================================ MENTAL MODEL =============================
SQL MIGRATIONS are how the database schema evolves in version-controlled steps.
Each ``migrations/NNNN_name.sql`` file is one forward change (create a table, add
an index, define an RLS policy, ...). This tiny runner applies any files that
haven't run yet, in filename order, and records each in a ``schema_migrations``
table so it is IDEMPOTENT — run it a hundred times and it only ever applies each
migration once. That table is the "where are we" bookmark for the schema.

It uses a SYNCHRONOUS psycopg connection (not the async pool the app uses) on
purpose: migrations are a one-shot deploy step, not request traffic, so the
simple blocking driver is the right tool. Exposed as the ``asi-migrate`` console
script (see pyproject) so deploys can just call it before starting the app.
===========================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

# All *.sql migration files live next to this module, in ./migrations.
MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(dsn: str) -> list[str]:
    """Apply every not-yet-applied migration and return the versions applied this
    run (empty list => already up to date). Safe to call repeatedly."""
    import psycopg  # lazy: only needed at deploy time, not for the in-memory path

    applied: list[str] = []
    # autocommit=False so each migration + its bookkeeping row commit together.
    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Bootstrap the ledger table itself (idempotent) before reading it.
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.commit()
            # Which versions have already run? (the set of bookmarks)
            cur.execute("SELECT version FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}
            # Sorted glob => migrations apply in deterministic, numeric filename
            # order, which is exactly the order they were authored to run in.
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                version = path.stem            # the filename (minus .sql) IS the version id
                if version in done:
                    continue                   # already applied -> skip (idempotency)
                sql = path.read_text(encoding="utf-8")
                cur.execute(sql)               # run the migration's statements
                # Record it in the SAME transaction, so a crash can't leave a
                # migration applied-but-untracked (or tracked-but-unapplied).
                cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
                conn.commit()
                applied.append(version)
    return applied


def main() -> int:
    """CLI entry point for the ``asi-migrate`` script. Reads DATABASE_URL from
    settings, runs the migrations, and prints what happened. Returns a process
    exit code: 2 if no DB is configured, 0 on success."""
    from app.config import get_settings

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    applied = run_migrations(settings.database_url)
    print(f"migrations applied: {applied or '(none — up to date)'}")
    return 0


if __name__ == "__main__":
    # ``python -m app.core.db.migrate`` -> run main() and exit with its code.
    raise SystemExit(main())
