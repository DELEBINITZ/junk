"""Migration runner (``asi-migrate`` console script).

Applies ``migrations/*.sql`` in order, idempotently, tracking applied versions in
``schema_migrations``. Uses a synchronous psycopg connection — run once at deploy.
"""

from __future__ import annotations

import sys
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(dsn: str) -> list[str]:
    import psycopg  # lazy

    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations "
                "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            conn.commit()
            cur.execute("SELECT version FROM schema_migrations")
            done = {r[0] for r in cur.fetchall()}
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                version = path.stem
                if version in done:
                    continue
                sql = path.read_text(encoding="utf-8")
                cur.execute(sql)
                cur.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (version,))
                conn.commit()
                applied.append(version)
    return applied


def main() -> int:
    from app.config import get_settings

    settings = get_settings()
    if not settings.database_url:
        print("DATABASE_URL not set", file=sys.stderr)
        return 2
    applied = run_migrations(settings.database_url)
    print(f"migrations applied: {applied or '(none — up to date)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
