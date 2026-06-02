"""PostgreSQL connection helper for the production-shaped RLS path."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://contract_user:contract_pass@localhost:5432/contract_intelligence",
)


@contextmanager
def tenant_connection(organization_id: str) -> Iterator[psycopg.Connection]:
    """Open a PostgreSQL connection scoped by the tenant RLS setting.

    RLS policies in `schema.sql` read `app.organization_id`; setting it at the
    connection level keeps SQL queries tenant-scoped even if a caller forgets a
    WHERE clause.
    """

    with psycopg.connect(DATABASE_URL) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT set_config('app.organization_id', %s, true)", (organization_id,))
        yield connection
