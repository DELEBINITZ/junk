"""FastAPI application factory + lifespan — the process entrypoint.

TWO PATTERNS LIVE HERE, and they're worth naming for a newcomer:

  * APP FACTORY (``create_app``) — a function that builds and returns the FastAPI
    app (middleware, routers, system endpoints) rather than a module-level app
    assembled by side effects. Factories are easy to call from tests with custom
    settings, and keep startup explicit.
  * LIFESPAN (``lifespan``) — an async context manager FastAPI runs once around
    the server's life: the code BEFORE ``yield`` is startup, the code AFTER is
    shutdown. This is where we call the composition root (bootstrap.build_services)
    exactly once and tear it down cleanly on exit.

Boot sequence: build services (which discovers capability modules), open the DB
pool if configured, wire auth state, seed demo data, expose the routers.
Everything downstream is config-gated, so ``uvicorn app.main:app`` runs on the
zero-infra (deterministic, self-hosted, offline) path out of the box.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from app.config import get_settings
from app.core.api import (
    ConcurrencyMiddleware,
    admin_router,
    auth_router,
    chat_router,
    install_error_handlers,
)
from app.core.bootstrap import build_services, seed_demo
from app.core.observability import configure_logging, get_logger

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown around the server's lifetime (see module docstring).
    Everything before ``yield`` runs at boot; the ``finally`` runs at shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log = get_logger("asi.main")

    # The single call to the composition root — builds the whole service graph.
    services = build_services(settings)
    # Only open a real DB pool when Postgres is the configured backend; the
    # in-memory default needs nothing. The import is local so the postgres driver
    # isn't even required on the zero-infra path.
    if settings.store_backend == "postgres":
        from app.core.db.postgres import get_database
        from app.core.errors import ConfigError

        db = get_database(settings)
        await db.open()
        # PROVE tenant isolation holds before taking traffic: a SUPERUSER/BYPASSRLS
        # DATABASE_URL silently defeats RLS (the store SELECTs have no WHERE org_id).
        # Prod = fail-closed (refuse to boot); dev = warn but continue so the local
        # superuser quickstart still works.
        try:
            await db.verify_rls_isolation()
        except ConfigError as exc:
            if settings.is_prod:
                raise
            log.warning("rls.selftest_failed",
                        extra={"event": "rls_selftest", "detail": str(exc)})

    # Stash long-lived objects on ``app.state`` so request dependencies (auth,
    # handlers) can reach them without rebuilding anything per request.
    app.state.services = services
    app.state.settings = settings

    await seed_demo(services)   # dev-only corpus seeding (no-op in prod, gated by config)
    log.info("startup.complete", extra={"event": "startup"})
    try:
        yield   # <-- the server runs and serves requests here, until shutdown
    finally:
        # Symmetric teardown: close every service, then the DB pool. Runs even if
        # startup partially failed, so we don't leak connections.
        await services.aclose()
        if settings.store_backend == "postgres":
            from app.core.db.postgres import get_database

            await get_database(settings).close()


def create_app() -> FastAPI:
    """The app factory: assemble the FastAPI app (middleware + routers + system
    endpoints) and attach the lifespan above. ``uvicorn app.main:app`` ends up
    calling the module-level ``app`` this returns."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    # CORS first. Note the subtle rule: the browser spec forbids wildcard origin
    # together with credentials, so we only allow credentials when origins are
    # explicitly listed (not "*").
    star = settings.cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=not star,  # '*' + credentials is invalid per CORS spec
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Back-pressure middleware: caps concurrent generations (global + per-org) and
    # queues the overflow, returning 503 when the queue is full. This is the
    # fairness/overload guard configured in config.py.
    app.add_middleware(
        ConcurrencyMiddleware,
        max_global=settings.max_concurrent_generations,
        queue_max=settings.queue_max_size,
    )
    # Register the AppError -> JSON problem-response handlers (see errors.py), so
    # typed errors raised anywhere become uniform responses.
    install_error_handlers(app)

    # The feature routers: auth (login/token), chat (the agent endpoints), admin.
    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    # ---- system endpoints ----
    # Liveness/readiness/metrics for ops and orchestrators (k8s probes, scrapers).
    @app.get("/healthz", tags=["system"])
    async def healthz() -> dict:
        # Liveness: the process is up and serving. Always 200 if we can answer.
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    async def readyz() -> JSONResponse:
        # Readiness: are we ready to take traffic yet? Only true once the lifespan
        # has finished building ``services``. 503 while still starting tells a load
        # balancer to hold off routing requests here.
        ready = getattr(app.state, "services", None) is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "starting"},
        )

    @app.get("/metrics", tags=["system"])
    async def metrics(fmt: str = "json") -> Response:
        # Expose the metrics registry either as Prometheus text (for a scraper) or
        # as a JSON snapshot (human/debug). Falls back to JSON if Prometheus isn't
        # available. See observability/metrics.py for what's tracked.
        services = getattr(app.state, "services", None)
        if services is None:
            return JSONResponse({"status": "starting"})
        if fmt == "prometheus":
            body = services.metrics.render_prometheus()
            if body is not None:
                return Response(content=body, media_type="text/plain; version=0.0.4")
        return JSONResponse(services.metrics.snapshot())

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        # Serve the bundled demo frontend if present; otherwise a tiny JSON pointer
        # to the docs and health endpoints.
        if _FRONTEND.exists():
            return FileResponse(str(_FRONTEND))
        return JSONResponse({"service": settings.app_name, "docs": "/docs", "health": "/healthz"})

    return app


# The module-level ASGI app uvicorn imports as ``app.main:app``. Building it at
# import time is what makes ``uvicorn app.main:app`` work with no extra glue.
app = create_app()
