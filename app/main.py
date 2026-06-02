"""FastAPI application factory + lifespan.

Boot: build services (which discovers capability modules), open the DB pool if
configured, wire auth state, seed demo data, and expose the routers. Everything
is config-gated, so ``uvicorn app.main:app`` runs on the zero-infra path out of
the box.
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
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    log = get_logger("asi.main")

    services = build_services(settings)
    if settings.store_backend == "postgres":
        from app.core.db.postgres import get_database

        await get_database(settings).open()

    # state for auth dependencies
    app.state.services = services
    app.state.settings = settings
    app.state.revocation_store = services.revocation_store
    if settings.auth_provider == "oidc":
        from app.core.security.oidc import get_oidc_verifier

        app.state.oidc_verifier = get_oidc_verifier(settings)

    await seed_demo(services)
    log.info("startup.complete", extra={"event": "startup"})
    try:
        yield
    finally:
        await services.aclose()
        if settings.store_backend == "postgres":
            from app.core.db.postgres import get_database

            await get_database(settings).close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    star = settings.cors_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=not star,  # '*' + credentials is invalid per CORS spec
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        ConcurrencyMiddleware,
        max_global=settings.max_concurrent_generations,
        queue_max=settings.queue_max_size,
    )
    install_error_handlers(app)

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    # ---- system endpoints ----
    @app.get("/healthz", tags=["system"])
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", tags=["system"])
    async def readyz() -> JSONResponse:
        ready = getattr(app.state, "services", None) is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "starting"},
        )

    @app.get("/metrics", tags=["system"])
    async def metrics(fmt: str = "json") -> Response:
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
        if _FRONTEND.exists():
            return FileResponse(str(_FRONTEND))
        return JSONResponse({"service": settings.app_name, "docs": "/docs", "health": "/healthz"})

    return app


app = create_app()
