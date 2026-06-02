"""FastAPI application factory for the Contract Intelligence PoC."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, Request

from app.admin_router import router as admin_router
from app.agent.router import router as ai_router
from app.auth.dependencies import require_user
from app.auth.router import router as auth_router
from app.config import settings
from app.db.repository import DataStore, get_store
from app.db.seed import seed_demo_data
from app.documents.router import router as documents_router
from app.domain import User
from app.mcp_server.router import router as mcp_router
from app.observability.logging import (
    configure_logging,
    reset_request_id,
    safe_extra,
    set_request_id,
)
from app.reports.router import router as reports_router

from app.core.api import router as core_router
from app.core.registry import get_registry


logger = logging.getLogger(__name__)


def create_app(seed: bool = True) -> FastAPI:
    """Create the API app and optionally seed the deterministic demo corpus."""

    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "app.startup.begin",
            extra=safe_extra(app_name=settings.app_name, seed_enabled=seed),
        )
        if seed:
            corpus_dir = Path(settings.corpus_dir)
            store = get_store()
            if store.is_empty() and corpus_dir.exists():
                logger.info(
                    "app.seed.begin",
                    extra=safe_extra(corpus_dir=str(corpus_dir)),
                )
                seed_demo_data(store, corpus_dir)
                logger.info(
                    "app.seed.complete",
                    extra=safe_extra(
                        documents=store.count_documents(),
                        organizations=store.count_organizations(),
                    ),
                )
        # Discover capability modules at boot (fail fast on an invalid manifest).
        registry = get_registry()
        logger.info("app.capabilities.ready", extra=safe_extra(modules=list(registry.modules)))
        logger.info("app.startup.complete")
        yield

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

    if settings.cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        """Log request outcome and latency with a stable request ID."""

        request_id = request.headers.get("x-request-id") or str(uuid4())
        token = set_request_id(request_id)
        start = perf_counter()
        status_code = 500
        log_request = request.url.path != "/health"
        if log_request:
            logger.info(
                "http.request.start",
                extra=safe_extra(
                    method=request.method,
                    path=request.url.path,
                    client=request.client.host if request.client else None,
                ),
            )
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        except Exception:
            logger.exception(
                "http.request.error",
                extra=safe_extra(method=request.method, path=request.url.path),
            )
            raise
        finally:
            if log_request:
                logger.info(
                    "http.request.complete",
                    extra=safe_extra(
                        method=request.method,
                        path=request.url.path,
                        status_code=status_code,
                        duration_ms=round((perf_counter() - start) * 1000, 2),
                    ),
                )
            reset_request_id(token)

    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(ai_router)
    app.include_router(reports_router)
    app.include_router(admin_router)
    app.include_router(mcp_router)
    app.include_router(core_router)

    @app.get("/health")
    def health(store: DataStore = Depends(get_store)):
        return {
            "status": "ok",
            "documents": store.count_documents(),
            "organizations": store.count_organizations(),
            "embedding_dimensions": store.embedder.dimensions,
        }

    @app.get("/me")
    def me(user: User = Depends(require_user)):
        return {
            "id": user.id,
            "organization_id": user.organization_id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
        }

    return app


app = create_app()
