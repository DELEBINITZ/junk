"""HTTP API: routers, error handlers, middleware."""

from app.core.api.admin_router import router as admin_router
from app.core.api.auth_router import router as auth_router
from app.core.api.chat_router import router as chat_router
from app.core.api.errors import install_error_handlers
from app.core.api.middleware import ConcurrencyMiddleware

__all__ = [
    "auth_router",
    "chat_router",
    "admin_router",
    "install_error_handlers",
    "ConcurrencyMiddleware",
]
