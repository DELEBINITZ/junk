from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

from security_intel.config import Settings
from security_intel.security.rbac import SecurityContext

_bearer_scheme = HTTPBearer(auto_error=False)


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _verify_api_key(request: Request, settings: Settings) -> None:
    """Check X-API-Key header or api_key query param."""
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key or api_key not in settings.api_key_list:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _decode_jwt(token: str, settings: Settings) -> dict:
    """Decode and verify JWT token."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


async def require_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> SecurityContext:
    """FastAPI dependency: verify API key + JWT, return SecurityContext."""
    settings = get_settings(request)
    _verify_api_key(request, settings)

    token = None
    if credentials:
        token = credentials.credentials
    else:
        token = request.query_params.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Missing authorization token")

    payload = _decode_jwt(token, settings)

    return SecurityContext(
        org_id=payload.get("org_id", "default"),
        user_id=payload.get("sub", "anonymous"),
        roles=tuple(payload.get("roles", ["viewer"])),
    )


async def optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> SecurityContext:
    """Auth dependency — strict in production, permissive in dev.

    Production (ENVIRONMENT=prod): requires valid API key + JWT. No fallback.
    Development: falls back to demo credentials for easy testing.
    """
    settings = get_settings(request)

    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if api_key and api_key in settings.api_key_list:
        token = None
        if credentials:
            token = credentials.credentials
        else:
            token = request.query_params.get("access_token")

        if token:
            try:
                payload = _decode_jwt(token, settings)
                return SecurityContext(
                    org_id=payload.get("org_id", "default"),
                    user_id=payload.get("sub", "anonymous"),
                    roles=tuple(payload.get("roles", ["viewer"])),
                )
            except HTTPException:
                if settings.environment == "prod":
                    raise
        elif settings.environment == "prod":
            raise HTTPException(status_code=401, detail="JWT required in production")
    elif settings.environment == "prod":
        raise HTTPException(status_code=401, detail="API key required in production")

    # Dev/demo fallback only
    return SecurityContext(org_id="demo", user_id="demo-user", roles=("admin",))
