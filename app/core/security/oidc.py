"""OIDC token verification (``auth_provider=oidc``).

Verifies an RS256/ES256 access token against the IdP's JWKS and projects the
configured claims into a :class:`SecurityContext`. Used in production when an
external IdP (Keycloak, Auth0, Entra, …) owns identity. The local JWT path
(``jwt.py``) is the zero-infra default.
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.errors import AuthError, ConfigError
from app.core.security.context import SecurityContext


class OIDCVerifier:
    def __init__(self, settings: Settings) -> None:
        if not settings.oidc_jwks_url:
            raise ConfigError("auth_provider=oidc requires OIDC_JWKS_URL")
        self.settings = settings
        self._jwk_client = None  # lazy

    def _client(self):
        if self._jwk_client is None:
            import jwt  # lazy

            self._jwk_client = jwt.PyJWKClient(self.settings.oidc_jwks_url)
        return self._jwk_client

    def verify(self, token: str) -> SecurityContext:
        import jwt  # lazy

        s = self.settings
        try:
            signing_key = self._client().get_signing_key_from_jwt(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=s.oidc_audience or None,
                issuer=s.oidc_issuer or None,
                options={"verify_aud": bool(s.oidc_audience)},
            )
        except Exception as exc:  # noqa: BLE001 - normalize to AuthError
            raise AuthError(f"oidc verification failed: {exc}") from exc

        org_id = str(claims.get(s.oidc_org_claim, "") or "")
        if not org_id:
            raise AuthError(f"oidc token missing org claim '{s.oidc_org_claim}'")
        roles = _coerce_roles(claims.get(s.oidc_roles_claim))
        return SecurityContext(
            org_id=org_id,
            user_id=str(claims.get("sub", "")),
            roles=roles,
            email=str(claims.get("email", "")),
            token_id=str(claims.get("jti", "")),
            claims=claims,
        )


def _coerce_roles(raw: object) -> tuple[str, ...]:
    if raw is None:
        return ("viewer",)
    if isinstance(raw, str):
        return tuple(r.strip() for r in raw.split(",") if r.strip()) or ("viewer",)
    if isinstance(raw, (list, tuple)):
        return tuple(str(r) for r in raw) or ("viewer",)
    return ("viewer",)


_verifier: OIDCVerifier | None = None


def get_oidc_verifier(settings: Settings) -> OIDCVerifier:
    global _verifier
    if _verifier is None:
        _verifier = OIDCVerifier(settings)
    return _verifier


__all__ = ["OIDCVerifier", "get_oidc_verifier"]
