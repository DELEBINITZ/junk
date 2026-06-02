"""OIDC token verification (``auth_provider=oidc``).

================================ OIDC IN ONE BREATH =======================
In production an enterprise usually wants its OWN identity provider (IdP) —
Keycloak, Auth0, Entra/Azure AD, Okta — to own logins. OIDC (OpenID Connect) is
the standard for that. The key difference from local JWT (jwt.py):

  * Local JWT is SYMMETRIC — we hold the secret, so we both sign and verify.
  * OIDC is ASYMMETRIC — the IdP signs tokens with its PRIVATE key; we verify
    with its PUBLIC key. We never possess the signing secret, so we can only
    *check* tokens, never mint them. Identity lives entirely with the IdP.

JWKS (JSON Web Key Set) is how we get those public keys: the IdP publishes them
at a well-known URL (``oidc_jwks_url``). A token's header names WHICH key signed
it (its "kid"); we fetch the matching public key from the JWKS and verify with
it. On top of the signature we also check the token was minted FOR us
(``audience``) and BY the expected issuer — standard OIDC validation.

Verifies an RS256/ES256 access token against the IdP's JWKS and projects the
configured claims into a :class:`SecurityContext`. Used in production when an
external IdP owns identity. The local JWT path (``jwt.py``) is the zero-infra
default. Either way the OUTPUT is the same SecurityContext, so the rest of the
app is identical no matter which provider authenticated the user.
===========================================================================
"""

from __future__ import annotations

from typing import Any

from app.config import Settings
from app.core.errors import AuthError, ConfigError
from app.core.security.context import SecurityContext


class OIDCVerifier:
    """Verifies IdP-issued tokens against the published JWKS. One instance is
    reused across requests (the JWKS client caches fetched keys)."""

    def __init__(self, settings: Settings) -> None:
        # No JWKS URL => we have no way to obtain the IdP's public keys, so OIDC
        # mode simply cannot work. Fail fast at construction, not mid-request.
        if not settings.oidc_jwks_url:
            raise ConfigError("auth_provider=oidc requires OIDC_JWKS_URL")
        self.settings = settings
        self._jwk_client = None  # lazy

    def _client(self):
        # Build the JWKS client lazily and keep it: PyJWKClient fetches and CACHES
        # the IdP's signing keys, so we don't hit the JWKS endpoint on every call.
        if self._jwk_client is None:
            import jwt  # lazy

            self._jwk_client = jwt.PyJWKClient(self.settings.oidc_jwks_url)
        return self._jwk_client

    def verify(self, token: str) -> SecurityContext:
        """Cryptographically verify the token, then map its claims onto our
        SecurityContext. Returns a TRUSTED identity or raises AuthError."""
        import jwt  # lazy

        s = self.settings
        try:
            # Read the token header's "kid", fetch the matching PUBLIC key from the
            # JWKS, and verify with it. ``audience``/``issuer`` checks ensure the
            # token was minted FOR this app BY the expected IdP, not just any valid
            # token from somewhere. (verify_aud only if an audience is configured.)
            signing_key = self._client().get_signing_key_from_jwt(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],   # asymmetric algorithms IdPs use
                audience=s.oidc_audience or None,
                issuer=s.oidc_issuer or None,
                options={"verify_aud": bool(s.oidc_audience)},
            )
        except Exception as exc:  # noqa: BLE001 - normalize to AuthError
            # Any verification failure (bad signature, wrong audience, expired,
            # unknown key) collapses to one auth error so callers stay simple.
            raise AuthError(f"oidc verification failed: {exc}") from exc

        # Different IdPs put the tenant under different claim names, so WHICH claim
        # carries the org is configurable. But notice it is still read ONLY from
        # the verified token — never from request input — preserving isolation.
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
    """Normalize however the IdP encodes roles into our ``tuple[str, ...]``.
    Different IdPs emit roles as a comma-string, a JSON array, or omit them — we
    accept all of those. The safe DEFAULT is the LOWEST privilege (``viewer``):
    if roles are missing or unrecognized we grant least access, never more."""
    if raw is None:
        return ("viewer",)
    if isinstance(raw, str):
        return tuple(r.strip() for r in raw.split(",") if r.strip()) or ("viewer",)
    if isinstance(raw, (list, tuple)):
        return tuple(str(r) for r in raw) or ("viewer",)
    return ("viewer",)


# Process-wide singleton verifier (so the JWKS key cache is shared, not rebuilt).
_verifier: OIDCVerifier | None = None


def get_oidc_verifier(settings: Settings) -> OIDCVerifier:
    """Return the shared verifier, constructing it once on first use."""
    global _verifier
    if _verifier is None:
        _verifier = OIDCVerifier(settings)
    return _verifier


__all__ = ["OIDCVerifier", "get_oidc_verifier"]
