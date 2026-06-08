from dataclasses import dataclass

ROLE_HIERARCHY = {"viewer": 0, "analyst": 1, "admin": 2}


@dataclass(frozen=True)
class SecurityContext:
    org_id: str
    user_id: str
    roles: tuple[str, ...]

    @property
    def max_role_level(self) -> int:
        return max((ROLE_HIERARCHY.get(r, 0) for r in self.roles), default=0)

    def has_role(self, required: str) -> bool:
        required_level = ROLE_HIERARCHY.get(required, 0)
        return self.max_role_level >= required_level


def check_rbac(roles: list[str], required_role: str) -> bool:
    """Check if user's roles satisfy the minimum required role."""
    user_max = max((ROLE_HIERARCHY.get(r, 0) for r in roles), default=0)
    return user_max >= ROLE_HIERARCHY.get(required_role, 0)


def require_role(required: str):
    """Decorator factory for tools that require a minimum role.

    Usage:
        @require_role("analyst")
        @tool
        async def sensitive_tool(...):
            ...
    """
    def decorator(func):
        original = func

        async def wrapper(*args, **kwargs):
            from langgraph.config import get_config
            config = get_config()
            roles = config["configurable"].get("roles", [])
            if not check_rbac(roles, required):
                return f"Access denied: requires '{required}' role or higher."
            return await original(*args, **kwargs)

        wrapper.__name__ = original.__name__
        wrapper.__doc__ = original.__doc__
        wrapper.__annotations__ = getattr(original, "__annotations__", {})
        return wrapper

    return decorator
