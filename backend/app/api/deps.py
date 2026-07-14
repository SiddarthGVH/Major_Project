"""
FastAPI Dependency Injection
Provides reusable dependencies for:
  - Database session
  - Current authenticated user
  - RBAC permission enforcement
"""
from typing import Annotated, Callable
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    ForbiddenException,
    UnauthorizedException,
    NotFoundException,
)
from app.core.security import decode_access_token
from app.database.connection import get_db
from app.models.user import User
from app.repositories.user_repository import UserRepository

# ── OAuth2 / Bearer extractor ─────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)


# ── Current User ──────────────────────────────────────────────────────────────

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Validate JWT access token and return the authenticated User.
    Raises UnauthorizedException if token is missing/invalid.
    """
    if not credentials:
        raise UnauthorizedException("Missing Bearer token.")

    payload = decode_access_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise UnauthorizedException("Invalid token payload.")

    user_repo = UserRepository(db)
    user = await user_repo.get_by_id_with_roles(UUID(user_id))

    if not user or not user.is_active or user.is_deleted:
        raise UnauthorizedException("User account not found or is inactive.")

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Alias that also asserts the user is active (for clarity in routes)."""
    return current_user


# ── Organisation-scoped current user ─────────────────────────────────────────

CurrentUser = Annotated[User, Depends(get_current_active_user)]
DBSession = Annotated[AsyncSession, Depends(get_db)]


# ── RBAC Permission Dependency Factory ────────────────────────────────────────

def require_permission(permission: str) -> Callable:
    """
    Dependency factory.
    Usage:
        @router.post("/", dependencies=[Depends(require_permission("company:create"))])
    """
    async def _check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> None:
        if not credentials:
            raise UnauthorizedException("Missing Bearer token.")

        payload = decode_access_token(credentials.credentials)
        permissions: list[str] = payload.get("permissions", [])

        # Superusers bypass permission checks
        if payload.get("role") == "admin":
            return

        if permission not in permissions:
            raise ForbiddenException(permission)

    return _check


def require_any_permission(*permissions: str) -> Callable:
    """Pass if the user has AT LEAST ONE of the given permissions."""
    async def _check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> None:
        if not credentials:
            raise UnauthorizedException("Missing Bearer token.")

        payload = decode_access_token(credentials.credentials)
        user_permissions: list[str] = payload.get("permissions", [])

        if payload.get("role") == "admin":
            return

        if not any(p in user_permissions for p in permissions):
            raise ForbiddenException(permissions[0])

    return _check


def require_role(*roles: str) -> Callable:
    """Pass if the user has any of the listed roles."""
    async def _check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    ) -> None:
        if not credentials:
            raise UnauthorizedException("Missing Bearer token.")

        payload = decode_access_token(credentials.credentials)
        user_role = payload.get("role", "")

        if user_role not in roles:
            raise ForbiddenException(f"Required role: {', '.join(roles)}")

    return _check
