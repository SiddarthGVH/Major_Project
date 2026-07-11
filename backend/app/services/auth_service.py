"""
Authentication Service
All authentication business logic lives here — never in route handlers.
"""
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import (
    DuplicateException,
    InvalidCredentialsException,
    NotFoundException,
    TokenExpiredException,
    UnauthorizedException,
    WeakPasswordException,
    BusinessRuleException,
)
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    generate_secure_token,
    hash_password,
    hash_token,
    verify_password,
    check_password_strength,
)
from app.core.logging import get_logger
from app.models.user import User
from app.repositories.organization_repository import OrganizationRepository
from app.repositories.role_repository import RoleRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    ResetPasswordRequest,
    ChangePasswordRequest,
)

logger = get_logger(__name__)


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:100]


class AuthService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.user_repo = UserRepository(db)
        self.org_repo = OrganizationRepository(db)
        self.role_repo = RoleRepository(db)

    # ── Register ──────────────────────────────────────────────────────────────

    async def register(self, payload: RegisterRequest, client_ip: str = "") -> TokenResponse:
        """
        Create a new Organization + Admin user.
        Used during the onboarding / sign-up flow.
        """
        # 1. Validate password strength
        is_valid, reason = check_password_strength(payload.password)
        if not is_valid:
            raise WeakPasswordException(reason)

        # 2. Check email uniqueness
        existing = await self.user_repo.get_by_email(payload.email.lower())
        if existing:
            raise DuplicateException("User", "email", payload.email)

        # 3. Create Organization
        org_name = payload.organization_name.strip()
        existing_org = await self.org_repo.get_by_name(org_name)
        if existing_org:
            raise DuplicateException("Organization", "name", org_name)

        slug = _slugify(org_name)
        organization = await self.org_repo.create(
            name=org_name,
            slug=slug,
        )

        # 4. Assign Admin role to new user
        admin_role = await self.role_repo.get_by_name("admin")

        # 5. Create User
        user = await self.user_repo.create(
            email=payload.email.lower(),
            full_name=payload.full_name.strip(),
            hashed_password=hash_password(payload.password),
            organization_id=organization.id,
            is_verified=True,       # Auto-verify on self-registration (email verify later)
            is_superuser=True,
            last_login_ip=client_ip,
            last_login_at=datetime.now(timezone.utc),
        )

        # 6. Assign admin role
        if admin_role:
            await self.user_repo.assign_roles(user, [admin_role.id], user.id)
            # Reload with roles
            user = await self.user_repo.get_by_id_with_roles(user.id)

        logger.info("New user registered", extra={"user_id": str(user.id), "org": org_name})

        return await self._build_tokens(user)

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(self, payload: LoginRequest, client_ip: str = "") -> TokenResponse:
        user = await self.user_repo.get_by_email(payload.email.lower())

        if not user or not verify_password(payload.password, user.hashed_password):
            raise InvalidCredentialsException()

        if not user.is_active:
            raise UnauthorizedException("Your account has been deactivated.")

        # Update last login metadata
        await self.user_repo.update(
            user,
            last_login_at=datetime.now(timezone.utc),
            last_login_ip=client_ip,
        )

        logger.info("User logged in", extra={"user_id": str(user.id)})
        return await self._build_tokens(user)

    # ── Refresh Token ─────────────────────────────────────────────────────────

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        payload = decode_refresh_token(refresh_token)
        user_id = UUID(payload["sub"])

        user = await self.user_repo.get_by_id_with_roles(user_id)
        if not user or not user.is_active:
            raise UnauthorizedException()

        return await self._build_tokens(user)

    # ── Forgot Password ───────────────────────────────────────────────────────

    async def forgot_password(self, email: str) -> str:
        """
        Generate a password reset token.
        Returns the plaintext token (to be sent via email).
        We store only the hash.
        """
        user = await self.user_repo.get_by_email(email.lower())
        if not user:
            # Security: do not reveal whether email exists
            logger.info("Password reset requested for unknown email", extra={"email": email})
            return ""

        token = generate_secure_token()
        token_hash = hash_token(token)
        expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES
        )

        await self.user_repo.update(
            user,
            password_reset_token=token_hash,
            password_reset_expires_at=expires_at,
        )

        logger.info("Password reset token generated", extra={"user_id": str(user.id)})
        return token  # Caller sends this in email

    # ── Reset Password ────────────────────────────────────────────────────────

    async def reset_password(self, payload: ResetPasswordRequest) -> None:
        token_hash = hash_token(payload.token)
        user = await self.user_repo.get_by_reset_token(token_hash)

        if not user:
            raise NotFoundException("Password reset token", payload.token[:8] + "...")

        if user.password_reset_expires_at and user.password_reset_expires_at < datetime.now(timezone.utc):
            raise TokenExpiredException()

        is_valid, reason = check_password_strength(payload.new_password)
        if not is_valid:
            raise WeakPasswordException(reason)

        await self.user_repo.update(
            user,
            hashed_password=hash_password(payload.new_password),
            password_reset_token=None,
            password_reset_expires_at=None,
        )
        logger.info("Password reset complete", extra={"user_id": str(user.id)})

    # ── Change Password ───────────────────────────────────────────────────────

    async def change_password(self, user: User, payload: ChangePasswordRequest) -> None:
        if not verify_password(payload.current_password, user.hashed_password):
            raise InvalidCredentialsException()

        is_valid, reason = check_password_strength(payload.new_password)
        if not is_valid:
            raise WeakPasswordException(reason)

        await self.user_repo.update(
            user,
            hashed_password=hash_password(payload.new_password),
        )
        logger.info("Password changed", extra={"user_id": str(user.id)})

    # ── Token builder ─────────────────────────────────────────────────────────

    async def _build_tokens(self, user: User) -> TokenResponse:
        """Build access + refresh token pair from user entity."""
        # Aggregate permissions from all assigned roles
        permissions: list[str] = []
        for ur in user.user_roles:
            if ur.role:
                perms = await self.role_repo.get_permissions_for_role(ur.role.id)
                permissions.extend(perms)
        permissions = list(set(permissions))

        role_name = user.primary_role or "sales_rep"

        access_token = create_access_token(
            user_id=user.id,
            organization_id=user.organization_id,
            role=role_name,
            permissions=permissions,
        )
        refresh_token = create_refresh_token(user_id=user.id)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )
