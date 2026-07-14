"""
User Management Service
All user business logic — create, update, activate, deactivate, assign roles.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ConflictException,
    DuplicateException,
    NotFoundException,
    ForbiddenException,
)
from app.core.security import hash_password
from app.core.logging import get_logger
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.repositories.role_repository import RoleRepository
from app.schemas.user import UserCreateRequest, UserUpdateRequest, UserResponse

logger = get_logger(__name__)


class UserService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.user_repo = UserRepository(db)
        self.role_repo = RoleRepository(db)

    # ── Create ────────────────────────────────────────────────────────────────

    async def create_user(
        self,
        payload: UserCreateRequest,
        organization_id: UUID,
        created_by: UUID,
    ) -> User:
        existing = await self.user_repo.get_by_email(payload.email.lower())
        if existing:
            raise DuplicateException("User", "email", payload.email)

        user = await self.user_repo.create(
            email=payload.email.lower(),
            full_name=payload.full_name.strip(),
            hashed_password=hash_password(payload.password),
            phone=payload.phone,
            job_title=payload.job_title,
            timezone=payload.timezone,
            locale=payload.locale,
            organization_id=organization_id,
            is_verified=True,
        )

        if payload.role_ids:
            await self.user_repo.assign_roles(user, payload.role_ids, created_by)

        user = await self.user_repo.get_by_id_with_roles(user.id)
        logger.info("User created", extra={"user_id": str(user.id), "created_by": str(created_by)})
        return user

    # ── List ──────────────────────────────────────────────────────────────────

    async def list_users(
        self,
        organization_id: UUID,
        search: Optional[str],
        page: int,
        page_size: int,
    ) -> Tuple[List[User], int]:
        return await self.user_repo.list_by_organization(
            organization_id, search, page, page_size
        )

    # ── Get ───────────────────────────────────────────────────────────────────

    async def get_user(self, user_id: UUID, organization_id: UUID) -> User:
        user = await self.user_repo.get_by_id_with_roles(user_id)
        if not user or user.organization_id != organization_id:
            raise NotFoundException("User", user_id)
        return user

    # ── Update ────────────────────────────────────────────────────────────────

    async def update_user(
        self,
        user_id: UUID,
        organization_id: UUID,
        payload: UserUpdateRequest,
    ) -> User:
        user = await self.get_user(user_id, organization_id)
        update_data = payload.model_dump(exclude_none=True)
        await self.user_repo.update(user, **update_data)
        return await self.user_repo.get_by_id_with_roles(user.id)

    # ── Activate / Deactivate ─────────────────────────────────────────────────

    async def activate_user(self, user_id: UUID, organization_id: UUID) -> User:
        user = await self.get_user(user_id, organization_id)
        if user.is_active:
            raise ConflictException("User is already active.")
        await self.user_repo.update(user, is_active=True)
        return user

    async def deactivate_user(self, user_id: UUID, organization_id: UUID, requestor_id: UUID) -> User:
        if user_id == requestor_id:
            raise ForbiddenException("You cannot deactivate your own account.")
        user = await self.get_user(user_id, organization_id)
        if not user.is_active:
            raise ConflictException("User is already inactive.")
        await self.user_repo.update(user, is_active=False)
        return user

    # ── Assign Roles ──────────────────────────────────────────────────────────

    async def assign_roles(
        self,
        user_id: UUID,
        organization_id: UUID,
        role_ids: List[UUID],
        assigned_by: UUID,
    ) -> User:
        user = await self.get_user(user_id, organization_id)
        await self.user_repo.assign_roles(user, role_ids, assigned_by)
        return await self.user_repo.get_by_id_with_roles(user.id)

    # ── Delete (soft) ─────────────────────────────────────────────────────────

    async def delete_user(self, user_id: UUID, organization_id: UUID, requestor_id: UUID) -> None:
        if user_id == requestor_id:
            raise ForbiddenException("You cannot delete your own account.")
        user = await self.get_user(user_id, organization_id)
        await self.user_repo.soft_delete(user)
        logger.info("User soft-deleted", extra={"user_id": str(user_id)})
