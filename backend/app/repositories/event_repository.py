"""
Domain event repository.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event import DomainEvent
from app.repositories.base import BaseRepository


class EventRepository(BaseRepository[DomainEvent]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(DomainEvent, db)

    async def get_active_by_id(self, event_id: UUID, organization_id: UUID) -> Optional[DomainEvent]:
        stmt = select(DomainEvent).where(
            DomainEvent.id == event_id,
            DomainEvent.organization_id == organization_id,
            DomainEvent.is_active.is_(True),
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_organization(
        self,
        organization_id: UUID,
        topic: Optional[str],
        aggregate_type: Optional[str],
        status: Optional[str],
        search: Optional[str],
        page: int,
        page_size: int,
    ) -> Tuple[List[DomainEvent], int]:
        stmt = select(DomainEvent).where(
            DomainEvent.organization_id == organization_id,
            DomainEvent.is_active.is_(True),
        )
        if topic:
            stmt = stmt.where(DomainEvent.topic == topic)
        if aggregate_type:
            stmt = stmt.where(DomainEvent.aggregate_type == aggregate_type)
        if status:
            stmt = stmt.where(DomainEvent.status == status)
        if search:
            term = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    DomainEvent.title.ilike(term),
                    DomainEvent.description.ilike(term),
                    DomainEvent.event_type.ilike(term),
                )
            )
        stmt = stmt.order_by(DomainEvent.created_at.desc())
        return await self.get_paginated(stmt, page, page_size)

    async def list_pending(self, organization_id: UUID, limit: int = 100) -> list[DomainEvent]:
        stmt = (
            select(DomainEvent)
            .where(
                DomainEvent.organization_id == organization_id,
                DomainEvent.is_active.is_(True),
                DomainEvent.status == "pending",
            )
            .order_by(DomainEvent.created_at.asc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
