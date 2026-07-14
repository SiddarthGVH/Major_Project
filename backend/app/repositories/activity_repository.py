"""
Activity Timeline Repository
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import ActivityTimeline
from app.repositories.base import BaseRepository


class ActivityTimelineRepository(BaseRepository[ActivityTimeline]):
    def __init__(self, db: AsyncSession) -> None:
        super().__init__(ActivityTimeline, db)

    def _base_query(self, organization_id: UUID):
        return select(ActivityTimeline).where(ActivityTimeline.organization_id == organization_id)

    async def list_by_organization(
        self,
        organization_id: UUID,
        entity_type: Optional[str],
        entity_id: Optional[UUID],
        action: Optional[str],
        search: Optional[str],
        page: int,
        page_size: int,
    ) -> Tuple[List[ActivityTimeline], int]:
        stmt = self._base_query(organization_id)

        if entity_type:
            stmt = stmt.where(ActivityTimeline.entity_type == entity_type)
        if entity_id:
            stmt = stmt.where(ActivityTimeline.entity_id == entity_id)
        if action:
            stmt = stmt.where(ActivityTimeline.action == action)
        if search:
            term = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    ActivityTimeline.title.ilike(term),
                    ActivityTimeline.description.ilike(term),
                    ActivityTimeline.action.ilike(term),
                    ActivityTimeline.entity_type.ilike(term),
                )
            )

        stmt = stmt.order_by(ActivityTimeline.created_at.desc())
        return await self.get_paginated(stmt, page, page_size)
