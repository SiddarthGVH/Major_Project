"""
Activity Timeline Service
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import ActivityTimeline
from app.repositories.activity_repository import ActivityTimelineRepository


class ActivityService:
    def __init__(self, db: AsyncSession) -> None:
        self.repo = ActivityTimelineRepository(db)

    async def list(
        self,
        organization_id: UUID,
        entity_type: Optional[str],
        entity_id: Optional[UUID],
        action: Optional[str],
        search: Optional[str],
        page: int,
        page_size: int,
    ) -> Tuple[List[ActivityTimeline], int]:
        return await self.repo.list_by_organization(
            organization_id=organization_id,
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            search=search,
            page=page,
            page_size=page_size,
        )