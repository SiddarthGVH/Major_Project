"""
Activity Repository
All database access for the Activity (Timeline) entity.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.repositories.base import BaseRepository
from app.utils.enums import ActivityType, ActivityEntityType


class ActivityRepository(BaseRepository[Activity]):

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Activity, db)

    # ── Create a timeline event ───────────────────────────────────────────────

    async def record(
        self,
        organization_id: UUID,
        activity_type: ActivityType,
        entity_type: ActivityEntityType,
        entity_id: UUID,
        title: str,
        description: Optional[str] = None,
        performed_by_id: Optional[UUID] = None,
        created_by: Optional[UUID] = None,
        lead_id: Optional[UUID] = None,
        deal_id: Optional[UUID] = None,
        contact_id: Optional[UUID] = None,
        company_id: Optional[UUID] = None,
        metadata_json: Optional[str] = None,
    ) -> Activity:
        """
        Convenience method used by service layer hooks.
        Automatically sets both the polymorphic entity_id and the
        specific concrete FK column.
        """
        return await self.create(
            organization_id=organization_id,
            created_by=created_by or performed_by_id,
            activity_type=activity_type.value,
            entity_type=entity_type.value,
            entity_id=entity_id,
            title=title,
            description=description,
            performed_by_id=performed_by_id,
            lead_id=lead_id,
            deal_id=deal_id,
            contact_id=contact_id,
            company_id=company_id,
            metadata_json=metadata_json,
        )

    # ── Timeline for a specific entity ───────────────────────────────────────

    async def get_timeline(
        self,
        organization_id: UUID,
        entity_type: ActivityEntityType,
        entity_id: UUID,
        page: int,
        page_size: int,
    ) -> Tuple[List[Activity], int]:
        """
        Fetch paginated timeline for one entity (lead/deal/contact/company).
        Ordered by created_at DESC (newest first).
        """
        stmt = (
            select(Activity)
            .where(
                Activity.organization_id == organization_id,
                Activity.entity_type == entity_type.value,
                Activity.entity_id == entity_id,
            )
            .order_by(Activity.created_at.desc())
        )
        return await self.get_paginated(stmt, page, page_size)

    # ── Full org timeline ─────────────────────────────────────────────────────

    async def get_org_timeline(
        self,
        organization_id: UUID,
        activity_type: Optional[ActivityType],
        entity_type: Optional[ActivityEntityType],
        page: int,
        page_size: int,
    ) -> Tuple[List[Activity], int]:
        stmt = (
            select(Activity)
            .where(Activity.organization_id == organization_id)
        )
        if activity_type:
            stmt = stmt.where(Activity.activity_type == activity_type.value)
        if entity_type:
            stmt = stmt.where(Activity.entity_type == entity_type.value)
        stmt = stmt.order_by(Activity.created_at.desc())
        return await self.get_paginated(stmt, page, page_size)
