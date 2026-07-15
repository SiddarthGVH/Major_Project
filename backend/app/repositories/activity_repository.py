"""
Activity Timeline Repository
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

<<<<<<< HEAD
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import ActivityTimeline
from app.repositories.base import BaseRepository
from app.utils.enums import SortOrder


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
        sort_order: SortOrder = SortOrder.DESC,
        created_by: Optional[UUID] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Tuple[List[ActivityTimeline], int]:
        stmt = self._base_query(organization_id)

        if entity_type:
            stmt = stmt.where(ActivityTimeline.entity_type == entity_type)
        if entity_id:
            stmt = stmt.where(ActivityTimeline.entity_id == entity_id)
        if action:
            stmt = stmt.where(ActivityTimeline.action == action)
        if created_by:
            stmt = stmt.where(ActivityTimeline.created_by == created_by)
        if from_date:
            stmt = stmt.where(ActivityTimeline.created_at >= from_date)
        if to_date:
            stmt = stmt.where(ActivityTimeline.created_at <= to_date)
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

        sort_clause = asc(ActivityTimeline.created_at) if sort_order == SortOrder.ASC else desc(ActivityTimeline.created_at)
        stmt = stmt.order_by(sort_clause, desc(ActivityTimeline.id))
        return await self.get_paginated(stmt, page, page_size)
=======
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
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
