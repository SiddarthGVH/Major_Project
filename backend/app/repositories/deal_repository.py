"""
Deal Repository
All database access for the Deal entity.
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.deal import Deal
from app.repositories.base import BaseRepository
from app.utils.enums import DealStage


class DealRepository(BaseRepository[Deal]):

    def __init__(self, db: AsyncSession) -> None:
        super().__init__(Deal, db)

    # ── Base query — scoped to org + not deleted ──────────────────────────────
    def _base_query(self, organization_id: UUID):
        return (
            select(Deal)
            .where(
                Deal.organization_id == organization_id,
                Deal.is_deleted == False,
            )
        )

    # ── Lookups ───────────────────────────────────────────────────────────────

    async def get_active_by_id(
        self, deal_id: UUID, organization_id: UUID
    ) -> Optional[Deal]:
        stmt = self._base_query(organization_id).where(Deal.id == deal_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_lead_id(
        self, lead_id: UUID, organization_id: UUID
    ) -> Optional[Deal]:
        """Used to prevent duplicate conversions of the same lead."""
        stmt = self._base_query(organization_id).where(Deal.lead_id == lead_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    # ── Paginated list with filters ───────────────────────────────────────────

    async def list_by_organization(
        self,
        organization_id: UUID,
        search: Optional[str],
        stage: Optional[DealStage],
        owner_id: Optional[UUID],
        company_id: Optional[UUID],
        contact_id: Optional[UUID],
        page: int,
        page_size: int,
    ) -> Tuple[List[Deal], int]:
        stmt = self._base_query(organization_id)

        if stage:
            stmt = stmt.where(Deal.stage == stage.value)
        if owner_id:
            stmt = stmt.where(Deal.owner_id == owner_id)
        if company_id:
            stmt = stmt.where(Deal.company_id == company_id)
        if contact_id:
            stmt = stmt.where(Deal.contact_id == contact_id)
        if search:
            term = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    Deal.title.ilike(term),
                    Deal.description.ilike(term),
                )
            )

        # Most recent deals first (newest on top of Kanban)
        stmt = stmt.order_by(Deal.created_at.desc())
        return await self.get_paginated(stmt, page, page_size)

    # ── Kanban — group by stage ───────────────────────────────────────────────

    async def list_by_stage(
        self,
        organization_id: UUID,
        stage: DealStage,
    ) -> List[Deal]:
        stmt = (
            self._base_query(organization_id)
            .where(Deal.stage == stage.value)
            .order_by(Deal.updated_at.desc())
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())
