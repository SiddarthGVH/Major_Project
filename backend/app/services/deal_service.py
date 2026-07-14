"""
Deal Management Service
"""
from decimal import Decimal
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import BusinessRuleException, ConflictException, NotFoundException
from app.core.logging import get_logger
from app.models.deal import Deal
from app.repositories.activity_repository import ActivityTimelineRepository
from app.repositories.company_repository import CompanyRepository
from app.repositories.contact_repository import ContactRepository
from app.repositories.deal_repository import DealRepository
from app.repositories.lead_repository import LeadRepository
from app.repositories.user_repository import UserRepository
from app.schemas.deal import DealCreateRequest, DealUpdateRequest
from app.utils.enums import DealSortField, DealStatus, LeadStatus, SortOrder

logger = get_logger(__name__)


class DealService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = DealRepository(db)
        self.activity_repo = ActivityTimelineRepository(db)
        self.company_repo = CompanyRepository(db)
        self.contact_repo = ContactRepository(db)
        self.lead_repo = LeadRepository(db)
        self.user_repo = UserRepository(db)

    async def create(self, payload: DealCreateRequest, organization_id: UUID, created_by: UUID) -> Deal:
        await self._validate_relations(
            organization_id,
            payload.company_id,
            payload.contact_id,
            payload.lead_id,
            payload.owner_id,
        )

        if payload.lead_id:
            existing = await self.repo.get_by_lead_id_in_org(payload.lead_id, organization_id)
            if existing:
                raise ConflictException("A deal already exists for this lead.")

        deal = await self.repo.create(
            **payload.model_dump(exclude_none=True),
            organization_id=organization_id,
            created_by=created_by,
        )
        logger.info("Deal created", extra={"deal_id": str(deal.id)})
        return deal

    async def list(
        self,
        organization_id: UUID,
        search: Optional[str],
        status: Optional[DealStatus],
        owner_id: Optional[UUID],
        company_id: Optional[UUID],
        contact_id: Optional[UUID],
        lead_id: Optional[UUID],
        min_amount: Optional[Decimal],
        max_amount: Optional[Decimal],
        sort_by: Optional[DealSortField],
        sort_order: SortOrder,
        page: int,
        page_size: int,
    ) -> Tuple[List[Deal], int]:
        return await self.repo.list_by_organization(
            organization_id,
            search,
            status,
            owner_id,
            company_id,
            contact_id,
            lead_id,
            min_amount,
            max_amount,
            sort_by,
            sort_order,
            page,
            page_size,
        )

    async def get(self, deal_id: UUID, organization_id: UUID) -> Deal:
        deal = await self.repo.get_active_by_id(deal_id, organization_id)
        if not deal:
            raise NotFoundException("Deal", deal_id)
        return deal

    async def update(self, deal_id: UUID, organization_id: UUID, payload: DealUpdateRequest) -> Deal:
        deal = await self.get(deal_id, organization_id)
        update_data = payload.model_dump(exclude_none=True)

        await self._validate_relations(
            organization_id,
            update_data.get("company_id"),
            update_data.get("contact_id"),
            update_data.get("lead_id"),
            update_data.get("owner_id"),
        )

        lead_id = update_data.get("lead_id")
        if lead_id:
            existing = await self.repo.get_by_lead_id_in_org(lead_id, organization_id)
            if existing and existing.id != deal_id:
                raise ConflictException("A deal already exists for this lead.")

        await self.repo.update(deal, **update_data)
        return deal

    async def delete(self, deal_id: UUID, organization_id: UUID) -> None:
        deal = await self.get(deal_id, organization_id)
        await self.repo.soft_delete(deal)
        logger.info("Deal deleted", extra={"deal_id": str(deal_id)})

    async def convert_from_lead(self, lead_id: UUID, organization_id: UUID, created_by: UUID) -> Deal:
        tx_context = self.db.begin_nested() if self.db.in_transaction() else self.db.begin()
        async with tx_context:
            lead = await self.lead_repo.get_active_by_id(lead_id, organization_id)
            if not lead:
                raise NotFoundException("Lead", lead_id)

            if lead.status == LeadStatus.CONVERTED.value:
                raise ConflictException("Lead has already been converted into a deal.")

            existing = await self.repo.get_by_lead_id_in_org(lead_id, organization_id)
            if existing:
                raise ConflictException("Lead has already been converted into a deal.")

            deal = await self.repo.create(
                name=lead.title,
                description=lead.description,
                status=DealStatus.OPEN.value,
                amount=lead.estimated_value,
                currency=lead.currency,
                probability=min(max((lead.score or 50), 0), 100),
                notes=lead.notes,
                owner_id=lead.owner_id,
                company_id=lead.company_id,
                contact_id=lead.contact_id,
                lead_id=lead.id,
                organization_id=organization_id,
                created_by=created_by,
            )

            await self.lead_repo.update(lead, status=LeadStatus.CONVERTED.value)

            await self.activity_repo.create(
                entity_type="deal",
                entity_id=deal.id,
                action="created_from_lead",
                title="Lead converted to deal",
                description=f"Lead '{lead.title}' was converted into deal '{deal.name}'.",
                payload={
                    "lead_id": str(lead.id),
                    "deal_id": str(deal.id),
                    "lead_title": lead.title,
                },
                organization_id=organization_id,
                created_by=created_by,
            )

            logger.info("Lead converted to deal", extra={"lead_id": str(lead.id), "deal_id": str(deal.id)})
            return deal

    async def _validate_relations(
        self,
        organization_id: UUID,
        company_id: Optional[UUID],
        contact_id: Optional[UUID],
        lead_id: Optional[UUID],
        owner_id: Optional[UUID],
    ) -> None:
        if company_id:
            company = await self.company_repo.get_active_by_id(company_id, organization_id)
            if not company:
                raise BusinessRuleException(f"Company '{company_id}' not found.")

        if contact_id:
            contact = await self.contact_repo.get_active_by_id(contact_id, organization_id)
            if not contact:
                raise BusinessRuleException(f"Contact '{contact_id}' not found.")

        if lead_id:
            lead = await self.lead_repo.get_active_by_id(lead_id, organization_id)
            if not lead:
                raise BusinessRuleException(f"Lead '{lead_id}' not found.")

        if owner_id:
            owner = await self.user_repo.get_by_id_with_roles(owner_id)
            if not owner or owner.organization_id != organization_id:
                raise BusinessRuleException(f"Owner (user) '{owner_id}' not found.")