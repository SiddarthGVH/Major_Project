"""
Contact Management Service
"""
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import DuplicateException, NotFoundException, BusinessRuleException
from app.core.logging import get_logger
from app.models.contact import Contact
from app.repositories.contact_repository import ContactRepository
from app.repositories.company_repository import CompanyRepository
from app.schemas.contact import ContactCreateRequest, ContactUpdateRequest

logger = get_logger(__name__)


class ContactService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = ContactRepository(db)
        self.company_repo = CompanyRepository(db)

    async def create(
        self,
        payload: ContactCreateRequest,
        organization_id: UUID,
        created_by: UUID,
    ) -> Contact:
        # Email uniqueness per org
        existing = await self.repo.get_by_email_in_org(
            str(payload.email), organization_id
        )
        if existing:
            raise DuplicateException("Contact", "email", str(payload.email))

        # Validate company belongs to same org
        if payload.company_id:
            company = await self.company_repo.get_active_by_id(
                payload.company_id, organization_id
            )
            if not company:
                raise BusinessRuleException(
                    f"Company '{payload.company_id}' not found in your organization."
                )

        contact = await self.repo.create(
            **payload.model_dump(exclude_none=True),
            organization_id=organization_id,
            created_by=created_by,
        )
        logger.info("Contact created", extra={"contact_id": str(contact.id)})
        return contact

    async def list(
        self,
        organization_id: UUID,
        search: Optional[str],
        company_id: Optional[UUID],
        page: int,
        page_size: int,
    ) -> Tuple[List[Contact], int]:
        return await self.repo.list_by_organization(
            organization_id, search, company_id, page, page_size
        )

    async def get(self, contact_id: UUID, organization_id: UUID) -> Contact:
        contact = await self.repo.get_active_by_id(contact_id, organization_id)
        if not contact:
            raise NotFoundException("Contact", contact_id)
        return contact

    async def update(
        self,
        contact_id: UUID,
        organization_id: UUID,
        payload: ContactUpdateRequest,
    ) -> Contact:
        contact = await self.get(contact_id, organization_id)

        update_data = payload.model_dump(exclude_none=True)

        # Check email uniqueness on email change
        if "email" in update_data and update_data["email"] != contact.email:
            existing = await self.repo.get_by_email_in_org(
                update_data["email"], organization_id
            )
            if existing and existing.id != contact_id:
                raise DuplicateException("Contact", "email", update_data["email"])

        # Validate company
        if "company_id" in update_data and update_data["company_id"]:
            company = await self.company_repo.get_active_by_id(
                update_data["company_id"], organization_id
            )
            if not company:
                raise BusinessRuleException("Company not found in your organization.")

        await self.repo.update(contact, **update_data)
        return contact

    async def delete(self, contact_id: UUID, organization_id: UUID) -> None:
        contact = await self.get(contact_id, organization_id)
        await self.repo.soft_delete(contact)
        logger.info("Contact deleted", extra={"contact_id": str(contact_id)})
