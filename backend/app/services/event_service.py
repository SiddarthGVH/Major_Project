"""
Domain event service.
"""
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event import DomainEvent
from app.repositories.event_repository import EventRepository


class EventService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self.repo = EventRepository(db)

    async def publish(
        self,
        organization_id: UUID,
        created_by: Optional[UUID],
        aggregate_type: str,
        aggregate_id: Optional[UUID],
        event_type: str,
        topic: str,
        title: str,
        description: Optional[str] = None,
        payload: Optional[dict] = None,
        source: Optional[str] = None,
        status: str = "pending",
    ) -> DomainEvent:
        return await self.repo.create(
            organization_id=organization_id,
            created_by=created_by,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type=event_type,
            topic=topic,
            title=title,
            description=description,
            payload=payload,
            source=source,
            status=status,
        )

    async def publish_event(
        self,
        organization_id: UUID,
        created_by: Optional[UUID],
        aggregate_type: str,
        aggregate_id: Optional[UUID],
        event_type: str,
        topic: str,
        title: str,
        description: Optional[str] = None,
        payload: Optional[dict] = None,
        source: Optional[str] = None,
        status: str = "pending",
    ) -> DomainEvent:
        """Alias that reads a little more clearly in integration code."""
        return await self.publish(
            organization_id=organization_id,
            created_by=created_by,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            event_type=event_type,
            topic=topic,
            title=title,
            description=description,
            payload=payload,
            source=source,
            status=status,
        )

    async def list(
        self,
        organization_id: UUID,
        topic: Optional[str],
        aggregate_type: Optional[str],
        status: Optional[str],
        search: Optional[str],
        page: int,
        page_size: int,
    ) -> Tuple[list[DomainEvent], int]:
        return await self.repo.list_by_organization(
            organization_id, topic, aggregate_type, status, search, page, page_size
        )
