"""
Domain event service.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.event import DomainEvent
from app.repositories.event_repository import EventRepository
from app.services.event_bus import EventEnvelope, event_bus


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
        event = await self.repo.create(
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
            processed_at=datetime.now(timezone.utc) if status == "processed" else None,
        )
        await event_bus.publish(
            EventEnvelope(
                event_id=event.id,
                organization_id=organization_id,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                event_type=event_type,
                topic=topic,
                title=title,
                description=description,
                payload=payload,
                source=source,
                status=status,
                created_at=event.created_at,
            )
        )
        return event

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

    async def replay(self, organization_id: UUID, event_id: Optional[UUID] = None) -> list[DomainEvent]:
        if event_id:
            event = await self.repo.get_active_by_id(event_id, organization_id)
            events = [event] if event else []
        else:
            events = await self.repo.list_pending(organization_id)
        envelopes = [
            EventEnvelope(
                event_id=event.id,
                organization_id=event.organization_id,
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                event_type=event.event_type,
                topic=event.topic,
                title=event.title,
                description=event.description,
                payload=event.payload,
                source=event.source,
                status=event.status,
                created_at=event.created_at,
            )
            for event in events
            if event is not None
        ]
        await event_bus.replay(envelopes)
        for event in events:
            if event is not None:
                event.status = "processed"
                event.processed_at = datetime.now(timezone.utc)
                self.db.add(event)
        await self.db.flush()
        return [event for event in events if event is not None]
