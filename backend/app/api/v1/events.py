"""
Domain event routes.
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query

from app.api.deps import CurrentUser, DBSession, require_permission
from app.schemas.common import PaginatedResponse, StandardResponse
from app.schemas.event import DomainEventResponse
from app.services.event_service import EventService

router = APIRouter()


@router.get(
    "",
    response_model=StandardResponse[PaginatedResponse[DomainEventResponse]],
    summary="List domain events",
    dependencies=[Depends(require_permission("activity:read"))],
)
async def list_events(
    current_user: CurrentUser,
    db: DBSession,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    topic: Optional[str] = Query(default=None),
    aggregate_type: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    search: Optional[str] = Query(default=None),
) -> dict:
    svc = EventService(db)
    events, total = await svc.list(
        current_user.organization_id,
        topic,
        aggregate_type,
        status,
        search,
        page,
        page_size,
    )
    paginated = PaginatedResponse.create(
        data=[DomainEventResponse.model_validate(event) for event in events],
        total=total,
        page=page,
        page_size=page_size,
    )
    return {"success": True, "message": "OK", "data": paginated}
