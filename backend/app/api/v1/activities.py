"""
Manual activity routes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import CurrentUser, DBSession, require_permission
from app.schemas.activity import ActivityTimelineCreateRequest, ActivityTimelineResponse
from app.schemas.common import StandardResponse
from app.services.activity_service import ActivityService

router = APIRouter()


@router.post(
    "",
    response_model=StandardResponse[ActivityTimelineResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a manual activity",
    dependencies=[Depends(require_permission("activity:create"))],
)
async def create_activity(payload: ActivityTimelineCreateRequest, current_user: CurrentUser, db: DBSession) -> dict:
    service = ActivityService(db)
    activity = await service.create(current_user.organization_id, current_user.id, payload)
    return {"success": True, "message": "Activity created.", "data": ActivityTimelineResponse.model_validate(activity)}
