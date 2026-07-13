"""
Email detail routes.
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, DBSession, require_permission
from app.schemas.common import StandardResponse
from app.schemas.email import EmailDetailResponse
from app.services.email_service import EmailService

router = APIRouter()


@router.get(
    "/{email_id}",
    response_model=StandardResponse[EmailDetailResponse],
    summary="Get an email by id",
    dependencies=[Depends(require_permission("email:read"))],
)
async def get_email(email_id: UUID, current_user: CurrentUser, db: DBSession) -> dict:
    service = EmailService(db)
    email = await service.get_by_id_response(current_user.organization_id, email_id)
    return {"success": True, "message": "OK", "data": email}
