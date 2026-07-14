"""
Activity Timeline Schemas (Pydantic V2)
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class ActivityTimelineResponse(BaseModel):
    id: UUID
    entity_type: str
    entity_id: UUID
    action: str
    title: str
    description: Optional[str]
    payload: Optional[dict]
    organization_id: UUID
    created_by: Optional[UUID]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}