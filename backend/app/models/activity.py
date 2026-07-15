"""
Activity Timeline Model
Stores auditable business events that should appear in the CRM timeline.
"""
import uuid
from typing import Optional

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, TenantMixin


class ActivityTimeline(Base, TenantMixin):
    __tablename__ = "activity_timeline_events"

    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    def __repr__(self) -> str:
        return f"<ActivityTimeline id={self.id} entity={self.entity_type!r} action={self.action!r}>"
=======

    # ── Relationships ─────────────────────────────────────────────────────────
    organization: Mapped["Organization"] = relationship(
        "Organization", lazy="select"
    )
    performed_by: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[performed_by_id], lazy="select"
    )
    lead: Mapped[Optional["Lead"]] = relationship(
        "Lead", foreign_keys=[lead_id], lazy="select"
    )
    deal: Mapped[Optional["Deal"]] = relationship(
        "Deal", foreign_keys=[deal_id], lazy="select"
    )
    contact: Mapped[Optional["Contact"]] = relationship(
        "Contact", foreign_keys=[contact_id], lazy="select"
    )
    company: Mapped[Optional["Company"]] = relationship(
        "Company", foreign_keys=[company_id], lazy="select"
    )

    def __repr__(self) -> str:
        return (
            f"<Activity id={self.id} type={self.activity_type!r} "
            f"entity={self.entity_type}:{self.entity_id}>"
        )
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
