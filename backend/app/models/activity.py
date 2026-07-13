"""
<<<<<<< HEAD
Activity Timeline Model
Stores auditable business events that should appear in the CRM timeline.
"""
import uuid
from typing import Optional

from sqlalchemy import Boolean, ForeignKey, JSON, String, Text
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

=======
Activity Model — The Event Timeline Backbone
═══════════════════════════════════════════════════════════════════════════════
Every important action in the CRM is recorded as an Activity.
This creates the immutable audit trail (Activity Timeline) that appears on
the Contact / Deal / Lead detail page.

Design Principle (from PULSE spec):
  "Every touch is an event — calls, emails, meetings, status changes all
   become timeline entries, so nothing about a deal's history is ever
   lost or hidden."

Activity Types (ActivityType enum):
  System types — auto-created by service layer hooks when state changes.
  Manual types — created by users (log a call, schedule a meeting, add note).
  AI types     — created by the AI/ML track (lead score, recommendation).

Polymorphic Entity Linking
───────────────────────────
  Activities are attached to ONE of: Lead, Deal, Contact, or Company.
  We use a simple entity_type + entity_id pattern (not SQLAlchemy polymorphic)
  so the AI/ML track and the Gmail sync can insert activities without needing
  to know which concrete FK to use.

  entity_type: ActivityEntityType enum  (lead | deal | contact | company)
  entity_id:   UUID of the target record

  Concrete FK columns (lead_id, deal_id, contact_id, company_id) are also
  stored for indexed lookups and to support JOIN queries in reports.
"""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TenantMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
    from app.models.user import User
    from app.models.lead import Lead
    from app.models.deal import Deal
    from app.models.contact import Contact
    from app.models.company import Company


class Activity(Base, TenantMixin):
    """
    A single immutable timeline event.

    Every service method that changes significant state calls
    ActivityService.record() to create one of these.  The AI/ML
    track reads this table directly for scoring and summarisation.
    """
    __tablename__ = "activities"
    __table_args__ = (
        # Composite index to efficiently fetch all activities for
        # a given entity (used by the timeline API)
        Index(
            "ix_activities_entity",
            "entity_type",
            "entity_id",
        ),
        # Index for timeline sorted by time per organisation
        Index(
            "ix_activities_org_created",
            "organization_id",
            "created_at",
        ),
    )

    # ── Type & description ────────────────────────────────────────────────────
    activity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="ActivityType enum value",
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Short human-readable summary shown on the timeline card",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Full details, e.g. call notes, email body excerpt",
    )

    # ── Polymorphic entity link ───────────────────────────────────────────────
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="ActivityEntityType: lead | deal | contact | company",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
        comment="UUID of the target Lead / Deal / Contact / Company",
    )

    # ── Concrete FK shortcuts (for indexed lookups) ───────────────────────────
    lead_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("leads.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    deal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("deals.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    company_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Who performed the action ──────────────────────────────────────────────
    performed_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who triggered this activity (null = system / AI)",
    )

    # ── Scheduled activities ──────────────────────────────────────────────────
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="For future-dated tasks/meetings",
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ── Metadata (AI / Gmail integration use these fields) ───────────────────
    metadata_json: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="JSON blob for AI scores, email IDs, call duration, etc.",
    )

    # ── Tenancy FK ────────────────────────────────────────────────────────────
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
    organization_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
<<<<<<< HEAD
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
