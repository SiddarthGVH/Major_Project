"""
Deal Model
<<<<<<< HEAD
Represents an active sales opportunity derived from a lead or created directly.
"""
import uuid
from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, Date, ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
=======
═══════════════════════════════════════════════════════════════════════════════
A Deal is a qualified sales opportunity converted from a Lead (or created
directly). It moves through Pipeline stages on the Kanban board.

Relationships
─────────────
  Organization  ──< Deal      (multi-tenant ownership)
  Lead          ──o Deal      (optional source lead)
  Contact       ──o Deal      (primary contact person)
  Company       ──o Deal      (associated company)
  User (owner)  ──o Deal      (assigned sales rep)

Pipeline Stages (DealStage enum)
──────────────────────────────────
  new → qualified → proposal → negotiation → won | lost

Design Decisions
─────────────────
  • stage is a plain string column (not FK) — enforced by DealStage enum
    in the service layer. No join needed to render the Kanban board.
  • lead_id is nullable — deals can be created directly without a prior lead.
  • Soft-delete follows the same pattern as all other models (is_deleted flag).
"""
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Numeric, String, Text
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TenantMixin

if TYPE_CHECKING:
    from app.models.organization import Organization
<<<<<<< HEAD
    from app.models.company import Company
    from app.models.contact import Contact
    from app.models.lead import Lead
    from app.models.user import User


class Deal(Base, TenantMixin):
    """Sales deal / opportunity record."""

    __tablename__ = "deals"
    __table_args__ = (
        UniqueConstraint("lead_id", name="uq_deal_lead_id"),
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="open", nullable=False, index=True)
    amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(15, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    expected_close_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True, index=True)
    probability: Mapped[int] = mapped_column(Integer, default=50, nullable=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

=======
    from app.models.lead import Lead
    from app.models.contact import Contact
    from app.models.company import Company
    from app.models.user import User
    from app.models.activity import Activity


class Deal(Base, TenantMixin):
    """
    An active sales opportunity derived from a converted Lead or entered
    manually by a sales representative.
    """
    __tablename__ = "deals"

    # ── Identity ──────────────────────────────────────────────────────────────
    title: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    stage: Mapped[str] = mapped_column(
        String(50),
        default="new",
        nullable=False,
        index=True,
        comment="DealStage enum value — validated in service layer",
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        default="medium",
        nullable=False,
        comment="DealPriority enum value",
    )

    # ── Value ─────────────────────────────────────────────────────────────────
    amount: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(15, 2),
        nullable=True,
        comment="Expected deal value",
    )
    currency: Mapped[str] = mapped_column(
        String(3), default="USD", nullable=False
    )
    probability: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Win probability 0–100",
    )

    # ── Dates ─────────────────────────────────────────────────────────────────
    expected_close_date: Mapped[Optional[date]] = mapped_column(
        Date, nullable=True
    )
    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Closure ───────────────────────────────────────────────────────────────
    close_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Soft delete ───────────────────────────────────────────────────────────
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # ── Foreign keys ─────────────────────────────────────────────────────────
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

    owner_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
=======
    # Nullable — a deal can exist without a prior lead (direct entry)
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
    lead_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("leads.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
<<<<<<< HEAD
    company_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
=======
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
    contact_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("contacts.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
<<<<<<< HEAD
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="deals", lazy="select"
    )
    owner: Mapped[Optional["User"]] = relationship(
        "User",
        foreign_keys=[owner_id],
        lazy="select",
    )
    company: Mapped[Optional["Company"]] = relationship("Company", lazy="select")
    contact: Mapped[Optional["Contact"]] = relationship("Contact", lazy="select")
    lead: Mapped[Optional["Lead"]] = relationship(
        "Lead", back_populates="deal", lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Deal id={self.id} name={self.name!r} status={self.status!r}>"
=======
    company_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    owner_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # ── Relationships ─────────────────────────────────────────────────────────
    organization: Mapped["Organization"] = relationship(
        "Organization", lazy="select"
    )
    lead: Mapped[Optional["Lead"]] = relationship(
        "Lead", lazy="select"
    )
    contact: Mapped[Optional["Contact"]] = relationship(
        "Contact", lazy="select"
    )
    company: Mapped[Optional["Company"]] = relationship(
        "Company", lazy="select"
    )
    owner: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[owner_id], lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Deal id={self.id} title={self.title!r} stage={self.stage!r}>"
>>>>>>> 2caa082038ab34692767356457dd0dab412d6960
