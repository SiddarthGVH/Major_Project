"""
SQLAlchemy Declarative Base and TimestampMixin
All domain models inherit from Base.
TimestampMixin provides audit fields (created_at, updated_at, etc.).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """
    Project-wide SQLAlchemy declarative base.
    All models are registered here for Alembic auto-detection.
    """
    pass


class AuditMixin:
    """
    Adds standard audit columns to every table:
      - id              UUID primary key
      - created_at      Timestamp of record creation
      - updated_at      Timestamp of last update (auto-maintained)
      - is_active       Soft-enable/disable flag
    """
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        server_default=func.now(),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )


class TenantMixin(AuditMixin):
    """
    Extends AuditMixin with multi-tenancy columns:
      - organization_id   The owning tenant
      - created_by        UUID of the user who created this record
    These are intentionally nullable so the base mixin can be shared
    across models that set them as actual FK columns in their own table definitions.
    """
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
