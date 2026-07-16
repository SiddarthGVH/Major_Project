"""
Models package.
Import all models here so Alembic auto-detects them and SQLAlchemy mapper
registration works correctly.
"""

from app.models.activity import ActivityTimeline  # noqa: F401
from app.models.company import Company  # noqa: F401
from app.models.contact import Contact  # noqa: F401
from app.models.email import GmailConnection, Email  # noqa: F401
from app.models.event import DomainEvent  # noqa: F401

