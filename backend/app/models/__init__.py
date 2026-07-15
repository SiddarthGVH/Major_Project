"""
Models package.
Import all models here so Alembic auto-detects them and SQLAlchemy mapper
registration works correctly.
"""

from app.models.activity import ActivityTimeline  # noqa: F401
from app.models.company import Company  # noqa: F401
from app.models.contact import Contact  # noqa: F401
from app.models.deal import Deal  # noqa: F401
<<<<<<< HEAD
from app.models.email import Email, GmailConnection  # noqa: F401
from app.models.event import DomainEvent  # noqa: F401
from app.models.lead import Lead  # noqa: F401
from app.models.organization import Organization  # noqa: F401
from app.models.pipeline import PipelineStage  # noqa: F401
from app.models.role import Permission, Role, RolePermission  # noqa: F401
from app.models.user import User, UserRole  # noqa: F401
=======
from app.models.activity import ActivityTimeline  # noqa: F401
from app.models.pipeline import PipelineStage  # noqa: F401
from app.models.email import GmailConnection, Email  # noqa: F401
from app.models.event import DomainEvent  # noqa: F401
>>>>>>> 8c70aea23112d0ec090a696619d810cd6c7fb7a2
