"""
Models package — import all models here so Alembic can discover them.
"""
from app.models.organization import Organization  # noqa: F401
from app.models.role import Role, Permission, RolePermission  # noqa: F401
from app.models.user import User, UserRole  # noqa: F401
from app.models.company import Company  # noqa: F401
from app.models.contact import Contact  # noqa: F401
from app.models.lead import Lead  # noqa: F401
from app.models.deal import Deal  # noqa: F401
from app.models.activity import ActivityTimeline  # noqa: F401
