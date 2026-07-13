"""
RBAC Permission Registry
Defines all available permissions and their role assignments.
Used by both the seeder and the require_permission() dependency.
"""
from enum import Enum
from typing import Dict, Set


class Permission(str, Enum):
    """
    Exhaustive list of granular permissions.
    Format: <resource>:<action>
    """
    # ── Users ──────────────────────────────────────────────────────────────────
    USER_CREATE      = "user:create"
    USER_READ        = "user:read"
    USER_UPDATE      = "user:update"
    USER_DELETE      = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"
    USER_ACTIVATE    = "user:activate"
    USER_DEACTIVATE  = "user:deactivate"

    # ── Organizations ──────────────────────────────────────────────────────────
    ORG_CREATE = "org:create"
    ORG_READ   = "org:read"
    ORG_UPDATE = "org:update"
    ORG_DELETE = "org:delete"

    # ── Companies ─────────────────────────────────────────────────────────────
    COMPANY_CREATE = "company:create"
    COMPANY_READ   = "company:read"
    COMPANY_UPDATE = "company:update"
    COMPANY_DELETE = "company:delete"

    # ── Contacts ──────────────────────────────────────────────────────────────
    CONTACT_CREATE = "contact:create"
    CONTACT_READ   = "contact:read"
    CONTACT_UPDATE = "contact:update"
    CONTACT_DELETE = "contact:delete"

    # ── Leads ─────────────────────────────────────────────────────────────────
    LEAD_CREATE = "lead:create"
    LEAD_READ = "lead:read"
    LEAD_UPDATE = "lead:update"
    LEAD_DELETE = "lead:delete"
    LEAD_ASSIGN = "lead:assign"
    LEAD_CONVERT = "lead:convert"

    # ── Deals ────────────────────────────────────────────────────────────────
    DEAL_CREATE = "deal:create"
    DEAL_READ = "deal:read"
    DEAL_UPDATE = "deal:update"
    DEAL_DELETE = "deal:delete"

    # ── Pipeline ─────────────────────────────────────────────────────────────
    PIPELINE_READ = "pipeline:read"
    PIPELINE_UPDATE = "pipeline:update"

    # ── Activity ─────────────────────────────────────────────────────────────
    ACTIVITY_CREATE = "activity:create"
    ACTIVITY_READ = "activity:read"

    # ── Emails / Gmail ───────────────────────────────────────────────────────
    EMAIL_READ = "email:read"
    EMAIL_SYNC = "email:sync"
    GMAIL_CONNECT = "gmail:connect"

    # ── Dashboard ────────────────────────────────────────────────────────────
    DASHBOARD_READ = "dashboard:read"

    # ── AI-ready APIs ─────────────────────────────────────────────────────────
    AI_ACCESS = "ai:access"

    # ── Reports ───────────────────────────────────────────────────────────────
    REPORT_VIEW   = "report:view"
    REPORT_EXPORT = "report:export"

    # ── System ────────────────────────────────────────────────────────────────
    SYSTEM_ADMIN = "system:admin"


class Role(str, Enum):
    """Built-in RBAC roles."""
    ADMIN     = "admin"
    MANAGER   = "manager"
    SALES_REP = "sales_rep"


# ── Role → Permission mapping ──────────────────────────────────────────────────
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {

    # Admin inherits every permission
    Role.ADMIN: set(Permission),

    Role.MANAGER: {
        Permission.USER_READ,
        Permission.USER_CREATE,
        Permission.USER_UPDATE,
        Permission.USER_ACTIVATE,
        Permission.USER_DEACTIVATE,
        Permission.ORG_READ,
        Permission.ORG_UPDATE,
        Permission.COMPANY_CREATE,
        Permission.COMPANY_READ,
        Permission.COMPANY_UPDATE,
        Permission.COMPANY_DELETE,
        Permission.CONTACT_CREATE,
        Permission.CONTACT_READ,
        Permission.CONTACT_UPDATE,
        Permission.CONTACT_DELETE,
        Permission.LEAD_CREATE,
        Permission.LEAD_READ,
        Permission.LEAD_UPDATE,
        Permission.LEAD_DELETE,
        Permission.LEAD_ASSIGN,
        Permission.LEAD_CONVERT,
        Permission.DEAL_CREATE,
        Permission.DEAL_READ,
        Permission.DEAL_UPDATE,
        Permission.DEAL_DELETE,
        Permission.PIPELINE_READ,
        Permission.PIPELINE_UPDATE,
        Permission.ACTIVITY_CREATE,
        Permission.ACTIVITY_READ,
        Permission.EMAIL_READ,
        Permission.EMAIL_SYNC,
        Permission.GMAIL_CONNECT,
        Permission.DASHBOARD_READ,
        Permission.AI_ACCESS,
        Permission.REPORT_VIEW,
        Permission.REPORT_EXPORT,
    },

    Role.SALES_REP: {
        Permission.USER_READ,
        Permission.COMPANY_CREATE,
        Permission.COMPANY_READ,
        Permission.COMPANY_UPDATE,
        Permission.CONTACT_CREATE,
        Permission.CONTACT_READ,
        Permission.CONTACT_UPDATE,
        Permission.LEAD_CREATE,
        Permission.LEAD_READ,
        Permission.LEAD_UPDATE,
        Permission.LEAD_CONVERT,
        Permission.DEAL_CREATE,
        Permission.DEAL_READ,
        Permission.DEAL_UPDATE,
        Permission.PIPELINE_READ,
        Permission.ACTIVITY_READ,
        Permission.EMAIL_READ,
        Permission.EMAIL_SYNC,
        Permission.DASHBOARD_READ,
        Permission.REPORT_VIEW,
    },
}


def get_permissions_for_role(role: Role) -> list[str]:
    """Return list of permission strings for a given role."""
    return [p.value for p in ROLE_PERMISSIONS.get(role, set())]
