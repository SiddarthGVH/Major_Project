"""
Domain Enums
All string enums used across models, schemas, and services.
"""
from enum import Enum


class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    PROPOSAL_SENT = "proposal_sent"
    NEGOTIATION = "negotiation"
    WON = "won"
    LOST = "lost"


class LeadSource(str, Enum):
    WEBSITE = "website"
    REFERRAL = "referral"
    COLD_CALL = "cold_call"
    EMAIL_CAMPAIGN = "email_campaign"
    SOCIAL_MEDIA = "social_media"
    LINKEDIN = "linkedin"
    TRADE_SHOW = "trade_show"
    PARTNER = "partner"
    INBOUND = "inbound"
    OTHER = "other"


class CompanyType(str, Enum):
    PROSPECT = "prospect"
    CUSTOMER = "customer"
    PARTNER = "partner"
    COMPETITOR = "competitor"
    VENDOR = "vendor"
    OTHER = "other"


class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


class OrgPlan(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
