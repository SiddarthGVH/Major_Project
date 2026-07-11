"""
Company Schemas (Pydantic V2)
"""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field, field_validator
import re


def _validate_url(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    if not re.match(r"^https?://", v):
        raise ValueError("URL must start with http:// or https://")
    return v


def _validate_phone(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    cleaned = re.sub(r"[\s\-\(\)\+\.]", "", v)
    if not cleaned.isdigit() or len(cleaned) < 7 or len(cleaned) > 15:
        raise ValueError("Invalid phone number format")
    return v


class CompanyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    domain: Optional[str] = Field(default=None, max_length=255)
    website: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(default=None, max_length=30)
    address: Optional[str] = None
    city: Optional[str] = Field(default=None, max_length=100)
    state: Optional[str] = Field(default=None, max_length=100)
    country: Optional[str] = Field(default=None, max_length=100)
    zip_code: Optional[str] = Field(default=None, max_length=20)
    industry: Optional[str] = Field(default=None, max_length=100)
    company_type: Optional[str] = Field(default=None, max_length=50)
    employee_count: Optional[int] = Field(default=None, ge=0)
    annual_revenue: Optional[str] = None
    linkedin_url: Optional[str] = Field(default=None, max_length=500)
    twitter_url: Optional[str] = Field(default=None, max_length=500)

    @field_validator("website", "linkedin_url", "twitter_url", mode="before")
    @classmethod
    def validate_url(cls, v):
        return _validate_url(v)

    @field_validator("phone", mode="before")
    @classmethod
    def validate_phone(cls, v):
        return _validate_phone(v)


class CompanyUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    domain: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    industry: Optional[str] = None
    company_type: Optional[str] = None
    employee_count: Optional[int] = Field(default=None, ge=0)
    annual_revenue: Optional[str] = None
    linkedin_url: Optional[str] = None
    twitter_url: Optional[str] = None

    @field_validator("website", "linkedin_url", "twitter_url", mode="before")
    @classmethod
    def validate_url(cls, v):
        return _validate_url(v)

    @field_validator("phone", mode="before")
    @classmethod
    def validate_phone(cls, v):
        return _validate_phone(v)


class CompanyResponse(BaseModel):
    id: UUID
    name: str
    domain: Optional[str]
    website: Optional[str]
    description: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: Optional[str]
    zip_code: Optional[str]
    industry: Optional[str]
    company_type: Optional[str]
    employee_count: Optional[int]
    annual_revenue: Optional[str]
    linkedin_url: Optional[str]
    twitter_url: Optional[str]
    owner_id: Optional[UUID]
    organization_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
