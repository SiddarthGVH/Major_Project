"""
API v1 Router
Aggregates all domain routers under /api/v1.
"""
from fastapi import APIRouter

from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router
from app.api.v1.organizations import router as orgs_router
from app.api.v1.companies import router as companies_router
from app.api.v1.contacts import router as contacts_router
from app.api.v1.leads import router as leads_router
from app.api.v1.deals import router as deals_router
from app.api.v1.activity import router as activity_router
from app.api.v1.health import router as health_router

api_router = APIRouter()

api_router.include_router(health_router, prefix="/health", tags=["Health"])
api_router.include_router(auth_router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users_router, prefix="/users", tags=["Users"])
api_router.include_router(orgs_router, prefix="/organizations", tags=["Organizations"])
api_router.include_router(companies_router, prefix="/companies", tags=["Companies"])
api_router.include_router(contacts_router, prefix="/contacts", tags=["Contacts"])
api_router.include_router(leads_router, prefix="/leads", tags=["Leads"])
api_router.include_router(deals_router, prefix="/deals", tags=["Deals"])
api_router.include_router(activity_router, prefix="/activity", tags=["Activity"])
