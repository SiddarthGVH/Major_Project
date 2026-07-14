"""
KALNET PULSE CRM — FastAPI Application Factory
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import PulseCRMException
from app.middlewares.exception_handler import (
    generic_exception_handler,
    pulse_exception_handler,
    validation_exception_handler,
)
from app.middlewares.logging import RequestLoggingMiddleware
from app.middlewares.request_id import RequestIDMiddleware
from app.api.v1.router import api_router

# ── Bootstrap logging before anything else ───────────────────────────────────
setup_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)
logger = get_logger(__name__)


# ── Lifespan (startup / shutdown hooks) ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Starting %s v%s [%s]",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
    )
    yield
    logger.info("Application shutdown complete.")


# ── Application factory ───────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=settings.DOCS_URL if not settings.is_production else None,
        redoc_url=settings.REDOC_URL if not settings.is_production else None,
        openapi_url=settings.OPENAPI_URL if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters — first registered = outermost) ─────────────
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.cors_methods_list,
        allow_headers=settings.cors_headers_list,
    )

    # ── Exception Handlers ────────────────────────────────────────────────────
    app.add_exception_handler(PulseCRMException, pulse_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": settings.DOCS_URL,
        }

    return app


app = create_app()
