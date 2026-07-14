"""
Async SQLAlchemy 2.0 Database Connection
- Creates the async engine with connection pooling
- Provides an async session factory
- Exposes get_db() dependency for FastAPI routes
"""
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────
# NullPool is used in test environments to avoid connection reuse issues.
# In production we rely on asyncpg's built-in pool.

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,                        # SQL query logging in debug mode
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    pool_recycle=settings.DATABASE_POOL_RECYCLE,
    pool_pre_ping=True,                         # Validate connections before checkout
)

# ── Session Factory ───────────────────────────────────────────────────────────
AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # Keep objects accessible after commit without re-query
    autocommit=False,
    autoflush=False,
)


# ── FastAPI Dependency ────────────────────────────────────────────────────────

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session for each request.
    Automatically commits on success, rolls back on any exception.
    Usage:
        @router.get("/")
        async def endpoint(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connection() -> bool:
    """
    Health-check helper — returns True if the DB is reachable.
    Called from the /health endpoint.
    """
    try:
        from sqlalchemy import text
        async with AsyncSessionFactory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False
