"""
Async SQLAlchemy 2.0 Database Connection
- Creates the async engine with connection pooling
- Provides an async session factory
- Exposes get_db() dependency for FastAPI routes
"""
from typing import AsyncGenerator

from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _normalize_database_url(raw_url: str) -> str:
    url = make_url(raw_url)
    if url.drivername == 'postgresql':
        url = url.set(drivername='postgresql+asyncpg')
    return str(url)

url = make_url(_normalize_database_url(settings.DATABASE_URL))
engine_kwargs = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,
}
if url.drivername.startswith("sqlite"):
    engine_kwargs["poolclass"] = NullPool
else:
    engine_kwargs.update(
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
        pool_timeout=settings.DATABASE_POOL_TIMEOUT,
        pool_recycle=settings.DATABASE_POOL_RECYCLE,
    )

engine = create_async_engine(_normalize_database_url(settings.DATABASE_URL), **engine_kwargs)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
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
    try:
        from sqlalchemy import text

        async with AsyncSessionFactory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        return False

