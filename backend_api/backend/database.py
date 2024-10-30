import contextlib
from os import environ
from typing import Any, AsyncIterator

# SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncConnection,
)
from sqlalchemy.orm import DeclarativeBase

DB_IP = environ.get("DATABASE_HOST", "localhost")
USER = environ.get("DATABASE_USER", "postgres")
PASSWORD = environ.get("DATABASE_PASSWORD", "yyyyyy")
DATABASE_NAME = environ.get("DATABASE_NAME", "AIaaS Backend")
DATABASE_PORT = environ.get("DATABASE_PORT", "5432")
DB_URL = f'postgresql+asyncpg://{USER}:{PASSWORD}@{DB_IP}:{DATABASE_PORT}/{DATABASE_NAME}'

print(DB_URL)

engine = create_async_engine(DB_URL)

Base: DeclarativeBase = declarative_base()

class DatabaseSessionManager:
    def __init__(self, host: str, engine_kwargs: dict[str, Any] = {}):
        self._engine = create_async_engine(host, **engine_kwargs)
        self._sessionmaker = async_sessionmaker(autocommit=False, bind=self._engine, expire_on_commit=False)

    async def close(self):
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")
        await self._engine.dispose()

        self._engine = None
        self._sessionmaker = None

    @contextlib.asynccontextmanager
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        if self._engine is None:
            raise Exception("DatabaseSessionManager is not initialized")

        async with self._engine.begin() as connection:
            try:
                yield connection
            except Exception:
                await connection.rollback()
                raise

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._sessionmaker is None:
            raise Exception("DatabaseSessionManager is not initialized")

        session = self._sessionmaker()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


sessionmanager = DatabaseSessionManager(DB_URL)


