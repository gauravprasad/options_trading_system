from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from typing import Generator
import contextlib

from app.config.settings import settings
from app.utils.logger import db_logger

# Create database engine with connection pooling
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
    echo=settings.debug,  # Log SQL queries in debug mode
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db_logger.error("Database session error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()

@contextlib.contextmanager
def get_db_context():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db_logger.error("Database context error", error=str(e))
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all tables in the database."""
    try:
        Base.metadata.create_all(bind=engine)
        db_logger.info("Database tables created successfully")
    except Exception as e:
        db_logger.error("Failed to create database tables", error=str(e))
        raise

def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        db_logger.info("Database connection successful")
        return True
    except Exception as e:
        db_logger.error("Database connection failed", error=str(e))
        return False