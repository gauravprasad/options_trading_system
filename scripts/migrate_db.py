#!/usr/bin/env python3
"""
Database migration script for the options trading system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from app.config.database import Base
from app.config.settings import settings
from app.models.database_models import *  # Import all models
from app.utils.logger import db_logger

def create_database():
    """Create the database if it doesn't exist."""
    try:
        # Parse the database URL to get database name
        db_url_parts = settings.database_url.split('/')
        db_name = db_url_parts[-1]
        base_url = '/'.join(db_url_parts[:-1])

        # Connect without specifying database
        engine = create_engine(base_url + '/mysql')

        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
            if not result.fetchone():
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                db_logger.info(f"Created database: {db_name}")
            else:
                db_logger.info(f"Database {db_name} already exists")

        engine.dispose()

    except Exception as e:
        db_logger.error(f"Failed to create database: {e}")
        raise

def run_migrations():
    """Run database migrations."""
    try:
        # Create database if it doesn't exist
        create_database()

        # Create engine with full database URL
        engine = create_engine(settings.database_url)

        # Create all tables
        Base.metadata.create_all(bind=engine)
        db_logger.info("Database tables created successfully")

        # You can add data migration logic here
        # For example, inserting default data, updating schema, etc.

        engine.dispose()

    except Exception as e:
        db_logger.error(f"Migration failed: {e}")
        raise

def drop_all_tables():
    """Drop all tables (use with caution!)."""
    try:
        engine = create_engine(settings.database_url)
        Base.metadata.drop_all(bind=engine)
        db_logger.warning("All tables dropped!")
        engine.dispose()
    except Exception as e:
        db_logger.error(f"Failed to drop tables: {e}")
        raise

def reset_database():
    """Reset the database (drop and recreate all tables)."""
    try:
        db_logger.warning("Resetting database...")
        drop_all_tables()
        run_migrations()
        db_logger.info("Database reset completed")
    except Exception as e:
        db_logger.error(f"Database reset failed: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database migration tool")
    parser.add_argument("--migrate", action="store_true", help="Run migrations")
    parser.add_argument("--reset", action="store_true", help="Reset database")
    parser.add_argument("--drop", action="store_true", help="Drop all tables")

    args = parser.parse_args()

    if args.reset:
        confirm = input("Are you sure you want to reset the database? This will delete all data! (yes/no): ")
        if confirm.lower() == 'yes':
            reset_database()
        else:
            print("Database reset cancelled.")
    elif args.drop:
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == 'yes':
            drop_all_tables()
        else:
            print("Drop tables cancelled.")
    elif args.migrate:
        run_migrations()
    else:
        print("Please specify an action: --migrate, --reset, or --drop")
        parser.print_help()