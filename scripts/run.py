#!/usr/bin/env python3
"""
Startup script for the options trading system.
Run this to start the application with all components.
"""

import sys
import os
import asyncio
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.main import app
from app.config.database import check_database_connection, create_tables
from app.config.settings import settings
from app.utils.logger import get_logger
import uvicorn

logger = get_logger("startup")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        'DATABASE_URL',
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'SMTP_USERNAME',
        'SMTP_PASSWORD'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please check your .env file or environment configuration")
        return False

    return True

def initialize_database():
    """Initialize database connection and create tables."""
    logger.info("Initializing database...")

    if not check_database_connection():
        logger.error("Failed to connect to database")
        return False

    try:
        create_tables()
        logger.info("Database tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        return False

def run_server():
    """Run the FastAPI server."""
    logger.info("Starting Options Trading System...")

    # Setup signal handlers
    setup_signal_handlers()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Initialize database
    if not initialize_database():
        sys.exit(1)

    # Start the server
    logger.info(f"Starting server on http://0.0.0.0:8000")
    logger.info(f"API documentation available at http://0.0.0.0:8000/docs")

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Options Trading System Startup")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    # Override settings with command line arguments
    if args.reload:
        settings.debug = True

    # Run the server
    try:
        if args.workers > 1:
            # Use Gunicorn for production with multiple workers
            import subprocess
            cmd = [
                "gunicorn",
                "app.main:app",
                "-w", str(args.workers),
                "-k", "uvicorn.workers.UvicornWorker",
                "--bind", f"{args.host}:{args.port}",
                "--access-logfile", "-",
                "--error-logfile", "-"
            ]
            subprocess.run(cmd)
        else:
            uvicorn.run(
                "app.main:app",
                host=args.host,
                port=args.port,
                reload=args.reload,
                log_level=settings.log_level.lower()
            )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)