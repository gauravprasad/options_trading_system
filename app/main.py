"""FastAPI main application entry point."""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from contextlib import asynccontextmanager

from app.config.database import create_tables, check_database_connection
from app.config.settings import settings
from app.utils.logger import get_logger, api_logger
from app.api.routes import trades, strategies, backtest, reports
from app.core.trading.alpaca_client import AlpacaClient

# Initialize logger
logger = get_logger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Options Trading System", version="1.0.0")

    # Initialize database
    if not check_database_connection():
        logger.error("Failed to connect to database")
        raise Exception("Database connection failed")

    # Create tables
    create_tables()
    logger.info("Database tables initialized")

    # Test Alpaca connection
    alpaca_client = AlpacaClient()
    if not alpaca_client.check_connection():
        logger.warning("Failed to connect to Alpaca API - continuing with limited functionality")
    else:
        logger.info("Alpaca API connection successful")

    yield

    # Shutdown
    logger.info("Shutting down Options Trading System")

# Create FastAPI app
app = FastAPI(
    title="AI Options Trading System",
    description="Comprehensive AI-powered options trading platform",
    version="1.0.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trades.router, prefix="/api/v1/trades", tags=["Trades"])
app.include_router(strategies.router, prefix="/api/v1/strategies", tags=["Strategies"])
app.include_router(backtest.router, prefix="/api/v1/backtest", tags=["Backtesting"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Options Trading System",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connection
    db_status = check_database_connection()

    # Check Alpaca connection
    alpaca_client = AlpacaClient()
    alpaca_status = alpaca_client.check_connection()

    return {
        "status": "healthy" if db_status and alpaca_status else "degraded",
        "database": "connected" if db_status else "disconnected",
        "alpaca": "connected" if alpaca_status else "disconnected",
        "timestamp": "2024-01-01T00:00:00Z"  # You'd use actual timestamp
    }

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="AI Options Trading System API",
        version="1.0.0",
        description="RESTful API for AI-powered options trading",
        routes=app.routes,
    )

    # Add additional API documentation
    openapi_schema["info"]["contact"] = {
        "name": "AI Trading Team",
        "email": "support@aitrading.com"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )