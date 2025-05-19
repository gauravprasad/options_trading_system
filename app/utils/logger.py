import structlog
import logging
import sys
from typing import Optional
from app.config.settings import settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.environment == "production"
        else structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Configure standard logging
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=getattr(logging, settings.log_level.upper()),
)

def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)

# Pre-configured loggers for different modules
trading_logger = get_logger("trading")
ai_logger = get_logger("ai")
backtest_logger = get_logger("backtest")
api_logger = get_logger("api")
db_logger = get_logger("database")
email_logger = get_logger("email")