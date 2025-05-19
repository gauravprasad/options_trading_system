from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Database
    database_url: str = os.getenv("DATABASE_URL", "mysql+pymysql://root:password@localhost:3306/options_trading")
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "10"))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))

    # Alpaca API
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    alpaca_data_url: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")

    # Email
    smtp_host: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_username: str = os.getenv("SMTP_USERNAME", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    email_from: str = os.getenv("EMAIL_FROM", "")

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Application
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Trading
    default_portfolio_size: float = float(os.getenv("DEFAULT_PORTFOLIO_SIZE", "100000"))
    max_position_size: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.05"))

    # AI/ML
    use_gpu: bool = os.getenv("USE_GPU", "False").lower() == "true"
    model_update_interval: int = int(os.getenv("MODEL_UPDATE_INTERVAL", "86400"))

    # External APIs
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")

    # Notifications
    send_email_notifications: bool = os.getenv("SEND_EMAIL_NOTIFICATIONS", "True").lower() == "true"
    send_trade_confirmations: bool = os.getenv("SEND_TRADE_CONFIRMATIONS", "True").lower() == "true"
    daily_report_time: str = os.getenv("DAILY_REPORT_TIME", "17:00")
    weekly_report_day: int = int(os.getenv("WEEKLY_REPORT_DAY", "5"))
    monthly_report_day: int = int(os.getenv("MONTHLY_REPORT_DAY", "1"))

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()