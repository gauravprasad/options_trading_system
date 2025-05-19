"""
Enums used throughout the application.
These can be imported separately for cleaner code.
"""

import enum

class TradeType(str, enum.Enum):
    """Types of trades in the system."""
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"

class TradeStatus(str, enum.Enum):
    """Status of a trade."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

class OptionType(str, enum.Enum):
    """Type of option contract."""
    CALL = "CALL"
    PUT = "PUT"

class OptionAction(str, enum.Enum):
    """Action taken on an option."""
    BUY = "BUY"
    SELL = "SELL"

class StrategyType(str, enum.Enum):
    """Available trading strategies."""
    COVERED_CALL = "COVERED_CALL"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    IRON_CONDOR = "IRON_CONDOR"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
    BUTTERFLY = "BUTTERFLY"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    CALENDAR_SPREAD = "CALENDAR_SPREAD"
    DIAGONAL_SPREAD = "DIAGONAL_SPREAD"
    COLLAR = "COLLAR"
    SYNTHETIC_LONG = "SYNTHETIC_LONG"
    SYNTHETIC_SHORT = "SYNTHETIC_SHORT"
    RATIO_SPREAD = "RATIO_SPREAD"

class OrderStatus(str, enum.Enum):
    """Status of orders."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderType(str, enum.Enum):
    """Types of orders."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class TimeInForce(str, enum.Enum):
    """Time in force for orders."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill

class MarketTrend(str, enum.Enum):
    """Market trend directions."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"

class VolatilityRegime(str, enum.Enum):
    """Volatility regimes."""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class RiskLevel(str, enum.Enum):
    """Risk levels for trades."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class AlertType(str, enum.Enum):
    """Types of alerts."""
    TRADE_EXECUTED = "TRADE_EXECUTED"
    TRADE_CLOSED = "TRADE_CLOSED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    PROFIT_TARGET_HIT = "PROFIT_TARGET_HIT"
    EXPIRATION_WARNING = "EXPIRATION_WARNING"
    MARGIN_CALL = "MARGIN_CALL"
    SYSTEM_ERROR = "SYSTEM_ERROR"

class EmailType(str, enum.Enum):
    """Types of emails sent by the system."""
    TRADE_CONFIRMATION = "TRADE_CONFIRMATION"
    DAILY_REPORT = "DAILY_REPORT"
    WEEKLY_REPORT = "WEEKLY_REPORT"
    MONTHLY_REPORT = "MONTHLY_REPORT"
    ALERT_NOTIFICATION = "ALERT_NOTIFICATION"
    SYSTEM_NOTIFICATION = "SYSTEM_NOTIFICATION"
    PERFORMANCE_REPORT = "PERFORMANCE_REPORT"

class DataSource(str, enum.Enum):
    """Data sources for market data."""
    ALPACA = "ALPACA"
    POLYGON = "POLYGON"
    YAHOO_FINANCE = "YAHOO_FINANCE"
    ALPHA_VANTAGE = "ALPHA_VANTAGE"
    IEX_CLOUD = "IEX_CLOUD"

class AssetClass(str, enum.Enum):
    """Asset classes."""
    STOCK = "STOCK"
    OPTION = "OPTION"
    ETF = "ETF"
    MUTUAL_FUND = "MUTUAL_FUND"
    BOND = "BOND"
    FUTURE = "FUTURE"
    FOREX = "FOREX"
    CRYPTO = "CRYPTO"

# Export all enums for easy importing
__all__ = [
    'TradeType', 'TradeStatus', 'OptionType', 'OptionAction', 'StrategyType',
    'OrderStatus', 'OrderType', 'TimeInForce', 'MarketTrend', 'VolatilityRegime',
    'RiskLevel', 'AlertType', 'EmailType', 'DataSource', 'AssetClass'
]