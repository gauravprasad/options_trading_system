"""Database models for the options trading system."""

from sqlalchemy import Column, String, DateTime, Decimal, Integer, Enum, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import enum

from app.config.database import Base

# Enums for database fields
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

def generate_id():
    """Generate a unique ID for trades."""
    return str(uuid.uuid4())

class Trade(Base):
    __tablename__ = "trades"

    trade_id = Column(String(36), primary_key=True, default=generate_id, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy_name = Column(Enum(StrategyType), nullable=False)
    trade_type = Column(Enum(TradeType), nullable=False, index=True)
    entry_date = Column(DateTime, nullable=False, default=func.now())
    exit_date = Column(DateTime, nullable=True)
    status = Column(Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN, index=True)

    # Underlying stock prices
    underlying_price_entry = Column(Decimal(10, 2), nullable=True)
    underlying_price_exit = Column(Decimal(10, 2), nullable=True)
    underlying_price_current = Column(Decimal(10, 2), nullable=True)

    # Profit/Loss calculations
    profit_loss_amount = Column(Decimal(12, 2), nullable=True)
    profit_loss_percentage = Column(Decimal(8, 4), nullable=True)
    potential_profit_amount = Column(Decimal(12, 2), nullable=True)
    potential_profit_percentage = Column(Decimal(8, 4), nullable=True)

    # Portfolio impact
    portfolio_value_at_entry = Column(Decimal(15, 2), nullable=True)
    position_size_percentage = Column(Decimal(5, 4), nullable=True)

    # Risk metrics
    max_risk = Column(Decimal(12, 2), nullable=True)
    max_profit = Column(Decimal(12, 2), nullable=True)
    break_even_points = Column(Text, nullable=True)  # JSON string of break-even prices

    # Strategy specific data
    strategy_parameters = Column(Text, nullable=True)  # JSON string of strategy parameters
    exit_reason = Column(String(100), nullable=True)

    # AI/ML related
    ai_confidence_score = Column(Decimal(3, 2), nullable=True)
    market_sentiment_score = Column(Decimal(3, 2), nullable=True)
    volatility_forecast = Column(Decimal(5, 4), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    legs = relationship("OptionLeg", back_populates="trade", cascade="all, delete-orphan")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_trade_symbol_date', 'symbol', 'entry_date'),
        Index('idx_trade_strategy_status', 'strategy_name', 'status'),
        Index('idx_trade_type_date', 'trade_type', 'entry_date'),
    )

class OptionLeg(Base):
    __tablename__ = "option_legs"

    leg_id = Column(String(36), primary_key=True, default=generate_id, index=True)
    trade_id = Column(String(36), ForeignKey("trades.trade_id", ondelete="CASCADE"), nullable=False, index=True)

    # Option specifications
    option_type = Column(Enum(OptionType), nullable=False)
    action = Column(Enum(OptionAction), nullable=False)
    strike_price = Column(Decimal(10, 2), nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    quantity = Column(Integer, nullable=False)

    # Pricing information
    entry_price = Column(Decimal(8, 4), nullable=True)
    exit_price = Column(Decimal(8, 4), nullable=True)
    current_price = Column(Decimal(8, 4), nullable=True)

    # Contract details
    contract_symbol = Column(String(30), nullable=True)
    contract_multiplier = Column(Integer, default=100)

    # Greeks at entry
    delta_entry = Column(Decimal(6, 4), nullable=True)
    gamma_entry = Column(Decimal(6, 4), nullable=True)
    theta_entry = Column(Decimal(6, 4), nullable=True)
    vega_entry = Column(Decimal(6, 4), nullable=True)

    # Current Greeks
    delta_current = Column(Decimal(6, 4), nullable=True)
    gamma_current = Column(Decimal(6, 4), nullable=True)
    theta_current = Column(Decimal(6, 4), nullable=True)
    vega_current = Column(Decimal(6, 4), nullable=True)

    # Profit/Loss for this leg
    leg_profit_loss = Column(Decimal(10, 2), nullable=True)
    leg_profit_loss_percentage = Column(Decimal(8, 4), nullable=True)

    # Order execution details
    alpaca_order_id = Column(String(50), nullable=True)
    fill_time = Column(DateTime, nullable=True)
    order_status = Column(String(20), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    trade = relationship("Trade", back_populates="legs")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_leg_trade_type', 'trade_id', 'option_type'),
        Index('idx_leg_expiry', 'expiry_date'),
        Index('idx_leg_strike', 'strike_price'),
    )

class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    run_id = Column(String(36), primary_key=True, default=generate_id, index=True)
    strategy_name = Column(Enum(StrategyType), nullable=False)
    symbol = Column(String(10), nullable=True)  # Null for multi-symbol backtests
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Decimal(15, 2), nullable=False)
    final_capital = Column(Decimal(15, 2), nullable=True)

    # Performance metrics
    total_return = Column(Decimal(8, 4), nullable=True)
    annual_return = Column(Decimal(8, 4), nullable=True)
    max_drawdown = Column(Decimal(8, 4), nullable=True)
    sharpe_ratio = Column(Decimal(6, 4), nullable=True)
    win_rate = Column(Decimal(5, 4), nullable=True)
    profit_factor = Column(Decimal(6, 4), nullable=True)

    # Trade statistics
    total_trades = Column(Integer, nullable=True)
    winning_trades = Column(Integer, nullable=True)
    losing_trades = Column(Integer, nullable=True)
    avg_trade_duration = Column(Integer, nullable=True)  # in days

    # Strategy parameters used
    parameters = Column(Text, nullable=True)  # JSON string

    # Market conditions during backtest
    market_volatility = Column(Decimal(6, 4), nullable=True)
    market_trend = Column(String(20), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="RUNNING")

class MarketData(Base):
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open_price = Column(Decimal(10, 4), nullable=False)
    high_price = Column(Decimal(10, 4), nullable=False)
    low_price = Column(Decimal(10, 4), nullable=False)
    close_price = Column(Decimal(10, 4), nullable=False)
    volume = Column(Integer, nullable=False)

    # Technical indicators
    rsi = Column(Decimal(5, 2), nullable=True)
    macd = Column(Decimal(8, 4), nullable=True)
    macd_signal = Column(Decimal(8, 4), nullable=True)
    bollinger_upper = Column(Decimal(10, 4), nullable=True)
    bollinger_lower = Column(Decimal(10, 4), nullable=True)
    sma_20 = Column(Decimal(10, 4), nullable=True)
    sma_50 = Column(Decimal(10, 4), nullable=True)
    ema_12 = Column(Decimal(10, 4), nullable=True)
    ema_26 = Column(Decimal(10, 4), nullable=True)

    # Volatility metrics
    implied_volatility = Column(Decimal(6, 4), nullable=True)
    historical_volatility = Column(Decimal(6, 4), nullable=True)

    # Sentiment scores
    sentiment_score = Column(Decimal(3, 2), nullable=True)
    news_sentiment = Column(Decimal(3, 2), nullable=True)

    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_market_data_symbol_time', 'symbol', 'timestamp'),
    )

class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(Enum(StrategyType), nullable=False, index=True)
    symbol = Column(String(10), nullable=True, index=True)
    period = Column(String(20), nullable=False)  # 'daily', 'weekly', 'monthly', 'yearly'
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Performance metrics for the period
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Decimal(12, 2), default=0)
    win_rate = Column(Decimal(5, 4), nullable=True)
    avg_win = Column(Decimal(10, 2), nullable=True)
    avg_loss = Column(Decimal(10, 2), nullable=True)
    profit_factor = Column(Decimal(6, 4), nullable=True)
    sharpe_ratio = Column(Decimal(6, 4), nullable=True)

    # Risk metrics
    max_drawdown = Column(Decimal(8, 4), nullable=True)
    var_95 = Column(Decimal(10, 2), nullable=True)  # Value at Risk 95%

    created_at = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index('idx_strategy_perf_strategy_period', 'strategy_name', 'period'),
    )

class EmailLog(Base):
    __tablename__ = "email_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email_type = Column(String(50), nullable=False)  # 'trade_alert', 'daily_report', etc.
    recipient = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    sent_at = Column(DateTime, default=func.now(), nullable=False)
    status = Column(String(20), default="SENT")  # 'SENT', 'FAILED', 'PENDING'
    error_message = Column(Text, nullable=True)

    # Reference to related trade if applicable
    trade_id = Column(String(36), ForeignKey("trades.trade_id"), nullable=True)

class SystemLog(Base):
    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False)  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON string with additional details
    timestamp = Column(DateTime, default=func.now(), nullable=False)

    # Stack trace for errors
    stack_trace = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_system_logs_level_time', 'level', 'timestamp'),
    )

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    setting_name = Column(String(100), nullable=False, unique=True)
    setting_value = Column(Text, nullable=False)
    setting_type = Column(String(20), nullable=False)  # 'string', 'number', 'boolean', 'json'
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

class PortfolioSnapshot(Base):
    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(DateTime, nullable=False, index=True)
    total_value = Column(Decimal(15, 2), nullable=False)
    cash_balance = Column(Decimal(15, 2), nullable=False)
    options_value = Column(Decimal(15, 2), nullable=False)
    stocks_value = Column(Decimal(15, 2), nullable=False)

    # Portfolio Greeks
    total_delta = Column(Decimal(10, 4), nullable=True)
    total_gamma = Column(Decimal(10, 4), nullable=True)
    total_theta = Column(Decimal(10, 4), nullable=True)
    total_vega = Column(Decimal(10, 4), nullable=True)

    # Risk metrics
    portfolio_beta = Column(Decimal(6, 4), nullable=True)
    var_1day = Column(Decimal(10, 2), nullable=True)
    var_1week = Column(Decimal(10, 2), nullable=True)

    # Active positions count
    open_positions = Column(Integer, default=0)
    expiring_soon = Column(Integer, default=0)  # expiring within 7 days

    created_at = Column(DateTime, default=func.now(), nullable=False)