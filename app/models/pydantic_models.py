from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

# Enums for API models
class TradeTypeAPI(str, Enum):
    LIVE = "LIVE"
    BACKTEST = "BACKTEST"

class TradeStatusAPI(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"

class OptionTypeAPI(str, Enum):
    CALL = "CALL"
    PUT = "PUT"

class OptionActionAPI(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class StrategyTypeAPI(str, Enum):
    COVERED_CALL = "COVERED_CALL"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    IRON_CONDOR = "IRON_CONDOR"
    BULL_PUT_SPREAD = "BULL_PUT_SPREAD"
    BEAR_CALL_SPREAD = "BEAR_CALL_SPREAD"
    BUTTERFLY = "BUTTERFLY"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"

# Base models
class OptionLegCreate(BaseModel):
    option_type: OptionTypeAPI = Field(..., description="Type of option (CALL or PUT)")
    action: OptionActionAPI = Field(..., description="Action to take (BUY or SELL)")
    strike_price: float = Field(..., gt=0, description="Strike price of the option")
    expiry_date: date = Field(..., description="Expiration date of the option")
    quantity: int = Field(..., gt=0, description="Number of contracts")

    @validator('expiry_date')
    def expiry_must_be_future(cls, v):
        if v <= date.today():
            raise ValueError('Expiry date must be in the future')
        return v

class OptionLegResponse(BaseModel):
    leg_id: str
    option_type: OptionTypeAPI
    action: OptionActionAPI
    strike_price: float
    expiry_date: date
    quantity: int
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    current_price: Optional[float] = None
    contract_symbol: Optional[str] = None
    leg_profit_loss: Optional[float] = None
    leg_profit_loss_percentage: Optional[float] = None

    # Greeks
    delta_entry: Optional[float] = None
    gamma_entry: Optional[float] = None
    theta_entry: Optional[float] = None
    vega_entry: Optional[float] = None
    delta_current: Optional[float] = None
    gamma_current: Optional[float] = None
    theta_current: Optional[float] = None
    vega_current: Optional[float] = None

    # Order details
    alpaca_order_id: Optional[str] = None
    fill_time: Optional[datetime] = None
    order_status: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TradeCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    strategy_name: StrategyTypeAPI = Field(..., description="Trading strategy name")
    trade_type: TradeTypeAPI = Field(default=TradeTypeAPI.LIVE, description="Type of trade")
    legs: List[OptionLegCreate] = Field(..., min_items=1, description="Option legs for the trade")

    # Optional strategy parameters
    max_risk: Optional[float] = Field(None, gt=0, description="Maximum risk amount")
    target_profit: Optional[float] = Field(None, gt=0, description="Target profit amount")
    strategy_parameters: Optional[Dict[str, Any]] = Field(None, description="Additional strategy parameters")

    @validator('symbol')
    def symbol_uppercase(cls, v):
        return v.upper()

class TradeResponse(BaseModel):
    trade_id: str
    symbol: str
    strategy_name: StrategyTypeAPI
    trade_type: TradeTypeAPI
    status: TradeStatusAPI
    entry_date: datetime
    exit_date: Optional[datetime] = None

    # Pricing information
    underlying_price_entry: Optional[float] = None
    underlying_price_exit: Optional[float] = None
    underlying_price_current: Optional[float] = None

    # P&L information
    profit_loss_amount: Optional[float] = None
    profit_loss_percentage: Optional[float] = None
    potential_profit_amount: Optional[float] = None
    potential_profit_percentage: Optional[float] = None

    # Risk metrics
    max_risk: Optional[float] = None
    max_profit: Optional[float] = None
    break_even_points: Optional[List[float]] = None

    # Portfolio impact
    portfolio_value_at_entry: Optional[float] = None
    position_size_percentage: Optional[float] = None

    # AI/ML metrics
    ai_confidence_score: Optional[float] = None
    market_sentiment_score: Optional[float] = None
    volatility_forecast: Optional[float] = None

    # Strategy details
    strategy_parameters: Optional[Dict[str, Any]] = None
    exit_reason: Optional[str] = None

    # Related data
    legs: List[OptionLegResponse] = []

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TradeUpdate(BaseModel):
    status: Optional[TradeStatusAPI] = None
    exit_reason: Optional[str] = None
    profit_loss_amount: Optional[float] = None
    profit_loss_percentage: Optional[float] = None

class TradeFilter(BaseModel):
    symbol: Optional[str] = None
    strategy_name: Optional[StrategyTypeAPI] = None
    trade_type: Optional[TradeTypeAPI] = None
    status: Optional[TradeStatusAPI] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    min_profit: Optional[float] = None
    max_profit: Optional[float] = None

class BacktestRequest(BaseModel):
    strategy_name: StrategyTypeAPI = Field(..., description="Strategy to backtest")
    symbol: Optional[str] = Field(None, description="Stock symbol (leave empty for all symbols)")
    start_date: date = Field(..., description="Backtest start date")
    end_date: date = Field(..., description="Backtest end date")
    initial_capital: float = Field(default=100000, gt=0, description="Initial capital amount")

    # Strategy parameters
    strategy_parameters: Optional[Dict[str, Any]] = Field(None, description="Strategy-specific parameters")

    # Risk management parameters
    max_position_size: Optional[float] = Field(0.05, gt=0, le=1, description="Maximum position size as % of portfolio")
    stop_loss_percentage: Optional[float] = Field(None, gt=0, le=1, description="Stop loss percentage")

    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('End date must be after start date')
        return v

class BacktestResponse(BaseModel):
    run_id: str
    strategy_name: StrategyTypeAPI
    symbol: Optional[str]
    start_date: date
    end_date: date
    initial_capital: float
    final_capital: Optional[float] = None

    # Performance metrics
    total_return: Optional[float] = None
    annual_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None

    # Trade statistics
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    avg_trade_duration: Optional[int] = None

    # Market conditions
    market_volatility: Optional[float] = None
    market_trend: Optional[str] = None

    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PortfolioSummary(BaseModel):
    total_value: float
    cash_balance: float
    options_value: float
    stocks_value: float

    # Portfolio Greeks
    total_delta: Optional[float] = None
    total_gamma: Optional[float] = None
    total_theta: Optional[float] = None
    total_vega: Optional[float] = None

    # Risk metrics
    portfolio_beta: Optional[float] = None
    var_1day: Optional[float] = None
    var_1week: Optional[float] = None

    # Position counts
    open_positions: int
    expiring_soon: int

    last_updated: datetime

class MarketDataResponse(BaseModel):
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int

    # Technical indicators
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None

    # Volatility metrics
    implied_volatility: Optional[float] = None
    historical_volatility: Optional[float] = None

    # Sentiment scores
    sentiment_score: Optional[float] = None
    news_sentiment: Optional[float] = None

class PerformanceReport(BaseModel):
    period: str
    period_start: date
    period_end: date
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    win_rate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    profit_factor: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None

class StrategyAnalysis(BaseModel):
    strategy_name: StrategyTypeAPI
    symbol: Optional[str]
    success_rate: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    best_markets: List[str]
    optimal_parameters: Dict[str, Any]

class AISignal(BaseModel):
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    recommended_strategy: StrategyTypeAPI
    predicted_direction: str  # 'UP', 'DOWN', 'SIDEWAYS'
    time_horizon: int  # days
    reasoning: str
    risk_factors: List[str]
    timestamp: datetime

class EmailRequest(BaseModel):
    email_type: str = Field(..., description="Type of email to send")
    recipient: Optional[str] = Field(None, description="Email recipient (uses default if not provided)")
    custom_subject: Optional[str] = Field(None, description="Custom email subject")
    custom_data: Optional[Dict[str, Any]] = Field(None, description="Additional data for email template")

class SystemStatus(BaseModel):
    system_health: str  # 'HEALTHY', 'WARNING', 'ERROR'
    trading_enabled: bool
    database_connected: bool
    alpaca_connected: bool
    last_heartbeat: datetime
    active_trades: int
    pending_orders: int
    system_uptime: str
    errors_last_24h: int

# Response wrappers
class TradeListResponse(BaseModel):
    trades: List[TradeResponse]
    total_count: int
    page: int
    per_page: int

class BacktestListResponse(BaseModel):
    backtests: List[BacktestResponse]
    total_count: int
    page: int
    per_page: int

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None
    error_code: Optional[str] = None