# AI Options Trading System

A comprehensive AI-powered options trading platform with automated strategy execution, backtesting, and risk management.

## Features

- **AI-Powered Stock Scanning**: Intelligent stock selection using machine learning and sentiment analysis
- **Multiple Trading Strategies**: Covered calls, iron condors, bull put spreads, and more
- **Automated Trading**: Live trading through Alpaca API with paper trading support
- **Comprehensive Backtesting**: Historical strategy testing with detailed performance metrics
- **Risk Management**: Real-time risk monitoring and position sizing
- **Email Notifications**: Automated trade confirmations and performance reports
- **RESTful API**: Complete API with Swagger documentation
- **Real-time Market Data**: Integration with multiple data providers

## Project Structure

```
options_trading_system/
├── app/                    # Main application package
│   ├── api/               # FastAPI routes
│   ├── config/            # Configuration and database setup
│   ├── core/              # Core business logic
│   │   ├── ai/           # AI/ML components
│   │   ├── strategies/    # Trading strategies
│   │   └── trading/       # Trading execution
│   ├── models/            # Database and API models
│   ├── services/          # Business logic services
│   └── utils/             # Utilities and logging
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── .env.template         # Environment variables template
```

## Installation

### Prerequisites

- Python 3.9+
- MySQL 8.0+
- Redis (for Celery)
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd options_trading_system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.template .env
   ```

   Edit `.env` file with your configuration:
   ```env
   # Database
   DATABASE_URL=mysql+pymysql://username:password@localhost:3306/options_trading

   # Alpaca API (Paper Trading)
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_SECRET_KEY=your_alpaca_secret_key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets

   # Email Configuration
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   EMAIL_FROM=your_email@gmail.com

   # Other configurations...
   ```

5. **Setup database**
   ```bash
   python scripts/migrate_db.py --migrate
   ```

## Running the Application

### Development Mode

```bash
python scripts/run.py --reload
```

### Production Mode

```bash
python scripts/run.py --workers 4
```

The application will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Usage

### Web Interface

Access the Swagger documentation at `http://localhost:8000/docs` to explore all available endpoints.

### Key API Endpoints

#### Trading
- `POST /api/v1/trades/` - Create a new trade
- `GET /api/v1/trades/` - List all trades
- `GET /api/v1/trades/{trade_id}` - Get specific trade
- `POST /api/v1/trades/{trade_id}/close` - Close a trade

#### Strategy Analysis
- `GET /api/v1/strategies/scan` - Scan stocks for opportunities
- `GET /api/v1/strategies/evaluate/{symbol}` - Evaluate specific symbol
- `POST /api/v1/strategies/execute` - Execute a strategy

#### Backtesting
- `POST /api/v1/backtest/` - Run a backtest
- `GET /api/v1/backtest/` - List backtests
- `GET /api/v1/backtest/{run_id}/results` - Get backtest results

#### Reports
- `GET /api/v1/reports/daily` - Daily performance report
- `GET /api/v1/reports/portfolio` - Portfolio summary
- `POST /api/v1/reports/email` - Send email report

### Example API Usage

```python
import requests

# Scan for covered call opportunities
response = requests.get(
    'http://localhost:8000/api/v1/strategies/scan?strategy=covered_call&limit=10'
)
opportunities = response.json()

# Create a new trade
trade_data = {
    "symbol": "AAPL",
    "strategy_name": "COVERED_CALL",
    "trade_type": "BACKTEST",
    "legs": [
        {
            "option_type": "CALL",
            "action": "SELL",
            "strike_price": 150.0,
            "expiry_date": "2024-03-15",
            "quantity": 1
        }
    ]
}

response = requests.post(
    'http://localhost:8000/api/v1/trades/',
    json=trade_data
)
trade = response.json()
```

## Configuration

### Trading Configuration

```python
# In .env file
DEFAULT_PORTFOLIO_SIZE=100000
MAX_POSITION_SIZE=0.05  # 5% of portfolio per position
RISK_FREE_RATE=0.05     # 5% annual risk-free rate
```

### Email Notifications

Configure email settings to receive:
- Trade confirmations
- Daily/weekly/monthly reports
- Alert notifications
- Performance summaries

### Scheduled Tasks

Set up cron jobs for automated operations:

```bash
# Daily stock scan at 6 AM
0 6 * * * cd /path/to/project && python scripts/daily_scanner.py

# Weekly report on Fridays at 5 PM
0 17 * * 5 cd /path/to/project && python -c "
import asyncio
from app.services.email_service import EmailService
asyncio.run(EmailService().send_weekly_report())
"
```

## Available Strategies

1. **Covered Call**: Buy stock + sell call option
2. **Protective Put**: Buy stock + buy put option
3. **Iron Condor**: Sell call spread + sell put spread
4. **Bull Put Spread**: Sell put + buy lower put
5. **Bear Call Spread**: Sell call + buy higher call
6. **Butterfly Spread**: Optimized for sideways movement
7. **Straddle**: Buy call + buy put (volatility play)
8. **Strangle**: Buy OTM call + buy OTM put

## Risk Management

The system includes comprehensive risk management:
- Position sizing based on portfolio percentage
- Maximum risk limits per trade
- Stop-loss and profit-taking rules
- Portfolio-level risk monitoring
- Real-time Greeks calculation
- VaR (Value at Risk) calculations

## Backtesting

Run historical simulations to validate strategies:

```python
# Example backtest request
{
    "strategy_name": "COVERED_CALL",
    "symbol": "AAPL",
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "max_position_size": 0.05
}
```

Results include:
- Total return and annualized return
- Sharpe ratio and maximum drawdown
- Win rate and profit factor
- Trade-by-trade analysis
- Performance attribution

## Monitoring and Alerts

- Real-time portfolio monitoring
- Automated alerts for:
    - Position limits exceeded
    - Stop-loss triggers
    - Expiration warnings
    - System errors
- Performance dashboards
- Custom alert configurations

## Security

- API key authentication
- Environment variable protection
- Database connection encryption
- Secure email transmission
- Input validation and sanitization

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
    - Check MySQL service is running
    - Verify database credentials in .env
    - Ensure database exists

2. **Alpaca API Errors**
    - Verify API keys are correct
    - Check account status
    - Ensure paper trading URL is used

3. **Email Delivery Issues**
    - Check SMTP settings
    - Verify app password for Gmail
    - Check firewall/network settings

### Logging

Logs are written to stdout and can be redirected:

```bash
python scripts/run.py > app.log 2>&1
```

Log levels can be configured in the environment:
```env
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Linting
flake8 app/

# Type checking
mypy app/

# Formatting
black app/
```

### Adding New Strategies

1. Create strategy class inheriting from `BaseStrategy`
2. Implement required methods
3. Add strategy to enum in `models/enums.py`
4. Update strategy evaluation in scanner

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[License information here]

## Support

For support and questions:
- Create an issue on GitHub
- Email: support@example.com
- Documentation: [Link to documentation]

## Disclaimer

This software is for educational and research purposes. Trading involves risk, and past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.