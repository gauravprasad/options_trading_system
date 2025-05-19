"""API routes for strategy management."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional

from app.config.database import get_db
from app.models.pydantic_models import StandardResponse, AISignal
from app.services.strategy_service import StrategyService
from app.utils.logger import api_logger

router = APIRouter()

@router.get("/scan")
async def scan_stocks(
        symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
        strategy: Optional[str] = Query(None, description="Specific strategy to evaluate"),
        limit: int = Query(50, ge=1, le=200),
        db: Session = Depends(get_db)
):
    """Scan stocks for trading opportunities."""
    try:
        strategy_service = StrategyService(db)

        symbol_list = symbols.split(',') if symbols else None

        results = await strategy_service.scan_opportunities(
            symbols=symbol_list,
            strategy=strategy,
            limit=limit
        )

        return {
            'success': True,
            'results': results,
            'count': len(results)
        }
    except Exception as e:
        api_logger.error("Failed to scan stocks", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/evaluate/{symbol}")
async def evaluate_symbol(
        symbol: str,
        strategy: Optional[str] = Query(None),
        db: Session = Depends(get_db)
):
    """Evaluate a specific symbol for trading opportunities."""
    try:
        strategy_service = StrategyService(db)

        evaluation = await strategy_service.evaluate_symbol(symbol, strategy)

        return evaluation
    except Exception as e:
        api_logger.error("Failed to evaluate symbol", symbol=symbol, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/signals/{symbol}")
async def get_trading_signals(
        symbol: str,
        db: Session = Depends(get_db)
):
    """Get AI trading signals for a symbol."""
    try:
        strategy_service = StrategyService(db)

        signals = await strategy_service.get_ai_signals(symbol)

        return signals
    except Exception as e:
        api_logger.error("Failed to get trading signals", symbol=symbol, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/execute")
async def execute_strategy(
        symbol: str,
        strategy: str,
        auto_execute: bool = False,
        db: Session = Depends(get_db)
):
    """Execute a trading strategy for a symbol."""
    try:
        strategy_service = StrategyService(db)

        result = await strategy_service.execute_strategy(
            symbol=symbol,
            strategy=strategy,
            auto_execute=auto_execute
        )

        return result
    except Exception as e:
        api_logger.error("Failed to execute strategy",
                         symbol=symbol, strategy=strategy, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/performance")
async def get_strategy_performance(
        strategy: Optional[str] = Query(None),
        symbol: Optional[str] = Query(None),
        period: str = Query("monthly", regex="^(daily|weekly|monthly|yearly)$"),
        db: Session = Depends(get_db)
):
    """Get strategy performance metrics."""
    try:
        strategy_service = StrategyService(db)

        performance = await strategy_service.get_strategy_performance(
            strategy=strategy,
            symbol=symbol,
            period=period
        )

        return performance
    except Exception as e:
        api_logger.error("Failed to get strategy performance", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/market-conditions")
async def get_market_conditions(
        symbol: Optional[str] = Query(None),
        db: Session = Depends(get_db)
):
    """Get current market conditions analysis."""
    try:
        strategy_service = StrategyService(db)

        conditions = await strategy_service.analyze_market_conditions(symbol)

        return conditions
    except Exception as e:
        api_logger.error("Failed to get market conditions", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))