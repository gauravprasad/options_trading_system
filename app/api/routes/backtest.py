"""API routes for backtesting."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from app.config.database import get_db
from app.models.pydantic_models import (
    BacktestRequest, BacktestResponse, BacktestListResponse, StandardResponse
)
from app.services.backtest_service import BacktestService
from app.utils.logger import api_logger

router = APIRouter()

@router.post("/", response_model=BacktestResponse)
async def run_backtest(
        backtest_request: BacktestRequest,
        db: Session = Depends(get_db)
):
    """Run a new backtest."""
    try:
        backtest_service = BacktestService(db)

        backtest = await backtest_service.run_backtest(backtest_request)

        api_logger.info("Backtest started",
                        run_id=backtest.run_id,
                        strategy=backtest_request.strategy_name.value)

        return backtest
    except Exception as e:
        api_logger.error("Failed to run backtest", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=BacktestListResponse)
async def get_backtests(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        db: Session = Depends(get_db)
):
    """Get list of backtests with filtering options."""
    try:
        backtest_service = BacktestService(db)

        backtests, total_count = await backtest_service.get_backtests(
            skip=skip,
            limit=limit,
            strategy=strategy,
            symbol=symbol,
            status=status
        )

        return BacktestListResponse(
            backtests=backtests,
            total_count=total_count,
            page=skip // limit + 1,
            per_page=limit
        )
    except Exception as e:
        api_logger.error("Failed to get backtests", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{run_id}", response_model=BacktestResponse)
async def get_backtest(
        run_id: str,
        db: Session = Depends(get_db)
):
    """Get a specific backtest by ID."""
    try:
        backtest_service = BacktestService(db)

        backtest = await backtest_service.get_backtest_by_id(run_id)

        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")

        return backtest
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to get backtest", run_id=run_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{run_id}", response_model=StandardResponse)
async def delete_backtest(
        run_id: str,
        db: Session = Depends(get_db)
):
    """Delete a backtest."""
    try:
        backtest_service = BacktestService(db)

        success = await backtest_service.delete_backtest(run_id)

        if not success:
            raise HTTPException(status_code=404, detail="Backtest not found")

        api_logger.info("Backtest deleted", run_id=run_id)

        return StandardResponse(
            success=True,
            message="Backtest deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to delete backtest", run_id=run_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{run_id}/results")
async def get_backtest_results(
        run_id: str,
        db: Session = Depends(get_db)
):
    """Get detailed backtest results."""
    try:
        backtest_service = BacktestService(db)

        results = await backtest_service.get_backtest_results(run_id)

        if not results:
            raise HTTPException(status_code=404, detail="Backtest not found")

        return results
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to get backtest results", run_id=run_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{run_id}/trades")
async def get_backtest_trades(
        run_id: str,
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        db: Session = Depends(get_db)
):
    """Get trades from a backtest."""
    try:
        backtest_service = BacktestService(db)

        trades = await backtest_service.get_backtest_trades(
            run_id=run_id,
            skip=skip,
            limit=limit
        )

        return {
            'trades': trades,
            'run_id': run_id,
            'count': len(trades)
        }
    except Exception as e:
        api_logger.error("Failed to get backtest trades", run_id=run_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{run_id}/compare")
async def compare_backtests(
        run_id: str,
        compare_with: List[str],
        db: Session = Depends(get_db)
):
    """Compare multiple backtests."""
    try:
        backtest_service = BacktestService(db)

        comparison = await backtest_service.compare_backtests([run_id] + compare_with)

        return comparison
    except Exception as e:
        api_logger.error("Failed to compare backtests", run_id=run_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))