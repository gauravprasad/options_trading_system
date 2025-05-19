"""API routes for trade management."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from app.config.database import get_db
from app.models.pydantic_models import (
    TradeCreate, TradeResponse, TradeUpdate, TradeFilter,
    TradeListResponse, StandardResponse
)
from app.services.trade_service import TradeService
from app.utils.logger import api_logger

router = APIRouter()

@router.post("/", response_model=TradeResponse)
async def create_trade(
        trade_data: TradeCreate,
        db: Session = Depends(get_db)
):
    """Create a new trade."""
    try:
        trade_service = TradeService(db)
        trade = await trade_service.create_trade(trade_data)
        api_logger.info("Trade created successfully", trade_id=trade.trade_id)
        return trade
    except Exception as e:
        api_logger.error("Failed to create trade", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=TradeListResponse)
async def get_trades(
        skip: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=1000),
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        status: Optional[str] = None,
        trade_type: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        db: Session = Depends(get_db)
):
    """Get list of trades with filtering options."""
    try:
        trade_service = TradeService(db)

        # Create filter
        filter_params = TradeFilter(
            symbol=symbol,
            strategy_name=strategy,
            status=status,
            trade_type=trade_type,
            start_date=start_date,
            end_date=end_date
        )

        trades, total_count = await trade_service.get_trades(
            filter_params=filter_params,
            skip=skip,
            limit=limit
        )

        return TradeListResponse(
            trades=trades,
            total_count=total_count,
            page=skip // limit + 1,
            per_page=limit
        )
    except Exception as e:
        api_logger.error("Failed to get trades", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{trade_id}", response_model=TradeResponse)
async def get_trade(
        trade_id: str,
        db: Session = Depends(get_db)
):
    """Get a specific trade by ID."""
    try:
        trade_service = TradeService(db)
        trade = await trade_service.get_trade_by_id(trade_id)

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        return trade
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to get trade", trade_id=trade_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{trade_id}", response_model=TradeResponse)
async def update_trade(
        trade_id: str,
        trade_update: TradeUpdate,
        db: Session = Depends(get_db)
):
    """Update a trade."""
    try:
        trade_service = TradeService(db)
        trade = await trade_service.update_trade(trade_id, trade_update)

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        api_logger.info("Trade updated successfully", trade_id=trade_id)
        return trade
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to update trade", trade_id=trade_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{trade_id}", response_model=StandardResponse)
async def delete_trade(
        trade_id: str,
        db: Session = Depends(get_db)
):
    """Delete a trade."""
    try:
        trade_service = TradeService(db)
        success = await trade_service.delete_trade(trade_id)

        if not success:
            raise HTTPException(status_code=404, detail="Trade not found")

        api_logger.info("Trade deleted successfully", trade_id=trade_id)
        return StandardResponse(
            success=True,
            message="Trade deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to delete trade", trade_id=trade_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/{trade_id}/close", response_model=TradeResponse)
async def close_trade(
        trade_id: str,
        db: Session = Depends(get_db)
):
    """Close an open trade."""
    try:
        trade_service = TradeService(db)
        trade = await trade_service.close_trade(trade_id)

        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        api_logger.info("Trade closed successfully", trade_id=trade_id)
        return trade
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to close trade", trade_id=trade_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{trade_id}/pnl")
async def get_trade_pnl(
        trade_id: str,
        db: Session = Depends(get_db)
):
    """Get current P&L for a trade."""
    try:
        trade_service = TradeService(db)
        pnl_data = await trade_service.calculate_trade_pnl(trade_id)

        if not pnl_data:
            raise HTTPException(status_code=404, detail="Trade not found")

        return pnl_data
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error("Failed to get trade P&L", trade_id=trade_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))