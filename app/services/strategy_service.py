"""Strategy service for managing trading strategies."""

import asyncio
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

from app.models.database_models import Trade, StrategyPerformance, MarketData
from app.models.enums import StrategyType
from app.core.ai.stock_scanner import StockScanner
from app.core.ai.sentiment_analysis import SentimentAnalyzer
from app.core.ai.technical_analysis import TechnicalAnalyzer
from app.core.strategies.covered_call import CoveredCallStrategy
from app.core.strategies.base_strategy import MarketConditions
from app.core.trading.alpaca_client import AlpacaClient
from app.utils.logger import trading_logger
from app.services.trade_service import TradeService

class StrategyService:
    """Service for strategy analysis and execution."""

    def __init__(self, db_session: Session):
        self.db = db_session
        self.scanner = StockScanner()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.alpaca_client = AlpacaClient()
        self.trade_service = TradeService(db_session)
        self.logger = trading_logger

        # Strategy mapping
        self.strategies = {
            StrategyType.COVERED_CALL: CoveredCallStrategy,
            # Add other strategies as they're implemented
        }

    async def scan_opportunities(self, symbols: List[str] = None,
                                 strategy: str = None, limit: int = 50) -> List[Dict]:
        """Scan for trading opportunities."""
        try:
            strategies = [strategy] if strategy else None
            results = await self.scanner.scan_universe(symbols, strategies)

            # Limit results
            limited_results = results[:limit]

            self.logger.info("Scan completed",
                             total_found=len(results),
                             returned=len(limited_results))

            return limited_results

        except Exception as e:
            self.logger.error("Failed to scan opportunities", error=str(e))
            raise

    async def evaluate_symbol(self, symbol: str, strategy: str = None) -> Dict:
        """Evaluate a specific symbol for trading opportunities."""
        try:
            # Get market data
            market_data = self.alpaca_client.get_market_data(symbol)

            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {symbol}")

            # Perform technical analysis
            technical_analysis = self.technical_analyzer.analyze(market_data)

            # Perform sentiment analysis
            sentiment_analysis = self.sentiment_analyzer.analyze(symbol)

            # Get options chain (if available)
            options_chain = self.alpaca_client.get_option_chain(symbol)

            # Create market conditions
            current_price = market_data['Close'].iloc[-1]
            market_conditions = MarketConditions(
                underlying_price=current_price,
                implied_volatility=sentiment_analysis.get('implied_volatility', 0.25),
                historical_volatility=technical_analysis.get('historical_volatility', 0.25),
                volume=int(market_data['Volume'].iloc[-1]),
                sentiment_score=sentiment_analysis.get('overall_sentiment', 0),
                technical_indicators=technical_analysis.get('indicators', {}),
                options_chain=options_chain
            )

            # Evaluate strategies
            strategy_evaluations = {}

            if strategy:
                # Evaluate specific strategy
                if strategy.upper() in [s.name for s in StrategyType]:
                    strategy_type = StrategyType[strategy.upper()]
                    if strategy_type in self.strategies:
                        strategy_class = self.strategies[strategy_type]
                        strategy_instance = strategy_class(symbol)
                        signal = strategy_instance.evaluate_entry_conditions(market_conditions)
                        strategy_evaluations[strategy] = {
                            'signal': signal.action,
                            'confidence': signal.confidence,
                            'reasoning': signal.reasoning,
                            'max_risk': signal.max_risk,
                            'max_profit': signal.max_profit,
                            'break_even_points': signal.break_even_points
                        }
            else:
                # Evaluate all available strategies
                for strategy_type, strategy_class in self.strategies.items():
                    try:
                        strategy_instance = strategy_class(symbol)
                        signal = strategy_instance.evaluate_entry_conditions(market_conditions)
                        strategy_evaluations[strategy_type.value] = {
                            'signal': signal.action,
                            'confidence': signal.confidence,
                            'reasoning': signal.reasoning,
                            'max_risk': signal.max_risk,
                            'max_profit': signal.max_profit,
                            'break_even_points': signal.break_even_points
                        }
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate {strategy_type.value}", error=str(e))

            evaluation_result = {
                'symbol': symbol,
                'current_price': current_price,
                'timestamp': datetime.now(),
                'technical_analysis': technical_analysis,
                'sentiment_analysis': sentiment_analysis,
                'strategy_evaluations': strategy_evaluations,
                'market_conditions': {
                    'underlying_price': current_price,
                    'implied_volatility': market_conditions.implied_volatility,
                    'historical_volatility': market_conditions.historical_volatility,
                    'volume': market_conditions.volume,
                    'sentiment_score': market_conditions.sentiment_score
                }
            }

            self.logger.info("Symbol evaluation completed", symbol=symbol)
            return evaluation_result

        except Exception as e:
            self.logger.error("Failed to evaluate symbol", symbol=symbol, error=str(e))
            raise

    async def get_ai_signals(self, symbol: str) -> Dict:
        """Get AI trading signals for a symbol."""
        try:
            # Use the scanner to analyze the symbol
            scanner_results = await self.scanner._analyze_single_stock(symbol, ['covered_call'])

            if not scanner_results:
                raise ValueError(f"Unable to analyze {symbol}")

            # Extract signals
            ai_signals = {
                'symbol': symbol,
                'overall_score': scanner_results['overall_score'],
                'ml_score': scanner_results['ml_score'],
                'recommendation': scanner_results['recommendation'],
                'technical_scores': scanner_results['technical_scores'],
                'sentiment_scores': scanner_results['sentiment_scores'],
                'strategy_scores': scanner_results['strategy_scores'],
                'timestamp': datetime.now(),
                'signals': []
            }

            # Generate detailed signals
            for strategy, score in scanner_results['strategy_scores'].items():
                if score > 0.6:  # Only include strong signals
                    signal_strength = 'STRONG' if score > 0.8 else 'MODERATE'
                    ai_signals['signals'].append({
                        'strategy': strategy,
                        'signal': 'BUY',
                        'strength': signal_strength,
                        'confidence': score,
                        'reasoning': f"AI model suggests {strategy} strategy with {score:.1%} confidence"
                    })

            return ai_signals

        except Exception as e:
            self.logger.error("Failed to get AI signals", symbol=symbol, error=str(e))
            raise

    async def execute_strategy(self, symbol: str, strategy: str,
                               auto_execute: bool = False) -> Dict:
        """Execute a trading strategy for a symbol."""
        try:
            # First evaluate the symbol
            evaluation = await self.evaluate_symbol(symbol, strategy)

            strategy_eval = evaluation['strategy_evaluations'].get(strategy)
            if not strategy_eval:
                raise ValueError(f"Strategy {strategy} not available for {symbol}")

            # Check if signal is favorable
            if strategy_eval['signal'] not in ['OPEN', 'BUY']:
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'action': 'NO_ACTION',
                    'reason': f"Signal not favorable: {strategy_eval['signal']}",
                    'confidence': strategy_eval['confidence']
                }

            # Get strategy instance to generate trade details
            strategy_type = StrategyType[strategy.upper()]
            strategy_class = self.strategies[strategy_type]
            strategy_instance = strategy_class(symbol)

            # Get market conditions again for fresh data
            market_data = self.alpaca_client.get_market_data(symbol)
            current_price = market_data['Close'].iloc[-1]

            # Create market conditions
            market_conditions = MarketConditions(
                underlying_price=current_price,
                implied_volatility=0.25,  # Would get from options data
                historical_volatility=evaluation['technical_analysis'].get('historical_volatility', 0.25),
                volume=int(market_data['Volume'].iloc[-1]),
                sentiment_score=evaluation['sentiment_analysis'].get('overall_sentiment', 0),
                technical_indicators=evaluation['technical_analysis'].get('indicators', {}),
                options_chain=self.alpaca_client.get_option_chain(symbol)
            )

            # Get entry signal with detailed legs
            signal = strategy_instance.evaluate_entry_conditions(market_conditions)

            if auto_execute and signal.action == 'OPEN':
                # Execute the trade
                from app.models.pydantic_models import TradeCreate, OptionLegCreate

                # Convert strategy legs to API format
                api_legs = []
                for leg in signal.legs:
                    if leg.get('asset_type') == 'option':
                        api_leg = OptionLegCreate(
                            option_type=leg['option_type'],
                            action=leg['action'],
                            strike_price=leg['strike_price'],
                            expiry_date=datetime.strptime(leg['expiry_date'], '%Y-%m-%d').date(),
                            quantity=leg['quantity']
                        )
                        api_legs.append(api_leg)

                # Create trade request
                trade_request = TradeCreate(
                    symbol=symbol,
                    strategy_name=strategy_type,
                    trade_type='LIVE',
                    legs=api_legs,
                    max_risk=signal.max_risk,
                    strategy_parameters={
                        'ai_confidence': signal.confidence,
                        'entry_reasoning': signal.reasoning
                    }
                )

                # Execute trade
                trade_result = await self.trade_service.create_trade(trade_request)

                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'action': 'TRADE_EXECUTED',
                    'trade_id': trade_result.trade_id,
                    'confidence': signal.confidence,
                    'max_risk': signal.max_risk,
                    'max_profit': signal.max_profit,
                    'reasoning': signal.reasoning
                }
            else:
                # Just return the recommendation
                return {
                    'symbol': symbol,
                    'strategy': strategy,
                    'action': 'RECOMMENDATION',
                    'signal': signal.action,
                    'confidence': signal.confidence,
                    'max_risk': signal.max_risk,
                    'max_profit': signal.max_profit,
                    'reasoning': signal.reasoning,
                    'legs': signal.legs,
                    'auto_execute': auto_execute
                }

        except Exception as e:
            self.logger.error("Failed to execute strategy",
                              symbol=symbol, strategy=strategy, error=str(e))
            raise

    async def get_strategy_performance(self, strategy: str = None,
                                       symbol: str = None, period: str = 'monthly') -> Dict:
        """Get strategy performance metrics."""
        try:
            query = self.db.query(StrategyPerformance)

            # Apply filters
            if strategy:
                strategy_enum = StrategyType[strategy.upper()]
                query = query.filter(StrategyPerformance.strategy_name == strategy_enum)

            if symbol:
                query = query.filter(StrategyPerformance.symbol == symbol.upper())

            query = query.filter(StrategyPerformance.period == period)

            # Get recent performance records
            performance_records = query.order_by(
                StrategyPerformance.period_end.desc()
            ).limit(12).all()

            if not performance_records:
                # Generate mock performance data if no records exist
                return self._generate_mock_performance(strategy, symbol, period)

            # Aggregate performance
            total_trades = sum(p.total_trades for p in performance_records)
            total_pnl = sum(float(p.total_pnl) for p in performance_records)
            winning_trades = sum(p.winning_trades for p in performance_records)

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0

            # Get latest record for additional metrics
            latest = performance_records[0]

            performance_summary = {
                'strategy': strategy,
                'symbol': symbol,
                'period': period,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'avg_pnl_per_trade': round(avg_pnl_per_trade, 2),
                'sharpe_ratio': float(latest.sharpe_ratio) if latest.sharpe_ratio else None,
                'max_drawdown': float(latest.max_drawdown) if latest.max_drawdown else None,
                'profit_factor': float(latest.profit_factor) if latest.profit_factor else None,
                'period_data': [
                    {
                        'period_start': p.period_start,
                        'period_end': p.period_end,
                        'total_pnl': float(p.total_pnl),
                        'trades': p.total_trades,
                        'win_rate': float(p.win_rate * 100) if p.win_rate else 0
                    }
                    for p in performance_records
                ]
            }

            return performance_summary

        except Exception as e:
            self.logger.error("Failed to get strategy performance", error=str(e))
            raise

    async def analyze_market_conditions(self, symbol: str = None) -> Dict:
        """Analyze current market conditions."""
        try:
            if symbol:
                # Analyze specific symbol
                market_data = self.alpaca_client.get_market_data(symbol)
                if market_data is None or market_data.empty:
                    raise ValueError(f"No market data for {symbol}")

                technical_analysis = self.technical_analyzer.analyze(market_data)
                sentiment_analysis = self.sentiment_analyzer.analyze(symbol)

                conditions = {
                    'symbol': symbol,
                    'current_price': market_data['Close'].iloc[-1],
                    'trend_direction': technical_analysis['trend_direction'],
                    'momentum_strength': technical_analysis['momentum_strength'],
                    'volatility_regime': technical_analysis['volatility_regime'],
                    'rsi': technical_analysis.get('rsi', 50),
                    'technical_score': technical_analysis['technical_score'],
                    'sentiment_score': sentiment_analysis['overall_sentiment'],
                    'sentiment_trend': sentiment_analysis['sentiment_trend'],
                    'implied_volatility': sentiment_analysis.get('implied_volatility', 0.25),
                    'volume_ratio': technical_analysis.get('volume_ratio_20', 1.0),
                    'timestamp': datetime.now()
                }
            else:
                # Analyze broader market (using SPY as proxy)
                spy_data = self.alpaca_client.get_market_data('SPY')
                if spy_data is None or spy_data.empty:
                    raise ValueError("Unable to get market data")

                technical_analysis = self.technical_analyzer.analyze(spy_data)
                sentiment_analysis = self.sentiment_analyzer.analyze('SPY')

                # Get VIX for market fear gauge (simulated)
                import random
                vix_level = random.uniform(15, 35)  # Simulated VIX

                conditions = {
                    'market_symbol': 'SPY',
                    'market_price': spy_data['Close'].iloc[-1],
                    'market_trend': technical_analysis['trend_direction'],
                    'market_volatility': technical_analysis['volatility_regime'],
                    'vix_level': vix_level,
                    'fear_greed_index': 50 + sentiment_analysis['overall_sentiment'] * 50,
                    'technical_score': technical_analysis['technical_score'],
                    'volume_trend': technical_analysis.get('volume_trend', 0),
                    'market_breadth': random.uniform(0.3, 0.7),  # Simulated
                    'sector_rotation': self._analyze_sector_rotation(),
                    'timestamp': datetime.now()
                }

            return conditions

        except Exception as e:
            self.logger.error("Failed to analyze market conditions", error=str(e))
            raise

    def _generate_mock_performance(self, strategy: str, symbol: str, period: str) -> Dict:
        """Generate mock performance data when no records exist."""
        import random

        # Set seed for consistent mock data
        random.seed(hash(f"{strategy}_{symbol}_{period}"))

        return {
            'strategy': strategy,
            'symbol': symbol,
            'period': period,
            'total_trades': random.randint(20, 100),
            'winning_trades': random.randint(12, 65),
            'losing_trades': random.randint(8, 35),
            'win_rate': round(random.uniform(55, 75), 2),
            'total_pnl': round(random.uniform(1000, 15000), 2),
            'avg_pnl_per_trade': round(random.uniform(50, 300), 2),
            'sharpe_ratio': round(random.uniform(1.2, 2.5), 2),
            'max_drawdown': round(random.uniform(-0.15, -0.05), 4),
            'profit_factor': round(random.uniform(1.3, 2.1), 2),
            'period_data': [],
            'note': 'Mock data - no historical records found'
        }

    def _analyze_sector_rotation(self) -> Dict:
        """Analyze sector rotation (simplified implementation)."""
        import random

        sectors = [
            'Technology', 'Healthcare', 'Financials', 'Energy',
            'Consumer Discretionary', 'Industrials', 'Materials',
            'Utilities', 'Real Estate', 'Consumer Staples'
        ]

        return {
            'leading_sectors': random.sample(sectors, 3),
            'lagging_sectors': random.sample(sectors, 3),
            'rotation_strength': random.uniform(0.3, 0.8)
        }