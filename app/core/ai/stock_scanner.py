"""AI-powered stock scanner for identifying options trading opportunities."""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import ta
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from app.utils.logger import ai_logger
from app.config.settings import settings
from app.core.ai.sentiment_analysis import SentimentAnalyzer
from app.core.ai.technical_analysis import TechnicalAnalyzer

class StockScanner:
    """
    AI-powered stock scanner for identifying options trading opportunities.

    Features:
    - Technical analysis with multiple indicators
    - Sentiment analysis from news and social media
    - Fundamental analysis integration
    - Machine learning scoring
    - Volume and liquidity filtering
    """

    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.scaler = StandardScaler()
        self.ml_model = None
        self.feature_names = []
        self.logger = ai_logger

        # Default universe of stocks to scan
        self.default_universe = [
            # Large cap tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            # Large cap traditional
            'JPM', 'JNJ', 'PG', 'WMT', 'V', 'UNH', 'HD', 'DIS',
            # S&P 500 leaders
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLI', 'XLK', 'XLP',
            # Other popular options stocks
            'PYPL', 'CRM', 'ADBE', 'ORCL', 'INTC', 'AMD', 'MU', 'QCOM',
            'BA', 'CAT', 'GE', 'F', 'GM', 'NKE', 'SBUX', 'MCD'
        ]

        # Initialize ML model if not already done
        self._initialize_ml_model()

    def _initialize_ml_model(self):
        """Initialize the machine learning model for stock scoring."""
        try:
            # For now, create a simple random forest
            # In production, you'd load a pre-trained model
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            # Define feature names for consistency
            self.feature_names = [
                'rsi', 'macd_signal', 'bb_position', 'volume_ratio',
                'price_change_1d', 'price_change_5d', 'price_change_20d',
                'volatility_ratio', 'sentiment_score', 'news_sentiment',
                'market_cap_log', 'pe_ratio', 'revenue_growth'
            ]

            # Train with synthetic data for now
            # In production, use historical data with known good/bad outcomes
            self._train_synthetic_model()

            self.logger.info("ML model initialized successfully")

        except Exception as e:
            self.logger.error("Failed to initialize ML model", error=str(e))
            self.ml_model = None

    def _train_synthetic_model(self):
        """Train the model with synthetic data. Replace with real historical data."""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000

        # Generate features
        X = np.random.rand(n_samples, len(self.feature_names))

        # Generate labels based on simple rules (replace with real outcomes)
        y = np.zeros(n_samples)
        for i in range(n_samples):
            score = 0
            # Favorable RSI (30-70)
            if 0.3 <= X[i, 0] <= 0.7:
                score += 1
            # Positive sentiment
            if X[i, 8] > 0.5:
                score += 1
            # Good volume
            if X[i, 3] > 0.7:
                score += 1
            # Recent positive performance
            if X[i, 4] > 0.5:
                score += 1

            y[i] = 1 if score >= 3 else 0

        # Scale features and train
        X_scaled = self.scaler.fit_transform(X)
        self.ml_model.fit(X_scaled, y)

    async def scan_universe(self, symbols: List[str] = None,
                            strategies: List[str] = None) -> List[Dict]:
        """
        Scan a universe of stocks for trading opportunities.

        Args:
            symbols: List of symbols to scan (uses default if None)
            strategies: List of strategy types to evaluate

        Returns:
            List of dictionaries with stock analysis and scores
        """
        if symbols is None:
            symbols = self.default_universe

        if strategies is None:
            strategies = ['covered_call', 'iron_condor', 'bull_put_spread']

        self.logger.info("Starting stock scanner",
                         symbol_count=len(symbols),
                         strategies=strategies)

        results = []

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all scan tasks
            future_to_symbol = {
                executor.submit(self._analyze_single_stock, symbol, strategies): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.logger.debug("Completed analysis", symbol=symbol)
                except Exception as e:
                    self.logger.error("Failed to analyze stock", symbol=symbol, error=str(e))

        # Sort by overall score
        results.sort(key=lambda x: x['overall_score'], reverse=True)

        self.logger.info("Stock scan completed",
                         total_analyzed=len(results),
                         top_score=results[0]['overall_score'] if results else 0)

        return results

    def _analyze_single_stock(self, symbol: str, strategies: List[str]) -> Dict:
        """Analyze a single stock for trading opportunities."""
        try:
            # Get market data
            market_data = self._get_market_data(symbol)
            if market_data is None:
                return None

            # Get fundamental data
            fundamental_data = self._get_fundamental_data(symbol)

            # Perform technical analysis
            technical_scores = self.technical_analyzer.analyze(market_data)

            # Perform sentiment analysis
            sentiment_scores = self.sentiment_analyzer.analyze(symbol)

            # Calculate volume and liquidity metrics
            volume_metrics = self._calculate_volume_metrics(market_data)

            # Extract features for ML model
            features = self._extract_features(
                market_data, technical_scores, sentiment_scores,
                fundamental_data, volume_metrics
            )

            # Get ML prediction
            ml_score = self._get_ml_prediction(features)

            # Evaluate for each strategy
            strategy_scores = {}
            for strategy in strategies:
                strategy_scores[strategy] = self._evaluate_for_strategy(
                    symbol, strategy, market_data, technical_scores, sentiment_scores
                )

            # Calculate overall score
            overall_score = self._calculate_overall_score(
                technical_scores, sentiment_scores, ml_score, strategy_scores
            )

            # Get current price and key metrics
            current_price = market_data['Close'].iloc[-1]
            volume = market_data['Volume'].iloc[-1]

            result = {
                'symbol': symbol,
                'current_price': float(current_price),
                'volume': int(volume),
                'overall_score': overall_score,
                'ml_score': ml_score,
                'technical_scores': technical_scores,
                'sentiment_scores': sentiment_scores,
                'strategy_scores': strategy_scores,
                'volume_metrics': volume_metrics,
                'fundamental_data': fundamental_data,
                'scan_timestamp': datetime.now(),
                'recommendation': self._generate_recommendation(overall_score, strategy_scores)
            }

            return result

        except Exception as e:
            self.logger.error("Error analyzing stock", symbol=symbol, error=str(e))
            return None

    def _get_market_data(self, symbol: str, period: str = '6mo') -> pd.DataFrame:
        """Get historical market data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                self.logger.warning("No market data found", symbol=symbol)
                return None

            return data

        except Exception as e:
            self.logger.error("Failed to get market data", symbol=symbol, error=str(e))
            return None

    def _get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            fundamental_data = {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', info.get('trailingPE', None)),
                'pb_ratio': info.get('priceToBook', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'profit_margin': info.get('profitMargins', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }

            return fundamental_data

        except Exception as e:
            self.logger.error("Failed to get fundamental data", symbol=symbol, error=str(e))
            return {}

    def _calculate_volume_metrics(self, market_data: pd.DataFrame) -> Dict:
        """Calculate volume and liquidity metrics."""
        try:
            volumes = market_data['Volume']

            metrics = {
                'avg_volume_20d': volumes.tail(20).mean(),
                'volume_ratio': volumes.iloc[-1] / volumes.tail(20).mean(),
                'volume_trend': (volumes.tail(5).mean() / volumes.tail(20).mean()) - 1,
                'price_volume_correlation': market_data['Close'].tail(20).corr(volumes.tail(20))
            }

            return metrics

        except Exception as e:
            self.logger.error("Failed to calculate volume metrics", error=str(e))
            return {}

    def _extract_features(self, market_data: pd.DataFrame, technical_scores: Dict,
                          sentiment_scores: Dict, fundamental_data: Dict,
                          volume_metrics: Dict) -> np.ndarray:
        """Extract features for ML model."""
        try:
            features = []

            # Technical features
            features.append(technical_scores.get('rsi', 50) / 100)  # Normalize RSI
            features.append(max(-1, min(1, technical_scores.get('macd_signal', 0))))  # MACD signal
            features.append(technical_scores.get('bollinger_position', 0.5))

            # Volume features
            features.append(volume_metrics.get('volume_ratio', 1))

            # Price change features
            features.append(max(-0.5, min(0.5, technical_scores.get('return_1d', 0))))
            features.append(max(-0.5, min(0.5, technical_scores.get('return_5d', 0))))
            features.append(max(-0.5, min(0.5, technical_scores.get('return_20d', 0))))

            # Volatility features
            volatility_ratio = technical_scores.get('historical_volatility', 0.25) / 0.25
            features.append(min(3.0, volatility_ratio))

            # Sentiment features
            features.append(max(-1, min(1, sentiment_scores.get('overall_sentiment', 0))))
            features.append(max(-1, min(1, sentiment_scores.get('news_sentiment', 0))))

            # Fundamental features
            market_cap = fundamental_data.get('market_cap', 1e9)
            features.append(np.log10(max(1e6, market_cap)) / 12)  # Normalize log market cap

            pe_ratio = fundamental_data.get('pe_ratio', 20)
            features.append(min(100, max(0, pe_ratio)) / 100)  # Normalize PE ratio

            revenue_growth = fundamental_data.get('revenue_growth', 0)
            features.append(max(-1, min(1, revenue_growth or 0)))

            # Ensure we have the right number of features
            while len(features) < len(self.feature_names):
                features.append(0.5)  # Default neutral value

            features = features[:len(self.feature_names)]  # Trim if too many

            return np.array(features)

        except Exception as e:
            self.logger.error("Failed to extract features", error=str(e))
            return np.array([0.5] * len(self.feature_names))

    def _get_ml_prediction(self, features: np.ndarray) -> float:
        """Get ML model prediction for the stock."""
        try:
            if self.ml_model is None:
                return 0.5  # Neutral score if no model

            # Scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Get prediction probability
            prediction_proba = self.ml_model.predict_proba(features_scaled)[0]

            # Return probability of positive class
            return prediction_proba[1] if len(prediction_proba) > 1 else 0.5

        except Exception as e:
            self.logger.error("Failed to get ML prediction", error=str(e))
            return 0.5

    def _evaluate_for_strategy(self, symbol: str, strategy: str,
                               market_data: pd.DataFrame, technical_scores: Dict,
                               sentiment_scores: Dict) -> float:
        """Evaluate stock suitability for a specific strategy."""
        try:
            if strategy == 'covered_call':
                return self._evaluate_covered_call_suitability(
                    market_data, technical_scores, sentiment_scores
                )
            elif strategy == 'iron_condor':
                return self._evaluate_iron_condor_suitability(
                    market_data, technical_scores, sentiment_scores
                )
            elif strategy == 'bull_put_spread':
                return self._evaluate_bull_put_spread_suitability(
                    market_data, technical_scores, sentiment_scores
                )
            else:
                return 0.5  # Default neutral score

        except Exception as e:
            self.logger.error("Failed to evaluate strategy suitability",
                              symbol=symbol, strategy=strategy, error=str(e))
            return 0.5

    def _evaluate_covered_call_suitability(self, market_data: pd.DataFrame,
                                           technical_scores: Dict,
                                           sentiment_scores: Dict) -> float:
        """Evaluate suitability for covered call strategy."""
        score = 0.5

        # Prefer stocks with moderate volatility
        vol = technical_scores.get('historical_volatility', 0.25)
        if 0.2 <= vol <= 0.4:
            score += 0.15
        elif vol > 0.4:
            score += 0.05  # Too volatile might be risky

        # Prefer sideways to slightly bullish trend
        trend = technical_scores.get('trend_direction', 'sideways')
        if trend == 'sideways':
            score += 0.2
        elif trend == 'bullish':
            score += 0.1

        # Good volume for options trading
        volume_ratio = technical_scores.get('volume_ratio_20', 1)
        if volume_ratio > 1.2:
            score += 0.1

        # Neutral to slightly positive sentiment
        sentiment = sentiment_scores.get('overall_sentiment', 0)
        if -0.2 <= sentiment <= 0.3:
            score += 0.1

        # Check if RSI is not extreme
        rsi = technical_scores.get('rsi', 50)
        if 30 <= rsi <= 70:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _evaluate_iron_condor_suitability(self, market_data: pd.DataFrame,
                                          technical_scores: Dict,
                                          sentiment_scores: Dict) -> float:
        """Evaluate suitability for iron condor strategy."""
        score = 0.5

        # Prefer low volatility
        vol = technical_scores.get('historical_volatility', 0.25)
        if vol < 0.2:
            score += 0.2
        elif vol < 0.3:
            score += 0.1

        # Strongly prefer sideways market
        trend = technical_scores.get('trend_direction', 'sideways')
        if trend == 'sideways':
            score += 0.25
        else:
            score -= 0.1

        # Prefer price in middle of range
        bollinger_pos = technical_scores.get('bollinger_position', 0.5)
        if 0.3 <= bollinger_pos <= 0.7:
            score += 0.15

        # Neutral sentiment preferred
        sentiment = sentiment_scores.get('overall_sentiment', 0)
        if abs(sentiment) < 0.2:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _evaluate_bull_put_spread_suitability(self, market_data: pd.DataFrame,
                                              technical_scores: Dict,
                                              sentiment_scores: Dict) -> float:
        """Evaluate suitability for bull put spread strategy."""
        score = 0.5

        # Prefer bullish trend
        trend = technical_scores.get('trend_direction', 'sideways')
        if trend == 'bullish':
            score += 0.2
        elif trend == 'sideways':
            score += 0.1

        # Good momentum indicators
        rsi = technical_scores.get('rsi', 50)
        if 40 <= rsi <= 70:
            score += 0.15

        # Positive MACD
        macd_hist = technical_scores.get('macd_histogram', 0)
        if macd_hist > 0:
            score += 0.1

        # Bullish sentiment
        sentiment = sentiment_scores.get('overall_sentiment', 0)
        if sentiment > 0.1:
            score += 0.15

        # Above moving average
        price_vs_sma = technical_scores.get('price_vs_sma20', 0)
        if price_vs_sma > 0.02:
            score += 0.1

        return min(1.0, max(0.0, score))

    def _calculate_overall_score(self, technical_scores: Dict, sentiment_scores: Dict,
                                 ml_score: float, strategy_scores: Dict) -> float:
        """Calculate overall score for the stock."""
        try:
            # Component weights
            weights = {
                'technical': 0.3,
                'sentiment': 0.2,
                'ml': 0.25,
                'strategy': 0.25
            }

            # Normalize scores
            technical_score = technical_scores.get('technical_score', 0.5)
            sentiment_score = (sentiment_scores.get('overall_sentiment', 0) + 1) / 2  # Convert -1,1 to 0,1
            strategy_avg = np.mean(list(strategy_scores.values()))

            # Calculate weighted average
            overall_score = (
                    weights['technical'] * technical_score +
                    weights['sentiment'] * sentiment_score +
                    weights['ml'] * ml_score +
                    weights['strategy'] * strategy_avg
            )

            return round(overall_score, 3)

        except Exception as e:
            self.logger.error("Failed to calculate overall score", error=str(e))
            return 0.5

    def _generate_recommendation(self, overall_score: float, strategy_scores: Dict) -> str:
        """Generate recommendation based on scores."""
        try:
            # Find best strategy
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            best_score = strategy_scores[best_strategy]

            if overall_score >= 0.8:
                return f"STRONG BUY - Excellent for {best_strategy.replace('_', ' ').title()}"
            elif overall_score >= 0.7:
                return f"BUY - Good for {best_strategy.replace('_', ' ').title()}"
            elif overall_score >= 0.6:
                return f"MODERATE BUY - Consider {best_strategy.replace('_', ' ').title()}"
            elif overall_score >= 0.4:
                return f"HOLD - Limited opportunities"
            else:
                return f"AVOID - Poor setup for options strategies"

        except Exception as e:
            self.logger.error("Failed to generate recommendation", error=str(e))
            return "NEUTRAL - Unable to generate recommendation"

    def scan_specific_symbols(self, symbols: List[str], strategy: str = None) -> List[Dict]:
        """
        Scan specific symbols for opportunities.

        Args:
            symbols: List of symbols to scan
            strategy: Specific strategy to evaluate (optional)

        Returns:
            List of analysis results
        """
        strategies = [strategy] if strategy else ['covered_call', 'iron_condor', 'bull_put_spread']
        return asyncio.run(self.scan_universe(symbols, strategies))

    def get_top_opportunities(self, strategy: str, count: int = 10) -> List[Dict]:
        """Get top opportunities for a specific strategy."""
        # Scan default universe
        results = asyncio.run(self.scan_universe(strategies=[strategy]))

        # Sort by strategy-specific score
        results.sort(key=lambda x: x['strategy_scores'].get(strategy, 0), reverse=True)

        return results[:count]

    def update_universe(self, new_symbols: List[str]):
        """Update the default scanning universe."""
        self.default_universe = list(set(self.default_universe + new_symbols))
        self.logger.info("Updated scanning universe", total_symbols=len(self.default_universe))

    def get_universe_stats(self) -> Dict:
        """Get statistics about the scanning universe."""
        return {
            'total_symbols': len(self.default_universe),
            'symbols': self.default_universe,
            'last_scan': None,  # You could track this
            'model_status': 'trained' if self.ml_model else 'not_trained'
        }

    async def send_daily_scan_results(self, opportunities: List[Dict]):
        """Send daily scan results via email (placeholder method for email service)."""
        try:
            # This would be implemented in the email service
            # For now, just log the results
            self.logger.info("Daily scan results ready", count=len(opportunities))

            # Top 5 opportunities
            top_5 = opportunities[:5]
            for i, opp in enumerate(top_5, 1):
                self.logger.info(f"Top {i}: {opp['symbol']} - Score: {opp['overall_score']:.3f}")

        except Exception as e:
            self.logger.error("Failed to send daily scan results", error=str(e))