import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re
from textblob import TextBlob
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import time

from app.utils.logger import ai_logger
from app.config.settings import settings

class SentimentAnalyzer:
    """
    Sentiment analysis for stocks using news, social media, and other data sources.
    """

    def __init__(self):
        self.logger = ai_logger
        self.news_sources = [
            'yahoo_finance',
            'alpha_vantage',
            'polygon'  # If you have Polygon subscription
        ]

    def analyze(self, symbol: str) -> Dict:
        """
        Perform comprehensive sentiment analysis for a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Dictionary with sentiment scores and details
        """
        try:
            self.logger.info("Starting sentiment analysis", symbol=symbol)

            # Get news sentiment
            news_sentiment = self._analyze_news_sentiment(symbol)

            # Get social media sentiment (placeholder - would integrate with Twitter API, Reddit, etc.)
            social_sentiment = self._analyze_social_sentiment(symbol)

            # Get analyst sentiment (from price targets, ratings)
            analyst_sentiment = self._analyze_analyst_sentiment(symbol)

            # Combine all sentiment sources
            combined_sentiment = self._combine_sentiment_scores(
                news_sentiment, social_sentiment, analyst_sentiment
            )

            result = {
                'overall_sentiment': combined_sentiment['overall'],
                'news_sentiment': news_sentiment['score'],
                'social_sentiment': social_sentiment['score'],
                'analyst_sentiment': analyst_sentiment['score'],
                'sentiment_confidence': combined_sentiment['confidence'],
                'sentiment_trend': combined_sentiment['trend'],
                'news_count': news_sentiment['count'],
                'key_themes': combined_sentiment['themes'],
                'analysis_timestamp': datetime.now()
            }

            self.logger.info("Sentiment analysis completed",
                             symbol=symbol,
                             overall_sentiment=result['overall_sentiment'])

            return result

        except Exception as e:
            self.logger.error("Failed to analyze sentiment", symbol=symbol, error=str(e))
            return self._default_sentiment_result()

    def _analyze_news_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from news articles."""
        try:
            # Get news from Yahoo Finance
            news_data = self._get_yahoo_news(symbol)

            if not news_data:
                return {'score': 0.0, 'count': 0, 'articles': []}

            sentiment_scores = []
            analyzed_articles = []

            for article in news_data[:20]:  # Analyze last 20 articles
                text = f"{article.get('title', '')} {article.get('summary', '')}"

                # Clean text
                text = self._clean_text(text)

                # Analyze sentiment using TextBlob
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity  # -1 to 1

                sentiment_scores.append(sentiment_score)
                analyzed_articles.append({
                    'title': article.get('title'),
                    'sentiment': sentiment_score,
                    'published': article.get('published_time'),
                    'source': article.get('source', 'Yahoo Finance')
                })

            # Calculate aggregate scores
            if sentiment_scores:
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_std = np.std(sentiment_scores)

                # Weight recent articles more heavily
                recent_weight = 0.7
                weighted_sentiment = (
                        recent_weight * np.mean(sentiment_scores[:5]) +
                        (1 - recent_weight) * np.mean(sentiment_scores[5:])
                ) if len(sentiment_scores) > 5 else avg_sentiment
            else:
                avg_sentiment = 0.0
                weighted_sentiment = 0.0
                sentiment_std = 0.0

            return {
                'score': weighted_sentiment,
                'average': avg_sentiment,
                'volatility': sentiment_std,
                'count': len(analyzed_articles),
                'articles': analyzed_articles
            }

        except Exception as e:
            self.logger.error("Failed to analyze news sentiment", symbol=symbol, error=str(e))
            return {'score': 0.0, 'count': 0, 'articles': []}

    def _get_yahoo_news(self, symbol: str) -> List[Dict]:
        """Get news from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            news_data = []
            for item in news:
                news_data.append({
                    'title': item.get('title'),
                    'summary': item.get('summary'),
                    'published_time': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'source': item.get('publisher')
                })

            return news_data

        except Exception as e:
            self.logger.error("Failed to get Yahoo news", symbol=symbol, error=str(e))
            return []

    def _analyze_social_sentiment(self, symbol: str) -> Dict:
        """
        Analyze social media sentiment.
        Note: This is a placeholder. In production, you'd integrate with:
        - Twitter API for tweets
        - Reddit API for r/stocks, r/investing posts
        - StockTwits API
        - Discord sentiment data
        """
        try:
            # Placeholder implementation
            # In reality, you'd fetch and analyze social media posts

            # Simulate social sentiment based on some factors
            import random
            random.seed(hash(symbol) % 100)

            # Generate realistic social sentiment
            base_sentiment = random.uniform(-0.3, 0.3)
            sentiment_volatility = random.uniform(0.1, 0.4)

            # Add some noise based on "social media trends"
            trend_factor = random.uniform(0.8, 1.2)
            social_sentiment = base_sentiment * trend_factor

            # Ensure within bounds
            social_sentiment = max(-1.0, min(1.0, social_sentiment))

            return {
                'score': social_sentiment,
                'volatility': sentiment_volatility,
                'volume': random.randint(100, 5000),  # Number of posts
                'trending': abs(social_sentiment) > 0.5,
                'sources': ['twitter', 'reddit', 'stocktwits']
            }

        except Exception as e:
            self.logger.error("Failed to analyze social sentiment", symbol=symbol, error=str(e))
            return {'score': 0.0, 'volatility': 0.0, 'volume': 0}

    def _analyze_analyst_sentiment(self, symbol: str) -> Dict:
        """Analyze analyst sentiment from ratings and price targets."""
        try:
            ticker = yf.Ticker(symbol)

            # Get analyst recommendations
            recommendations = ticker.recommendations

            if recommendations is None or recommendations.empty:
                return {'score': 0.0, 'count': 0, 'trend': 'neutral'}

            # Get recent recommendations (last 3 months)
            recent_recommendations = recommendations.tail(20)

            # Convert recommendations to sentiment scores
            recommendation_scores = {
                'strongBuy': 1.0,
                'buy': 0.5,
                'hold': 0.0,
                'sell': -0.5,
                'strongSell': -1.0
            }

            sentiment_scores = []
            for _, rec in recent_recommendations.iterrows():
                for rating, count in rec.items():
                    if rating in recommendation_scores and count > 0:
                        sentiment_scores.extend([recommendation_scores[rating]] * int(count))

            if sentiment_scores:
                analyst_sentiment = np.mean(sentiment_scores)
            else:
                analyst_sentiment = 0.0

            # Determine trend (compare recent vs older recommendations)
            if len(recent_recommendations) >= 10:
                recent_sentiment = np.mean([
                    recommendation_scores.get(rating, 0) * count
                    for rating, count in recent_recommendations.tail(5).mean().items()
                    if rating in recommendation_scores
                ])
                older_sentiment = np.mean([
                    recommendation_scores.get(rating, 0) * count
                    for rating, count in recent_recommendations.head(5).mean().items()
                    if rating in recommendation_scores
                ])

                if recent_sentiment > older_sentiment + 0.1:
                    trend = 'improving'
                elif recent_sentiment < older_sentiment - 0.1:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'

            return {
                'score': analyst_sentiment,
                'count': len(sentiment_scores),
                'trend': trend,
                'recent_score': recent_sentiment if len(recent_recommendations) >= 10 else analyst_sentiment
            }

        except Exception as e:
            self.logger.error("Failed to analyze analyst sentiment", symbol=symbol, error=str(e))
            return {'score': 0.0, 'count': 0, 'trend': 'neutral'}

    def _combine_sentiment_scores(self, news_sentiment: Dict,
                                  social_sentiment: Dict,
                                  analyst_sentiment: Dict) -> Dict:
        """Combine different sentiment sources into overall score."""
        try:
            # Weights for different sentiment sources
            weights = {
                'news': 0.4,
                'social': 0.3,
                'analyst': 0.3
            }

            # Extract scores
            news_score = news_sentiment.get('score', 0.0)
            social_score = social_sentiment.get('score', 0.0)
            analyst_score = analyst_sentiment.get('score', 0.0)

            # Calculate weighted average
            overall_sentiment = (
                    weights['news'] * news_score +
                    weights['social'] * social_score +
                    weights['analyst'] * analyst_score
            )

            # Calculate confidence based on availability and consistency of data
            confidence_factors = []

            # News confidence
            if news_sentiment.get('count', 0) > 5:
                confidence_factors.append(0.8)
            elif news_sentiment.get('count', 0) > 0:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.1)

            # Social confidence (placeholder)
            if social_sentiment.get('volume', 0) > 1000:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.4)

            # Analyst confidence
            if analyst_sentiment.get('count', 0) > 10:
                confidence_factors.append(0.9)
            elif analyst_sentiment.get('count', 0) > 0:
                confidence_factors.append(0.6)
            else:
                confidence_factors.append(0.2)

            confidence = np.mean(confidence_factors)

            # Determine overall trend
            trend_scores = [
                news_sentiment.get('score', 0),
                social_sentiment.get('score', 0),
                analyst_sentiment.get('recent_score', analyst_sentiment.get('score', 0))
            ]

            if np.mean(trend_scores) > 0.2:
                trend = 'positive'
            elif np.mean(trend_scores) < -0.2:
                trend = 'negative'
            else:
                trend = 'neutral'

            # Extract key themes (placeholder)
            themes = self._extract_themes(news_sentiment, social_sentiment, analyst_sentiment)

            return {
                'overall': overall_sentiment,
                'confidence': confidence,
                'trend': trend,
                'themes': themes,
                'component_scores': {
                    'news': news_score,
                    'social': social_score,
                    'analyst': analyst_score
                }
            }

        except Exception as e:
            self.logger.error("Failed to combine sentiment scores", error=str(e))
            return {
                'overall': 0.0,
                'confidence': 0.0,
                'trend': 'neutral',
                'themes': []
            }

    def _extract_themes(self, news_sentiment: Dict,
                        social_sentiment: Dict,
                        analyst_sentiment: Dict) -> List[str]:
        """Extract key themes from sentiment analysis."""
        themes = []

        # Analyze news articles for common themes
        articles = news_sentiment.get('articles', [])
        if articles:
            # Simple keyword analysis
            all_titles = ' '.join([article.get('title', '') for article in articles])
            common_themes = [
                'earnings', 'revenue', 'growth', 'acquisition', 'merger',
                'partnership', 'product', 'expansion', 'innovation', 'competition',
                'regulation', 'lawsuit', 'dividend', 'buyback'
            ]

            for theme in common_themes:
                if theme in all_titles.lower():
                    themes.append(theme)

        # Add trend-based themes
        if analyst_sentiment.get('trend') == 'improving':
            themes.append('analyst_upgrade')
        elif analyst_sentiment.get('trend') == 'declining':
            themes.append('analyst_downgrade')

        # Add sentiment strength themes
        overall_sentiment = (
                                    news_sentiment.get('score', 0) +
                                    social_sentiment.get('score', 0) +
                                    analyst_sentiment.get('score', 0)
                            ) / 3

        if overall_sentiment > 0.5:
            themes.append('very_bullish')
        elif overall_sentiment > 0.2:
            themes.append('bullish')
        elif overall_sentiment < -0.5:
            themes.append('very_bearish')
        elif overall_sentiment < -0.2:
            themes.append('bearish')

        return list(set(themes))  # Remove duplicates

    def _clean_text(self, text: str) -> str:
        """Clean text for sentiment analysis."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.lower()

    def _default_sentiment_result(self) -> Dict:
        """Return default sentiment result when analysis fails."""
        return {
            'overall_sentiment': 0.0,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'analyst_sentiment': 0.0,
            'sentiment_confidence': 0.0,
            'sentiment_trend': 'neutral',
            'news_count': 0,
            'key_themes': [],
            'analysis_timestamp': datetime.now()
        }