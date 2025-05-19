#!/usr/bin/env python3
"""
Daily stock scanner script - can be run as a scheduled job.
"""

import sys
import os
from pathlib import Path
import asyncio
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.ai.stock_scanner import StockScanner
from app.services.email_service import EmailService
from app.utils.logger import get_logger

logger = get_logger("daily_scanner")

async def run_daily_scan():
    """Run daily stock scan and send results."""
    try:
        logger.info("Starting daily stock scan...")

        # Initialize scanner
        scanner = StockScanner()

        # Run scan for all strategies
        strategies = ['covered_call', 'iron_condor', 'bull_put_spread']
        results = await scanner.scan_universe(strategies=strategies)

        # Get top opportunities
        top_opportunities = results[:10]

        logger.info(f"Scan completed. Found {len(results)} opportunities")

        # Send email with results
        email_service = EmailService()
        await email_service.send_daily_scan_results(top_opportunities)

        # Log top opportunities
        for i, opportunity in enumerate(top_opportunities, 1):
            logger.info(f"Top {i}: {opportunity['symbol']} - Score: {opportunity['overall_score']:.3f}")

        return True

    except Exception as e:
        logger.error(f"Daily scan failed: {e}")
        return False

async def main():
    """Main function."""
    success = await run_daily_scan()
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())