"""Email service for sending notifications and reports."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiosmtplib
from jinja2 import Environment, FileSystemLoader
from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import os

from app.config.settings import settings
from app.utils.logger import email_logger
from app.models.database_models import Trade, EmailLog

class EmailService:
    """Service for sending emails and notifications."""

    def __init__(self):
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_username = settings.smtp_username
        self.smtp_password = settings.smtp_password
        self.email_from = settings.email_from
        self.logger = email_logger

        # Setup Jinja2 environment for templates
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates', 'email')
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))

    async def send_trade_confirmation(self, trade_id: str, recipient: Optional[str] = None):
        """Send trade confirmation email."""
        try:
            # Get trade details (you'd fetch from database)
            # For now, using mock data
            trade_data = {
                'trade_id': trade_id,
                'symbol': 'AAPL',
                'strategy': 'Covered Call',
                'entry_date': datetime.now(),
                'legs': [
                    {
                        'action': 'SELL',
                        'option_type': 'CALL',
                        'strike': 150.0,
                        'expiry': '2024-02-15',
                        'quantity': 1
                    }
                ]
            }

            subject = f"Trade Confirmation - {trade_data['symbol']} {trade_data['strategy']}"

            # Render email template
            template = self.jinja_env.get_template('trade_confirmation.html')
            html_content = template.render(trade=trade_data)

            # Send email
            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='trade_confirmation'
            )

            self.logger.info("Trade confirmation email sent", trade_id=trade_id)

        except Exception as e:
            self.logger.error("Failed to send trade confirmation", trade_id=trade_id, error=str(e))
            raise

    async def send_trade_closure_notification(self, trade_id: str, recipient: Optional[str] = None):
        """Send trade closure notification."""
        try:
            # Mock trade data
            trade_data = {
                'trade_id': trade_id,
                'symbol': 'AAPL',
                'strategy': 'Covered Call',
                'entry_date': datetime.now() - timedelta(days=30),
                'exit_date': datetime.now(),
                'pnl': 250.50,
                'pnl_percentage': 12.75,
                'exit_reason': 'Profit Target Hit'
            }

            subject = f"Trade Closed - {trade_data['symbol']} {trade_data['strategy']} (${trade_data['pnl']:.2f})"

            template = self.jinja_env.get_template('trade_closure.html')
            html_content = template.render(trade=trade_data)

            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='trade_closure'
            )

            self.logger.info("Trade closure email sent", trade_id=trade_id)

        except Exception as e:
            self.logger.error("Failed to send trade closure notification", trade_id=trade_id, error=str(e))
            raise

    async def send_daily_report(self, recipient: Optional[str] = None):
        """Send daily performance report."""
        try:
            # Generate daily report data
            report_data = await self._generate_daily_report_data()

            subject = f"Daily Trading Report - {datetime.now().strftime('%Y-%m-%d')}"

            template = self.jinja_env.get_template('daily_report.html')
            html_content = template.render(report=report_data)

            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='daily_report'
            )

            self.logger.info("Daily report email sent")

        except Exception as e:
            self.logger.error("Failed to send daily report", error=str(e))
            raise

    async def send_weekly_report(self, recipient: Optional[str] = None):
        """Send weekly performance report."""
        try:
            # Generate weekly report data
            report_data = await self._generate_weekly_report_data()

            subject = f"Weekly Trading Report - Week of {datetime.now().strftime('%Y-%m-%d')}"

            template = self.jinja_env.get_template('weekly_report.html')
            html_content = template.render(report=report_data)

            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='weekly_report'
            )

            self.logger.info("Weekly report email sent")

        except Exception as e:
            self.logger.error("Failed to send weekly report", error=str(e))
            raise

    async def send_monthly_report(self, recipient: Optional[str] = None):
        """Send monthly performance report."""
        try:
            # Generate monthly report data
            report_data = await self._generate_monthly_report_data()

            subject = f"Monthly Trading Report - {datetime.now().strftime('%B %Y')}"

            template = self.jinja_env.get_template('monthly_report.html')
            html_content = template.render(report=report_data)

            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='monthly_report'
            )

            self.logger.info("Monthly report email sent")

        except Exception as e:
            self.logger.error("Failed to send monthly report", error=str(e))
            raise

    async def send_alert_notification(self, alert_type: str, message: str,
                                      details: Dict = None, recipient: Optional[str] = None):
        """Send alert notification."""
        try:
            subject = f"Trading Alert: {alert_type}"

            template = self.jinja_env.get_template('alert_notification.html')
            html_content = template.render(
                alert_type=alert_type,
                message=message,
                details=details or {},
                timestamp=datetime.now()
            )

            await self._send_email(
                to_email=recipient or self.email_from,
                subject=subject,
                html_content=html_content,
                email_type='alert_notification'
            )

            self.logger.info("Alert notification sent", alert_type=alert_type)

        except Exception as e:
            self.logger.error("Failed to send alert notification", alert_type=alert_type, error=str(e))
            raise

    async def _send_email(self, to_email: str, subject: str, html_content: str,
                          email_type: str, attachments: List[str] = None):
        """Send email using aiosmtplib."""
        try:
            # Create message
            message = MIMEMultipart('alternative')
            message['From'] = self.email_from
            message['To'] = to_email
            message['Subject'] = subject

            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            message.attach(html_part)

            # Add attachments if any
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            message.attach(part)

            # Send email
            await aiosmtplib.send(
                message,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_username,
                password=self.smtp_password,
                use_tls=True
            )

            # Log email
            await self._log_email(to_email, subject, email_type, 'SENT')

        except Exception as e:
            # Log failed email
            await self._log_email(to_email, subject, email_type, 'FAILED', str(e))
            raise

    async def _log_email(self, recipient: str, subject: str, email_type: str,
                         status: str, error_message: str = None):
        """Log email sending attempt."""
        try:
            # This would typically save to database
            # For now, just log to file
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'recipient': recipient,
                'subject': subject,
                'email_type': email_type,
                'status': status,
                'error_message': error_message
            }

            self.logger.info("Email logged", **log_entry)

        except Exception as e:
            self.logger.error("Failed to log email", error=str(e))

    async def _generate_daily_report_data(self) -> Dict:
        """Generate data for daily report."""
        # Mock data - replace with actual database queries
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': 5,
            'winning_trades': 3,
            'losing_trades': 2,
            'total_pnl': 1250.75,
            'win_rate': 60.0,
            'best_trade': {'symbol': 'AAPL', 'pnl': 450.25},
            'worst_trade': {'symbol': 'TSLA', 'pnl': -125.50},
            'portfolio_value': 105425.50,
            'active_positions': 8
        }

    async def _generate_weekly_report_data(self) -> Dict:
        """Generate data for weekly report."""
        # Mock data - replace with actual database queries
        return {
            'week_start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'week_end': datetime.now().strftime('%Y-%m-%d'),
            'total_trades': 23,
            'winning_trades': 15,
            'losing_trades': 8,
            'total_pnl': 3275.25,
            'win_rate': 65.2,
            'sharpe_ratio': 1.85,
            'max_drawdown': -450.75,
            'best_strategy': {'name': 'Covered Call', 'pnl': 1825.50},
            'portfolio_growth': 3.2
        }

    async def _generate_monthly_report_data(self) -> Dict:
        """Generate data for monthly report."""
        # Mock data - replace with actual database queries
        return {
            'month': datetime.now().strftime('%B %Y'),
            'total_trades': 95,
            'winning_trades': 62,
            'losing_trades': 33,
            'total_pnl': 12850.75,
            'win_rate': 65.3,
            'monthly_return': 12.8,
            'annual_return': 154.2,
            'sharpe_ratio': 2.1,
            'max_drawdown': -1250.25,
            'top_performers': [
                {'symbol': 'AAPL', 'pnl': 2350.50},
                {'symbol': 'MSFT', 'pnl': 1875.25},
                {'symbol': 'GOOGL', 'pnl': 1425.75}
            ],
            'strategy_performance': [
                {'name': 'Covered Call', 'trades': 45, 'pnl': 6250.25},
                {'name': 'Iron Condor', 'trades': 30, 'pnl': 3825.50},
                {'name': 'Bull Put Spread', 'trades': 20, 'pnl': 2775.00}
            ]
        }