"""
Alerting system — webhook/email notifications for live trading events.

Alert levels:
  INFO     — non-urgent (idle detection, single model degradation)
  WARNING  — moderate (Sharpe degraded, drawdown approaching limit)
  CRITICAL — immediate action required (kill switch, margin call, all models unhealthy)

Integration:
  alert_manager = AlertManager(AlertConfig(discord_webhook_url="..."))
  alert_manager.check_and_notify(metrics, session_id)
"""
from __future__ import annotations

import json
import logging
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    enabled: bool = True

    discord_webhook_url: str = ""
    slack_webhook_url: str = ""
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: list = field(default_factory=list)
    email_from: str = ""

    sharpe_degradation_threshold: float = -0.3
    hitrate_degradation_threshold: float = 0.30
    drawdown_alert_pct: float = -0.10
    kill_switch_notify: bool = True
    regime_idle_hours: float = 24.0
    daily_pnl_alert_threshold: float = -500.0
    equity_floor_pct: float = 0.50

    cooldown_info: int = 3600
    cooldown_warning: int = 900
    cooldown_critical: int = 300


class AlertManager:
    def __init__(self, config: AlertConfig | None = None):
        self.config = config or AlertConfig()
        self._last_sent: Dict[str, float] = {}

    def _in_cooldown(self, key: str, level: str) -> bool:
        cd = {"info": self.config.cooldown_info,
              "warning": self.config.cooldown_warning,
              "critical": self.config.cooldown_critical}.get(level, 3600)
        last = self._last_sent.get(key, 0)
        return (time.time() - last) < cd

    def _mark_sent(self, key: str):
        self._last_sent[key] = time.time()

    def check_and_notify(
        self,
        metrics: dict,
        session_id: str,
        engine_state: dict | None = None,
    ):
        if not self.config.enabled:
            return

        # 1. Per-model Sharpe degradation
        for model, health in metrics.get("per_model_health", {}).items():
            sr = health.get("rolling_sharpe")
            if sr is not None and sr < self.config.sharpe_degradation_threshold:
                key = f"sharpe:{session_id}:{model}"
                if not self._in_cooldown(key, AlertLevel.WARNING):
                    self._send(AlertLevel.WARNING,
                               f"[{session_id}] {model} Sharpe degraded: {sr:.2f} "
                               f"(threshold: {self.config.sharpe_degradation_threshold})")
                    self._mark_sent(key)

        # 2. Committee marked unhealthy
        if not metrics.get("committee_healthy", True):
            key = f"committee:unhealthy:{session_id}"
            if not self._in_cooldown(key, AlertLevel.CRITICAL):
                self._send(AlertLevel.CRITICAL,
                           f"[{session_id}] Committee unhealthy — >50% models degraded. "
                           "Trades automatically suppressed.")
                self._mark_sent(key)

        # 3. Kill switch (from engine state)
        if engine_state and engine_state.get("killed") and self.config.kill_switch_notify:
            key = f"kill:{session_id}"
            if not self._in_cooldown(key, AlertLevel.CRITICAL):
                reason = engine_state.get("kill_reason", "unknown")
                level = engine_state.get("kill_level", "unknown")
                self._send(AlertLevel.CRITICAL,
                           f"[{session_id}] KILL SWITCH (Level {level}): {reason}")
                self._mark_sent(key)

        # 4. Drawdown alert (from engine state)
        if engine_state:
            equity = engine_state.get("equity", 0)
            initial = getattr(self.config, "initial_equity", 10000) or 10000
            if initial > 0 and equity > 0:
                dd_pct = (equity / initial) - 1.0
                if dd_pct < self.config.drawdown_alert_pct:
                    key = f"dd:{session_id}"
                    if not self._in_cooldown(key, AlertLevel.WARNING):
                        self._send(AlertLevel.WARNING,
                                   f"[{session_id}] Drawdown: {dd_pct*100:.1f}% "
                                   f"(threshold: {self.config.drawdown_alert_pct*100:.0f}%)")
                        self._mark_sent(key)

    def _send(self, level: str, message: str):
        payloads = []

        if self.config.discord_webhook_url:
            payloads.append(("discord", self.config.discord_webhook_url,
                             {"content": f"**[{level.upper()}]** {message}"}))

        if self.config.slack_webhook_url:
            payloads.append(("slack", self.config.slack_webhook_url,
                             {"text": f"[{level.upper()}] {message}"}))

        for channel, url, payload in payloads:
            try:
                import requests
                resp = requests.post(url, json=payload, timeout=5)
                if resp.status_code >= 400:
                    logger.warning("Alert %s failed: HTTP %d — %s",
                                   channel, resp.status_code, resp.text[:200])
            except Exception:
                logger.exception("Alert %s failed", channel)

        if self.config.email_recipients and self.config.email_smtp_host:
            try:
                msg = MIMEText(message, "plain", "utf-8")
                msg["Subject"] = f"[{level.upper()}] KodaQuant Alert"
                msg["From"] = self.config.email_from or "alerts@kodaquant.local"
                msg["To"] = ", ".join(self.config.email_recipients)
                with smtplib.SMTP(self.config.email_smtp_host,
                                  self.config.email_smtp_port, timeout=10) as s:
                    s.starttls()
                    if self.config.email_username:
                        s.login(self.config.email_username, self.config.email_password)
                    s.send_message(msg)
            except Exception:
                logger.exception("Alert email failed")
