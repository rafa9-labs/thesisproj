# KodaQuant LEAN Bridge Algorithm
# Minimal passthrough algorithm that initializes Interactive Brokers brokerage
# and subscribes to forex data. All trade decisions arrive externally via the
# LEAN Engine REST API (POST /live/commands).
#
# LEAN executes SubmitOrderCommand, CancelOrderCommand, and LiquidateCommand
# natively through its core engine — no custom OnCommand handler needed.

from AlgorithmImports import *


class KodaQuantBridge(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)

        # --- Brokerage ---
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
        self.SetAccountCurrency("USD")

        # --- Cash ---
        self.SetCash(10000)

        # --- Data subscriptions ---
        # LEAN needs at least one data subscription to run the event loop and
        # track portfolio prices. The OnData handler is intentionally empty —
        # all signals come from external REST API commands.
        self.eurusd = self.AddForex("EURUSD", Resolution.Hour, Market.Oanda)
        self.gbpusd = self.AddForex("GBPUSD", Resolution.Hour, Market.Oanda)
        self.usdjpy = self.AddForex("USDJPY", Resolution.Hour, Market.Oanda)

        # --- Warm-up ---
        self.SetWarmUp(TimeSpan.FromDays(7))

        self.Debug("KodaQuant LEAN Bridge initialized. Awaiting external commands.")

    def OnData(self, data):
        pass
