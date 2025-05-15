#
# DynamicSimulator class

# This class simulates a dynamic execution strategy for trading a fixed number of shares over a day.
# It adapts execution speed based on observed market volume and forecasted remaining volume.
# The class supports both Almgren-Chriss and urgency-adjusted sinh-based trajectory methods.
# Key control logic:
#   - The first bucket always follows the static volume profile.
#   - Subsequent buckets adjust trading urgency based on deviation between observed and forecasted volume.
#   - A minimum trading threshold is enforced to avoid stalling execution.
#   - The final bucket trades all remaining shares to ensure full execution.

import numpy as np
import pandas as pd
from taq.ExecutionCostModel import AlmgrenChrissExecution

class DynamicSimulator:
    def __init__(self, initial_shares: int, eta: float = 0.01, sigma: float = 0.02, lam: float = 0.01, mode: str = "sinh", static_profile: list = None):
        """
        Initialize the dynamic simulator.

        Args:
            initial_shares (int): total shares to execute by end of day.
            eta (float): impact cost coefficient.
        """
        self.initial_shares = initial_shares
        self.eta = eta
        self.sigma = sigma
        self.lam = lam
        self.mode = mode
        self.remaining_shares = initial_shares
        self.execution_log = []
        self.static_profile = static_profile if static_profile is not None else []
        self.static_profile_sum = sum(self.static_profile)

    def _sinh_trajectory(self, remaining_shares: int, gamma: float, horizon: int) -> np.ndarray:
        """
        Compute a sinh-based execution trajectory over the remaining horizon.

        Args:
            remaining_shares (int): total shares left to execute.
            gamma (float): trajectory shape parameter.
            horizon (int): number of remaining buckets.

        Returns:
            np.ndarray: number of shares to trade in each bucket (length = horizon).
        """
        t = np.arange(1, horizon + 1)
        weights = np.sinh(gamma * (horizon - t + 1))
        weights /= weights.sum()
        return np.round(weights * remaining_shares).astype(int)

    # Computes a trading trajectory that adjusts urgency based on deviation ratio.
    # A higher urgency leads to faster execution in earlier buckets.
    # This serves as the core of the dynamic adjustment mechanism.
    def _urgency_adjusted_sinh_trajectory(self, remaining_shares: int, deviation_ratio: float, horizon: int, base_weights: np.ndarray = None, base_gamma: float = 1.2, alpha: float = 1.0) -> np.ndarray:
        urgency = base_gamma - alpha * deviation_ratio
        urgency = np.clip(urgency, 0.5, 5.0)
        if base_weights is None:
            t = np.arange(1, horizon + 1)
            weights = np.sinh(urgency * (horizon - t + 1))
        else:
            base_weights = np.array(base_weights)
            weights = base_weights * np.sinh(urgency * np.arange(horizon, 0, -1))
        weights /= weights.sum()
        return np.round(weights * remaining_shares).astype(int)

    # Main simulation loop
    # Iterates through each time bucket, executing trades based on current forecasts and past observations.
    # First bucket uses static allocation. Later buckets apply adaptive strategies.
    # Key pivot logic includes deviation-driven urgency adjustment and enforcement of constraints.
    def simulate(self, volume_forecasts: list, observed_volumes: list = None):
        """
        Run dynamic execution simulation based on forecasted volume fractions.

        Args:
            volume_forecasts (list of list): per-step volume forecasts (each is a list of future fractions).
            observed_volumes (list): actual observed market volume fractions to update expectations (optional).

        Returns:
            pd.Series: executed shares per bucket.
        """
        executed = []

        for t, forecast in enumerate(volume_forecasts):
            if t == 0:  # [TEST: test_simulation_output_length, test_total_shares_traded]
                static_weight = self.static_profile[0] if len(self.static_profile) > 0 else 1 / 13
                shares_to_trade = int(round(static_weight * self.initial_shares))
                self.remaining_shares -= shares_to_trade
                executed.append(shares_to_trade)
                self.execution_log.append({
                    "bucket": t,
                    "forecast": forecast,
                    "executed": shares_to_trade,
                    "remaining": self.remaining_shares
                })
                continue

            if self.remaining_shares <= 0:  # [TEST: test_total_shares_traded]
                executed.append(0)
                continue

            remaining_fraction = sum(forecast)
            if remaining_fraction <= 0:  # [TEST: test_total_shares_traded]
                executed.append(0)
                continue

            horizon = len(forecast)

            # If we observe actual volumes, compute the cumulative deviation ratio
            if observed_volumes is not None and t > 0:  # [TEST: test_trajectory_shape]
                cum_observed = sum(observed_volumes[:t])
                cum_forecast_ratio = sum([sum(f) for f in volume_forecasts[:t]]) / len(volume_forecasts)
                deviation_ratio = (cum_observed - cum_forecast_ratio) / (cum_forecast_ratio + 1e-6)
            else:
                deviation_ratio = 0.0

            if self.mode == "ac":  # [TEST: test_ac_mode_total_shares_traded, test_ac_mode_output_length]
                ac = AlmgrenChrissExecution(
                    T=horizon / 13, N=horizon, X=self.remaining_shares,
                    eta=self.eta, beta=0.5, sigma=self.sigma, lam=self.lam
                )
                holdings = ac.optimal_trajectory()
                shares_to_trade = int(round(holdings[0] - holdings[1]))
            else:
                # Use adjusted urgency-based trajectory
                if len(self.static_profile[t:]) != horizon:
                    print(f"Warning: static_profile slice length {len(self.static_profile[t:])} != forecast horizon {horizon}")
                    executed.append(0)
                    continue
                base_weights = np.array(self.static_profile[t:])
                if base_weights.sum() == 0:
                    base_weights = np.ones(horizon) / horizon
                else:
                    base_weights = base_weights / base_weights.sum()
                trajectory = self._urgency_adjusted_sinh_trajectory(self.remaining_shares, deviation_ratio, horizon, base_weights=base_weights)

                shares_to_trade = trajectory[0]

            # Enforce minimum floor of 3% of remaining
            min_trade = max(int(0.05 * self.remaining_shares), 1)
            shares_to_trade = max(shares_to_trade, min_trade)

            if t == len(volume_forecasts) - 1 and self.remaining_shares > 0:  # [TEST: test_total_shares_traded]
                shares_to_trade = self.remaining_shares

            self.remaining_shares -= shares_to_trade
            executed.append(shares_to_trade)

            self.execution_log.append({
                "bucket": t,
                "forecast": forecast,
                "executed": shares_to_trade,
                "remaining": self.remaining_shares
            })

        return pd.Series(executed)
    # Summarizes execution statistics
    # Calculates impact, execution prices, and costs per bucket.
    # Helps evaluate how the strategy performed under given market assumptions.
    def get_execution_details(self, trades: pd.Series, midquotes: np.ndarray, daily_volume: int = 1_000_000):
        """
        Compute detailed execution stats (impacts, prices, costs) bucket by bucket.

        Args:
            trades (pd.Series): shares executed per bucket.
            midquotes (np.ndarray): midquote prices per bucket.
            daily_volume (int): assumed market volume for participation rate.

        Returns:
            pd.DataFrame: bucket-wise execution log with impact, price, and cost.
        """
        participation_rate = trades / daily_volume
        impacts = self.eta * participation_rate
        exec_prices = midquotes + impacts
        cost = trades * exec_prices

        return pd.DataFrame({
            "Bucket": np.arange(len(trades)),
            "Shares Traded": trades,
            "Midquote": midquotes,
            "Impact": impacts,
            "Exec Price": exec_prices,
            "Cost": cost
        })