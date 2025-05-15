import numpy as np
import pandas as pd

# This class represents a simple static execution strategy.
# Given a static volume profile (e.g., average volume per bucket) and the total shares to execute,
# it allocates trading quantities across all time buckets proportionally to the profile.

class StaticSimulator:
    def __init__(self, volume_profile: pd.Series, total_shares: int):
        """
        Initialize with a static volume profile and total shares to execute.

        Args:
            volume_profile (pd.Series): fractional volume per bucket (length 13, sums to 1)
            total_shares (int): total number of shares to trade
        """
        # Store the input static profile and the total number of shares to be traded.
        self.volume_profile = volume_profile
        self.total_shares = total_shares
        # TAG: passing_logic_store_input_parameters

    def simulate(self) -> pd.Series:
        """
        Simulate execution using the static profile.

        Returns:
            pd.Series: executed shares per bucket (length 13)
        """
        # Compute the number of shares to trade in each bucket by scaling the static profile by total shares.
        # This assumes perfect volume forecasting and no adjustment throughout the day.
        executed = self.volume_profile * self.total_shares
        # TAG: terminal_logic_output_rounding_and_return
        return executed.round().astype(int)