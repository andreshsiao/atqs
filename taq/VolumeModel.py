import os
import numpy as np
import pandas as pd
from taq.TAQQuotesReader import TAQQuotesReader
from datetime import datetime
from typing import List, Dict


class VolumeModel:
    def __init__(self, feature_dir: str):
        """
        Initialize the model with the directory where feature matrices are stored.
        """
        self.feature_dir = feature_dir
        self.volume_data = self._load_volume_data()


    def _load_volume_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and aggregate quote volume data from all binRQ files.

        Returns:
            Dict[str, pd.DataFrame]: Keys are stock tickers; values are DataFrames
            indexed by date with 13 columns (volume buckets).
        """
        quote_dir = self.feature_dir  # should point to quotes/extracted/binRQ/
        files = [f for f in os.listdir(quote_dir) if f.endswith("_quotes.binRQ")]
        volume_data = {}

        for file in files:
            symbol = file.split("_")[0]
            reader = TAQQuotesReader(os.path.join(quote_dir, file))
            timestamps = reader.get_nanos() // 1_000_000_000  # nanoseconds to seconds
            sizes = reader.get_size()
            datetimes = pd.to_datetime(timestamps, unit="s")
            df = pd.DataFrame({"time": datetimes, "size": sizes})

            df["bucket"] = ((df["time"].dt.hour * 60 + df["time"].dt.minute - 570) // 30).clip(0, 12)
            df["date"] = df["time"].dt.date

            grouped = df.groupby(["date", "bucket"]).sum().reset_index()
            pivoted = grouped.pivot(index="date", columns="bucket", values="size").fillna(0)

            # Normalize to get fractional volume
            normalized = pivoted.div(pivoted.sum(axis=1), axis=0)
            volume_data[symbol] = normalized

        return volume_data

    def get_static_profile(self) -> pd.Series:
        """
        Compute a static volume profile over 13 half-hour intervals.
        Averages the volume fraction across all available dates and symbols.

        Returns:
            pd.Series: average fractional volume per bucket (length 13), normalized to sum to 1
        """
        all_profiles = []

        for symbol, df in self.volume_data.items():
            if df.shape[1] != 13:
                continue  # Skip if unexpected number of buckets
            all_profiles.append(df.mean(axis=0))  # average over days for this symbol

        if not all_profiles:
            raise ValueError("No valid volume profiles found.")

        avg_profile = pd.concat(all_profiles, axis=1).mean(axis=1)
        avg_profile /= avg_profile.sum()  # Normalize to sum to 1
        avg_profile.index.name = "bucket"
        return avg_profile

    def fit_dynamic_model(self, volume_by_bucket: pd.DataFrame):
        """
        Fit a conditional expectation model to estimate expected remaining volume
        fractions given observed volume up to the current bucket.
        Stores the conditional average profiles by prefix volume patterns.
        """
        self.conditional_profiles = {}

        for i in range(1, 13):  # for each possible current bucket (1 to 12)
            prefix = volume_by_bucket.iloc[:, :i]
            suffix = volume_by_bucket.iloc[:, i:]

            # Round prefixes to 2 decimals to group similar early profiles
            grouped = prefix.round(2).apply(tuple, axis=1)
            suffix_mean_by_prefix = {}

            for key in grouped.unique():
                idx = grouped[grouped == key].index
                suffix_mean = suffix.loc[idx].mean()
                suffix_mean_by_prefix[key] = suffix_mean.values

            self.conditional_profiles[i] = suffix_mean_by_prefix

    def predict_remaining_volume(self, observed: List[float], current_bucket: int) -> List[float]:
        """
        Predict the remaining volume profile given observed volumes.

        Args:
            observed (List[float]): observed volume fractions so far.
            current_bucket (int): current time bucket index (0-based).

        Returns:
            List[float]: predicted remaining volume fractions for remaining buckets.
        """
        if not hasattr(self, "conditional_profiles"):
            raise RuntimeError("Call fit_dynamic_model before prediction.")

        prefix = tuple(np.round(observed, 2).tolist())
        suffix_map = self.conditional_profiles.get(current_bucket, {})
        prediction = suffix_map.get(prefix)

        if prediction is None:
            all_suffixes = list(suffix_map.values())
            if all_suffixes:
                prediction = np.mean(all_suffixes, axis=0)
            else:
                num_remaining = 13 - current_bucket
                prediction = [1 / num_remaining] * num_remaining

        return list(np.array(prediction) / np.sum(prediction))