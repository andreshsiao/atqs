import os
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from taq.TAQQuotesReader import TAQQuotesReader



class VolumeModel:
    def __init__(self, feature_dir: str, target_ticker: str = None):
        """
        Initialize the model with the directory where feature matrices are stored.
        """
        self.feature_dir = feature_dir
        self.target_ticker = target_ticker
        self.volume_data = self._load_volume_data()


    def _load_volume_data(self) -> Dict[str, pd.DataFrame]:
        """
        Recursively load and aggregate quote volume data from all binRQ files.

        Returns:
            Dict[str, pd.DataFrame]: Keys are stock tickers; values are DataFrames
            indexed by date with 13 columns (volume buckets).
        """
        volume_data = {}

        print("Starting to load volume data...")

        for root, _, files in os.walk(self.feature_dir):
            for file in files:
                if not file.endswith("_quotes.binRQ"):
                    continue

                symbol = file.split("_")[0]
                if self.target_ticker and symbol != self.target_ticker:
                    continue

                print(f"Processing {file} from {root.split('/')[-1]}...")

                reader = TAQQuotesReader(os.path.join(root, file))
                n = reader.getN()
                timestamps = [reader.getMillisFromMidn(i) for i in range(n)]
                sizes = [reader.getBidSize(i) + reader.getAskSize(i) for i in range(n)]
                folder_name = os.path.basename(root)
                base_date = pd.to_datetime(folder_name)
                datetimes = [base_date + pd.to_timedelta(ms, unit="ms") for ms in timestamps]
                df = pd.DataFrame({"time": datetimes, "size": sizes})

                df["bucket"] = ((df["time"].dt.hour * 60 + df["time"].dt.minute - 570) // 30).clip(0, 12)
                df["date"] = df["time"].dt.date

                grouped = df.groupby(["date", "bucket"])["size"].sum().reset_index()
                pivoted = grouped.pivot(index="date", columns="bucket", values="size").fillna(0)

                # Ensure all 13 buckets exist
                pivoted = pivoted.reindex(columns=range(13), fill_value=0)

                # Normalize to get fractional volume
                normalized = pivoted.div(pivoted.sum(axis=1), axis=0)

                if symbol in volume_data:
                    volume_data[symbol] = pd.concat([volume_data[symbol], normalized])
                else:
                    volume_data[symbol] = normalized

        print("Finished loading volume data.")

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
            # Instead of averaging normalized fractions, sum raw volumes per bucket across days
            total_volume_per_bucket = df.sum(axis=0)
            all_profiles.append(total_volume_per_bucket)

        if not all_profiles:
            raise ValueError("No valid volume profiles found.")

        avg_profile = pd.concat(all_profiles, axis=1).sum(axis=1)
        avg_profile /= avg_profile.sum()  # Normalize to sum to 1
        avg_profile.index.name = "bucket"
        # TAG: Terminal logic point for static profile computation. Tested in test_static_profile.
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
            # TAG: Terminal logic point for fitting conditional profiles. Tested in test_fit_and_predict.

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

        result = list(np.array(prediction) / np.sum(prediction))
        # TAG: Terminal logic point for dynamic prediction. Tested in test_fit_and_predict.
        return result

    def evaluate_static_model(self, train_frac=0.8):
        """
        Evaluate static profile accuracy using train-test split.
        Returns MAE, RMSE, and R2 on the test set.
        Also saves a bar plot of mean error per bucket.
        """

        all_profiles = []
        for symbol, df in self.volume_data.items():
            if df.shape[1] == 13:
                all_profiles.append(df)

        if not all_profiles:
            raise ValueError("No valid volume profiles found for evaluation.")

        volume_df = pd.concat(all_profiles)
        train = volume_df.iloc[:int(train_frac * len(volume_df))]
        test = volume_df.iloc[int(train_frac * len(volume_df)):]

        static_profile_train = train.mean(axis=0)
        preds = np.tile(static_profile_train.values, (len(test), 1))

        mae = mean_absolute_error(test, preds)
        rmse = np.sqrt(mean_squared_error(test, preds))
        r2 = r2_score(test, preds)

        # Plot bucket-wise mean error
        bucket_error = (test - preds).mean(axis=0)
        bucket_error.plot(kind="bar", title="Mean Error by Time Bucket")
        plt.xlabel("Bucket")
        plt.ylabel("Mean Error")
        plt.tight_layout()
        plt.savefig("volume_model_error_by_bucket.png")
        plt.close()

        # TAG: Terminal logic point for evaluation metrics. Tested in test_static_model_evaluation.
        return mae, rmse, r2

    def get_date_split(self, train_frac=0.8):
        """
        Returns a sorted list of all available dates and a train/test split.
        """
        all_dates = []
        for df in self.volume_data.values():
            all_dates.extend(df.index.tolist())
        unique_dates = sorted(set(all_dates))
        split_idx = int(train_frac * len(unique_dates))
        return unique_dates[:split_idx], unique_dates[split_idx:]

    def get_static_profile_by_dates(self, train_dates: List):
        """
        Compute a static profile from only the specified training dates.
        """
        # Ensure train_dates are datetime.date objects for proper filtering
        train_dates = [pd.to_datetime(d).date() for d in train_dates]
        all_profiles = []
        for symbol, df in self.volume_data.items():
            filtered = df[df.index.isin(train_dates)]
            if not filtered.empty:
                print(f"Using {symbol} for static profile: {filtered.shape}")
                # Instead of averaging normalized fractions, sum raw volumes per bucket across days
                total_volume_per_bucket = filtered.sum(axis=0)
                all_profiles.append(total_volume_per_bucket)
        if not all_profiles:
            raise ValueError("No valid training data for static profile.")
        avg_profile = pd.concat(all_profiles, axis=1).sum(axis=1)
        avg_profile /= avg_profile.sum()
        avg_profile.index.name = "bucket"
        return avg_profile