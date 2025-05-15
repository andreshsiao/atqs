# This script implements a full pipeline for evaluating a guaranteed VWAP execution strategy
# using a static and dynamic simulator and quote volume data. It reads TAQ quote data, computes
# volume profiles, simulates execution, and evaluates slippage-adjusted execution cost.

import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from taq.MyDirectories import MyDirectories, BASE_PATH
from taq.DataProcessor import DataProcessor
from taq.TAQQuotesReader import TAQQuotesReader
from taq.NLSEstimator import NLSImpactEstimator
from taq.Utils import extract_tar_files, extract_all_quotes, get_stock_list  # Importing from Utils

def evaluate_guaranteed_vwap_charge(volume_model, simulator_cls, client_ticker, client_shares, lam, static_profile, test_dates):
    client_charge_total = 0
    first_day_details = None

    # Loop through test dates to simulate execution on each day and accumulate client charge
    for idx, test_date in enumerate(test_dates):
        try:
            day_profile = volume_model.volume_data[client_ticker].loc[test_date]
        except KeyError:
            continue

        volume_std = 0.05
        noise = np.random.normal(0, volume_std, size=13)

        # Add noise to the static volume profile to simulate imperfect forecasts
        noisy_profile = np.clip(static_profile + noise, 0, None)
        noisy_profile /= noisy_profile.sum()

        # Build volume forecasts per bucket
        volume_forecasts = [list(noisy_profile[i:]) for i in range(13)]

        # Normalize observed volume
        observed = list(day_profile.values)
        observed = observed / np.sum(observed)

        # Instantiate the simulator and run execution simulation using forecasts and observed volumes
        simulator = simulator_cls(initial_shares=client_shares, eta=0.142, sigma=0.02, lam=lam, mode="sinh", static_profile=static_profile)
        trades = simulator.simulate(volume_forecasts, observed_volumes=observed)

        trades = np.maximum(trades, 0)
        total_traded = trades.sum()
        remaining = client_shares - total_traded
        if remaining > 0:
            trades.iloc[-1] += remaining

        # Compute execution prices and evaluate cost based on VWAP benchmark
        midquotes = np.linspace(100, 101, 13)
        details_df = simulator.get_execution_details(trades, midquotes)
        exec_prices = details_df["Exec Price"].values
        ewap = np.sum(trades * exec_prices) / np.sum(trades)
        vwap_prices = midquotes
        vwap = np.sum(day_profile * vwap_prices) / np.sum(day_profile)
        slippage = ewap - vwap
        cost = ewap + lam * slippage
        client_charge_total += cost

        # Print and save detailed bucket execution summary for one representative day
        if first_day_details is None and idx == 12:
            print(f"\n--- Bucket Execution Summary (First Test Day: {test_date}) ---")
            print(details_df.to_string(index=False))
            details_df.to_csv("test_day_bucket_log.csv", index=False)

    return client_charge_total / len(test_dates)

def main():
    # === Data Extraction Phase ===
    # Unpack raw quote and trade data from tar archives
    quotes_extract_dir = MyDirectories.getQuotesDir()
    quotes_tar_dir = os.path.join(quotes_extract_dir, "..")
    extract_tar_files(quotes_tar_dir, quotes_extract_dir)

    trades_extract_dir = MyDirectories.getQuotesDir()
    trades_tar_dir = os.path.join(trades_extract_dir, "..")
    extract_tar_files(trades_tar_dir, trades_extract_dir)

    # === Simulation and Evaluation Phase ===
    # Set up volume model, run simulator, and evaluate guaranteed VWAP quote
    from taq.VolumeModel import VolumeModel
    from taq.DynamicSimulator import DynamicSimulator

    quotes_path = os.path.join(BASE_PATH, "../data/quotes/extracted")
    client_ticker = "GE"
    client_shares = 10000
    volume_model = VolumeModel(quotes_path, target_ticker=client_ticker)

    # Split the available data into training and testing periods
    train_dates, test_dates = volume_model.get_date_split(train_frac=0.8)
    static_profile = volume_model.get_static_profile_by_dates(train_dates)

    # Run dynamic simulator on test data and compute average adjusted execution cost
    client_charge = evaluate_guaranteed_vwap_charge(
        volume_model, DynamicSimulator, client_ticker, client_shares, lam=0.01,
        static_profile=static_profile, test_dates=test_dates
    )

    # Output final VWAP guarantee quote for the client
    print(f"Guaranteed VWAP execution quote: ${round(client_charge, 4)} per share")
    print(f"(Client: {client_ticker}, Size: {client_shares} shares)")

if __name__ == "__main__":
    main()