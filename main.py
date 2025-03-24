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

def main():
    # Extract quote and trade data from tar files
    quotes_extract_dir = MyDirectories.getQuotesDir()
    quotes_tar_dir = os.path.join(quotes_extract_dir, "..")
    extract_tar_files(quotes_tar_dir, quotes_extract_dir)

    trades_extract_dir = MyDirectories.getQuotesDir()
    trades_tar_dir = os.path.join(trades_extract_dir, "..")
    extract_tar_files(trades_tar_dir, trades_extract_dir)

    # Initialize data processor
    processor = DataProcessor(quotes_extract_dir)

    feature_matrices = {
        "2min_returns": defaultdict(dict),
        "total_volume": defaultdict(dict),
        "arrival_price": defaultdict(dict),
        "imbalance": defaultdict(dict),
        "terminal_price": defaultdict(dict),
    }

    # Process each extracted date folder
    for date_folder in sorted(os.listdir(quotes_extract_dir)):
        date_path = os.path.join(quotes_extract_dir, date_folder)
        if not os.path.isdir(date_path):
            continue  # Skip if not a folder

        stock_list = get_stock_list(date_path)

        for stock in stock_list:
            print(f"Processing stock {date_folder}: {stock}")

            stock_file_path = os.path.join(date_path, f"{stock}_quotes.binRQ")
            reader = TAQQuotesReader(stock_file_path)  # Read binary data
            daily_data = extract_all_quotes(reader)

            processor.add_midquote_to_data(daily_data)

            # Compute required metrics
            two_minute_returns = processor.compute_midquote_returns(daily_data)
            total_volume = processor.compute_total_daily_volume(daily_data)
            arrival_price = processor.compute_arrival_price(daily_data)
            imbalance = processor.compute_imbalance(daily_data)
            terminal_price = processor.compute_terminal_price(daily_data)

            # Populate feature matrices
            feature_matrices["2min_returns"][stock][date_folder] = two_minute_returns
            feature_matrices["total_volume"][stock][date_folder] = total_volume
            feature_matrices["arrival_price"][stock][date_folder] = arrival_price
            feature_matrices["imbalance"][stock][date_folder] = imbalance
            feature_matrices["terminal_price"][stock][date_folder] = terminal_price

    # Save feature matrices to CSV
    feature_dir = os.path.join(BASE_PATH, "../data/feature_matrices")
    os.makedirs(feature_dir, exist_ok=True)

    for feature, matrix in feature_matrices.items():
        df = pd.DataFrame(matrix).T.sort_index()
        df.to_csv(os.path.join(feature_dir, f"{feature}.csv"))

    # Initialize the NLSImpactEstimator with the feature directory
    estimator = NLSImpactEstimator(feature_dir)

    # Build dataset using all available stocks
    stocks = list(estimator.features["total_volume"].index)
    x_all, y_all = estimator.build_dataset(stocks)

    # Fit the non-linear impact model to obtain eta and beta estimates
    eta, beta = estimator.fit_nls(x_all, y_all)
    print(f"Overall Estimates: eta = {eta}, beta = {beta}")

    # Bootstrap
    boot_pairs = estimator.bootstrap_estimates(x_all, y_all, n_iter=1000)
    eta_se_pairs = np.std(boot_pairs[:, 0])
    beta_se_pairs = np.std(boot_pairs[:, 1])
    t_eta_pairs = eta / eta_se_pairs if eta_se_pairs != 0 else float('nan')
    t_beta_pairs = beta / beta_se_pairs if beta_se_pairs != 0 else float('nan')

    # Residual Bootstrap
    boot_resid = estimator.residual_bootstrap_estimates(x_all, y_all, eta, beta, n_iter=1000)
    eta_se_resid = np.std(boot_resid[:, 0]) if len(boot_resid) > 0 else float('nan')
    beta_se_resid = np.std(boot_resid[:, 1]) if len(boot_resid) > 0 else float('nan')
    t_eta_resid = eta / eta_se_resid if eta_se_resid != 0 else float('nan')
    t_beta_resid = beta / beta_se_resid if beta_se_resid != 0 else float('nan')

    # Write parameter estimates and t-statistics (from pairs bootstrap) to params_part1.txt
    with open("params_part1.txt", "w") as f:
        f.write(f"eta = {eta}\n")
        f.write(f"t-eta = {t_eta_pairs}\n")
        f.write(f"beta = {beta}\n")
        f.write(f"t-beta = {t_beta_pairs}\n")
    print("Parameter estimates and t-values (pairs bootstrap) written to params_part1.txt")

    # Residual Analysis (Almgren et al.)
    y_hat = estimator.impact_model(x_all, eta, beta)
    residuals = y_all - y_hat

    # Histogram of residuals
    plt.figure()
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.title("Histogram of NLS Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.savefig("nls_residual_histogram.png")
    plt.close()

    # Q-Q plot of residuals
    sm.qqplot(residuals, line='s')
    plt.legend(["Residuals", "Theoretical Quantiles"])
    plt.title("Q-Q Plot of NLS Residuals")
    plt.savefig("nls_qq_plot.png")
    plt.close()

    # Log Q-Q plot of residuals
    log_residuals = np.log(np.abs(residuals[residuals != 0]))  # Avoid log(0) by filtering out zeros
    sm.qqplot(log_residuals, line='s')
    plt.legend(["Log Residuals", "Theoretical Quantiles"])
    plt.title("Log Q-Q Plot of NLS Residuals")
    plt.savefig("nls_log_qq_plot.png")
    plt.close()

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test statistic: {shapiro_stat}, p-value: {shapiro_p}")

    # Compare Parameters for High vs. Low Activity Stocks
    estimator.compare_stock_groups()

    # Extra Credit: White's Test for Heteroskedasticity
    estimator.test_heteroskedasticity(x_all, y_all)

if __name__ == "__main__":
    main()