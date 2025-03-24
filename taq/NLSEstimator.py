import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm

# Class to estimate the impact of imbalances on stock prices using Nonlinear Least Squares (NLS)
class NLSImpactEstimator:
    def __init__(self, feature_dir):
        # Initialize the estimator with the directory containing feature data
        self.feature_dir = feature_dir
        self.features = {}
        self._load_features()

    def _load_features(self):
        # Load feature data from CSV files into a dictionary
        feature_names = ["2min_returns", "total_volume", "arrival_price", "imbalance", "terminal_price"]
        for name in feature_names:
            path = os.path.join(self.feature_dir, f"{name}.csv")
            self.features[name] = pd.read_csv(path, index_col=0)

    def get_avg_volume_by_stock(self):
        # Compute the average trading volume for each stock and sort in descending order
        df = self.features["total_volume"]
        return df.mean(axis=1).sort_values(ascending=False)

    def split_stock_groups(self):
        # Split stocks into two groups: high activity and low activity based on average volume
        avg_vol = self.get_avg_volume_by_stock()
        midpoint = len(avg_vol) // 2
        high_activity = avg_vol.index[:midpoint].tolist()
        low_activity = avg_vol.index[midpoint:].tolist()
        return high_activity, low_activity

    def build_dataset(self, stocks):
        # Build the dataset of imbalance (x) and price impact (y) for the given stocks
        x_vals, y_vals = [], []
        for stock in stocks:
            try:
                # Retrieve imbalance, arrival price, and terminal price for the stock
                imbalances = self.features["imbalance"].loc[stock]
                arrivals = self.features["arrival_price"].loc[stock]
                terminals = self.features["terminal_price"].loc[stock]
            except KeyError:
                # Skip stocks with missing data
                continue
            for date in imbalances.index:
                try:
                    # Compute absolute imbalance (qv) and absolute price impact
                    qv = abs(imbalances[date])
                    impact = abs(terminals[date] - arrivals[date])
                    if np.isnan(qv) or np.isnan(impact):
                        continue
                    x_vals.append(qv)
                    y_vals.append(impact)
                except:
                    # Skip any errors during data processing
                    continue
        return np.array(x_vals), np.array(y_vals)

    def impact_model(self, x, eta, beta):
        # Define the nonlinear impact model: impact = eta * imbalance^beta
        return eta * x**beta

    def fit_nls(self, x, y):
        # Fit the nonlinear impact model to the data using curve fitting
        return curve_fit(self.impact_model, x, y, p0=(0.01, 0.5))[0]

    def bootstrap_estimates(self, x, y, n_iter=1000):
        # Perform bootstrap resampling to estimate model parameters
        estimates = []
        n = len(x)
        for _ in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            x_sample, y_sample = x[idx], y[idx]
            try:
                params = self.fit_nls(x_sample, y_sample)
                estimates.append(params)
            except:
                # Skip iterations with fitting errors
                continue
        return np.array(estimates)
    
    def residual_bootstrap_estimates(self, x, y, eta, beta, n_iter=1000):
        # Perform residual bootstrap to estimate model parameters
        y_hat = self.impact_model(x, eta, beta)
        residuals = y - y_hat
        estimates = []
        n = len(y)
        for _ in range(n_iter):
            # Resample residuals and generate new y values
            sampled_resid = np.random.choice(residuals, size=n, replace=True)
            y_boot = y_hat + sampled_resid
            try:
                params_boot = self.fit_nls(x, y_boot)
                estimates.append(params_boot)
            except Exception:
                # Skip iterations with fitting errors
                continue
        return np.array(estimates)

    def compare_stock_groups(self):
        # Compare the impact model parameters for high and low activity stock groups
        high, low = self.split_stock_groups()
        x_high, y_high = self.build_dataset(high)
        x_low, y_low = self.build_dataset(low)
        eta_high, beta_high = self.fit_nls(x_high, y_high)
        eta_low, beta_low = self.fit_nls(x_low, y_low)
        print("High Activity Stocks: eta =", eta_high, ", beta =", beta_high)
        print("Low Activity Stocks: eta =", eta_low, ", beta =", beta_low)

    def test_heteroskedasticity(self, x, y):
        # Test for heteroskedasticity in the residuals using White's test
        eta, beta = self.fit_nls(x, y)
        y_hat = self.impact_model(x, eta, beta)
        residuals = y - y_hat

        # White's test requires a linear regression model, so we use fitted values as regressor
        exog = sm.add_constant(y_hat)
        white_test = het_white(residuals, exog)
        labels = ['Test Statistic', 'Test p-value', 'F-Statistic', 'F p-value']
        print("\nWhite's Test for Heteroskedasticity:")
        for label, val in zip(labels, white_test):
            print(f"{label}: {val:.4f}")

if __name__ == "__main__":
    # Main script to initialize the estimator and perform analysis
    feature_dir = "../feature_matrices"  # Directory containing feature data
    estimator = NLSImpactEstimator(feature_dir)
    estimator.compare_stock_groups()  # Compare high and low activity stock groups
    x_all, y_all = estimator.build_dataset(estimator.features["total_volume"].index.tolist())
    estimator.test_heteroskedasticity(x_all, y_all)  # Test for heteroskedasticity