import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm

class NLSImpactEstimator:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        self.features = {}
        self._load_features()

    def _load_features(self):
        feature_names = ["2min_returns", "total_volume", "arrival_price", "imbalance", "terminal_price"]
        for name in feature_names:
            path = os.path.join(self.feature_dir, f"{name}.csv")
            self.features[name] = pd.read_csv(path, index_col=0)

    def get_avg_volume_by_stock(self):
        df = self.features["total_volume"]
        return df.mean(axis=1).sort_values(ascending=False)

    def split_stock_groups(self):
        avg_vol = self.get_avg_volume_by_stock()
        midpoint = len(avg_vol) // 2
        high_activity = avg_vol.index[:midpoint].tolist()
        low_activity = avg_vol.index[midpoint:].tolist()
        return high_activity, low_activity

    def build_dataset(self, stocks):
        x_vals, y_vals = [], []
        for stock in stocks:
            try:
                imbalances = self.features["imbalance"].loc[stock]
                arrivals = self.features["arrival_price"].loc[stock]
                terminals = self.features["terminal_price"].loc[stock]
            except KeyError:
                continue
            for date in imbalances.index:
                try:
                    qv = abs(imbalances[date])
                    impact = abs(terminals[date] - arrivals[date])
                    if np.isnan(qv) or np.isnan(impact):
                        continue
                    x_vals.append(qv)
                    y_vals.append(impact)
                except:
                    continue
        return np.array(x_vals), np.array(y_vals)

    def impact_model(self, x, eta, beta):
        return eta * x**beta

    def fit_nls(self, x, y):
        return curve_fit(self.impact_model, x, y, p0=(0.01, 0.5))[0]

    def bootstrap_estimates(self, x, y, n_iter=1000):
        estimates = []
        n = len(x)
        for _ in range(n_iter):
            idx = np.random.choice(n, n, replace=True)
            x_sample, y_sample = x[idx], y[idx]
            try:
                params = self.fit_nls(x_sample, y_sample)
                estimates.append(params)
            except:
                continue
        return np.array(estimates)
    
    def residual_bootstrap_estimates(self, x, y, eta, beta, n_iter=1000):
        y_hat = self.impact_model(x, eta, beta)
        residuals = y - y_hat
        estimates = []
        n = len(y)
        for _ in range(n_iter):
            sampled_resid = np.random.choice(residuals, size=n, replace=True)
            y_boot = y_hat + sampled_resid
            try:
                params_boot = self.fit_nls(x, y_boot)
                estimates.append(params_boot)
            except Exception:
                continue
        return np.array(estimates)

    def compare_stock_groups(self):
        high, low = self.split_stock_groups()
        x_high, y_high = self.build_dataset(high)
        x_low, y_low = self.build_dataset(low)
        eta_high, beta_high = self.fit_nls(x_high, y_high)
        eta_low, beta_low = self.fit_nls(x_low, y_low)
        print("High Activity Stocks: eta =", eta_high, ", beta =", beta_high)
        print("Low Activity Stocks: eta =", eta_low, ", beta =", beta_low)

    def test_heteroskedasticity(self, x, y):
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
    feature_dir = "../feature_matrices"
    estimator = NLSImpactEstimator(feature_dir)
    estimator.compare_stock_groups()
    x_all, y_all = estimator.build_dataset(estimator.features["total_volume"].index.tolist())
    estimator.test_heteroskedasticity(x_all, y_all)