import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from taq.NLSEstimator import NLSImpactEstimator

class TestNLSImpactEstimator(unittest.TestCase):
    @patch("taq.NLSEstimator.pd.read_csv")
    def setUp(self, mock_read_csv):
        # Simulate feature DataFrames
        index = ['AAPL', 'GOOG']
        columns = ['2021-01-01', '2021-01-02']
        fake_data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=columns)
        mock_read_csv.return_value = fake_data

        self.estimator = NLSImpactEstimator("dummy_dir")

    def test_get_avg_volume_by_stock(self):
        avg_vol = self.estimator.get_avg_volume_by_stock()
        self.assertEqual(avg_vol.index.tolist(), ['GOOG', 'AAPL'])

    def test_split_stock_groups(self):
        high, low = self.estimator.split_stock_groups()
        self.assertEqual(high, ['GOOG'])
        self.assertEqual(low, ['AAPL'])

    def test_build_dataset(self):
        x, y = self.estimator.build_dataset(['AAPL'])
        self.assertTrue(isinstance(x, np.ndarray))
        self.assertTrue(isinstance(y, np.ndarray))

    def test_fit_nls(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 0.5 * x ** 0.6
        params = self.estimator.fit_nls(x, y)
        self.assertEqual(len(params), 2)

    def test_bootstrap_estimates(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 0.5 * x ** 0.6
        boot_params = self.estimator.bootstrap_estimates(x, y, n_iter=10)
        self.assertEqual(boot_params.shape[1], 2)

    def test_residual_bootstrap_estimates(self):
        x = np.array([1, 2, 3, 4, 5])
        y = 0.5 * x ** 0.6
        eta, beta = self.estimator.fit_nls(x, y)
        res_boot = self.estimator.residual_bootstrap_estimates(x, y, eta, beta, n_iter=10)
        self.assertEqual(res_boot.shape[1], 2)

if __name__ == "__main__":
    unittest.main()
