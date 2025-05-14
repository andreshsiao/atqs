import unittest
import pandas as pd
import numpy as np
from taq.volume_models.VolumeModel import VolumeModel

class TestVolumeModel(unittest.TestCase):
    def setUp(self):
        # Create mock volume data for two symbols over three dates
        self.mock_data = {
            "AAPL": pd.DataFrame([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.06, 0.06, 0.06, 0.06, 0.06],
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04]
            ], columns=list(range(13)), index=pd.date_range("2023-01-01", periods=3)),
            "MSFT": pd.DataFrame([
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.04, 0.04, 0.04],
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05],
                [0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.06, 0.06, 0.06, 0.06, 0.06]
            ], columns=list(range(13)), index=pd.date_range("2023-01-01", periods=3))
        }

        self.model = VolumeModel(feature_dir=".")
        self.model.volume_data = self.mock_data

    def test_static_profile(self):
        profile = self.model.get_static_profile()
        self.assertEqual(len(profile), 13)
        self.assertAlmostEqual(profile.sum(), 1.0, places=4)

    def test_fit_and_predict(self):
        # combine mock data into one DataFrame
        combined = pd.concat(self.mock_data.values())
        self.model.fit_dynamic_model(combined)

        observed = [0.1, 0.1, 0.1]
        prediction = self.model.predict_remaining_volume(observed, current_bucket=3)
        self.assertTrue(isinstance(prediction, list))
        self.assertAlmostEqual(sum(prediction), 1.0, delta=1e-2)

if __name__ == "__main__":
    unittest.main()
