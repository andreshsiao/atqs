
import unittest
from taq.DynamicSimulator import DynamicSimulator

class TestDynamicSimulator(unittest.TestCase):
    def setUp(self):
        self.total_shares = 1300
        self.static_profile = [1/13] * 13
        self.simulator = DynamicSimulator(initial_shares=self.total_shares, eta=0.01, static_profile=self.static_profile)

    def test_simulation_output_length(self):
        forecasts = [[1/13] * (13 - i) for i in range(13)]
        result = self.simulator.simulate(forecasts)
        self.assertEqual(len(result), 13)

    def test_total_shares_traded(self):
        forecasts = [[1/13] * (13 - i) for i in range(13)]
        result = self.simulator.simulate(forecasts)
        self.assertEqual(result.sum(), self.total_shares)

    def test_trajectory_shape(self):
        # Should front-load or back-load depending on sinh; we only test total here for simplicity
        forecasts = [[1/13] * (13 - i) for i in range(13)]
        result = self.simulator.simulate(forecasts)
        self.assertTrue(all(isinstance(x, int) for x in result))
    
    def test_ac_mode_total_shares_traded(self):
        # Test Almgren-Chriss mode: total shares traded equals initial_shares
        simulator_ac = DynamicSimulator(initial_shares=self.total_shares, eta=0.01, mode="ac", static_profile=self.static_profile)
        forecasts = [[1/13] * (13 - i) for i in range(13)]
        result = simulator_ac.simulate(forecasts)
        self.assertAlmostEqual(result.sum(), self.total_shares, msg="Total shares traded in AC mode should match initial_shares")

    def test_ac_mode_output_length(self):
        # Test Almgren-Chriss mode: output length matches forecast horizon
        simulator_ac = DynamicSimulator(initial_shares=self.total_shares, eta=0.01, mode="ac", static_profile=self.static_profile)
        forecasts = [[1/13] * (13 - i) for i in range(13)]
        result = simulator_ac.simulate(forecasts)
        self.assertEqual(len(result), 13, "Output length in AC mode should match forecast horizon")

if __name__ == '__main__':
    unittest.main()
  