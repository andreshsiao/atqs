import unittest
import pandas as pd
from taq.StaticSimulator import StaticSimulator

class TestStaticSimulator(unittest.TestCase):
    def setUp(self):
        # Create a simple static profile with 13 buckets summing to 1
        self.profile = pd.Series([1/13] * 13)
        self.total_shares = 1300
        self.simulator = StaticSimulator(volume_profile=self.profile, total_shares=self.total_shares)

    def test_simulation_output_length(self):
        result = self.simulator.simulate()
        self.assertEqual(len(result), 13)

    def test_simulation_total_shares(self):
        result = self.simulator.simulate()
        self.assertEqual(result.sum(), self.total_shares)

    def test_simulation_integer_output(self):
        result = self.simulator.simulate()
        self.assertTrue(all(isinstance(v, int) for v in result))

if __name__ == '__main__':
    unittest.main()
