import unittest
from taq.DataProcessor import DataProcessor

def mock_time_to_millis(time_str):
    h, m = map(int, time_str.split(":"))
    return (h * 60 + m) * 60 * 1000

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Set up a mock DataProcessor instance and sample daily data for testing
        self.processor = DataProcessor("mock_dir")
        self.daily_data = [
            {"timestamp": mock_time_to_millis("09:30"), "bid_price": 100, "ask_price": 102, "bid_size": 50, "ask_size": 40, "volume": 100},
            {"timestamp": mock_time_to_millis("09:32"), "bid_price": 101, "ask_price": 103, "bid_size": 60, "ask_size": 45, "volume": 150},
            {"timestamp": mock_time_to_millis("09:34"), "bid_price": 102, "ask_price": 104, "bid_size": 55, "ask_size": 50, "volume": 200},
            {"timestamp": mock_time_to_millis("09:36"), "bid_price": 103, "ask_price": 105, "bid_size": 52, "ask_size": 48, "volume": 250},
            {"timestamp": mock_time_to_millis("09:38"), "bid_price": 104, "ask_price": 106, "bid_size": 58, "ask_size": 53, "volume": 300}
        ]

    def test_add_midquote_to_data(self):
        # Test if the method correctly adds a "mid_quote" field to each entry in the data
        self.processor.add_midquote_to_data(self.daily_data)
        for entry in self.daily_data:
            self.assertIn("mid_quote", entry)

    def test_compute_midquote_returns(self):
        # Test if the method computes midquote returns as a list for a given interval
        self.processor.add_midquote_to_data(self.daily_data)
        returns = self.processor.compute_midquote_returns(self.daily_data, interval=2)
        self.assertIsInstance(returns, list)

    def test_compute_total_daily_volume(self):
        # Test if the method computes the total daily volume correctly by summing bid and ask sizes
        total_vol = self.processor.compute_total_daily_volume(self.daily_data)
        self.assertEqual(total_vol, sum(d["bid_size"] + d["ask_size"] for d in self.daily_data))

    def test_compute_arrival_price(self):
        # Test if the method computes the arrival price as a float
        self.processor.add_midquote_to_data(self.daily_data)
        price = self.processor.compute_arrival_price(self.daily_data)
        self.assertTrue(isinstance(price, float))

    def test_compute_imbalance(self):
        # Test if the method computes the order imbalance correctly as the difference between bid and ask sizes
        imbalance = self.processor.compute_imbalance(self.daily_data)
        expected = sum(entry["bid_size"] - entry["ask_size"] for entry in self.daily_data)
        self.assertEqual(imbalance, expected)

    def test_compute_vwap(self):
        # Test if the method computes the volume-weighted average price (VWAP) as a float for a given time range
        self.processor.add_midquote_to_data(self.daily_data)
        vwap = self.processor.compute_vwap(self.daily_data, "09:30", "09:38")
        self.assertTrue(isinstance(vwap, float))

    def test_compute_terminal_price(self):
        # Test if the method computes the terminal price as the "mid_quote" of the last entry in the data
        self.processor.add_midquote_to_data(self.daily_data)
        price = self.processor.compute_terminal_price(self.daily_data)
        self.assertEqual(price, self.daily_data[-1]["mid_quote"])

    def test_filter_time_range(self):
        # Test if the method filters the data correctly to include only entries within the specified time range
        filtered = self.processor.filter_time_range(self.daily_data, "09:30", "09:34")
        for entry in filtered:
            self.assertTrue(mock_time_to_millis("09:30") <= entry["timestamp"] <= mock_time_to_millis("09:34"))

if __name__ == "__main__":
    unittest.main()
