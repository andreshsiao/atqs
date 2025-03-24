import unittest
from taq.DataProcessor import DataProcessor
from datetime import datetime

def mock_time_to_millis(time_str):
    h, m = map(int, time_str.split(":"))
    return (h * 60 + m) * 60 * 1000

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor("mock_dir")
        self.daily_data = [
            {"timestamp": mock_time_to_millis("09:30"), "bid_price": 100, "ask_price": 102, "bid_size": 50, "ask_size": 40, "volume": 100},
            {"timestamp": mock_time_to_millis("09:32"), "bid_price": 101, "ask_price": 103, "bid_size": 60, "ask_size": 45, "volume": 150},
            {"timestamp": mock_time_to_millis("09:34"), "bid_price": 102, "ask_price": 104, "bid_size": 55, "ask_size": 50, "volume": 200},
            {"timestamp": mock_time_to_millis("09:36"), "bid_price": 103, "ask_price": 105, "bid_size": 52, "ask_size": 48, "volume": 250},
            {"timestamp": mock_time_to_millis("09:38"), "bid_price": 104, "ask_price": 106, "bid_size": 58, "ask_size": 53, "volume": 300}
        ]

    def test_add_midquote_to_data(self):
        self.processor.add_midquote_to_data(self.daily_data)
        for entry in self.daily_data:
            self.assertIn("mid_quote", entry)

    def test_compute_midquote_returns(self):
        self.processor.add_midquote_to_data(self.daily_data)
        returns = self.processor.compute_midquote_returns(self.daily_data, interval=2)
        self.assertIsInstance(returns, list)

    def test_compute_total_daily_volume(self):
        total_vol = self.processor.compute_total_daily_volume(self.daily_data)
        self.assertEqual(total_vol, sum(d["bid_size"] + d["ask_size"] for d in self.daily_data))

    def test_compute_arrival_price(self):
        self.processor.add_midquote_to_data(self.daily_data)
        price = self.processor.compute_arrival_price(self.daily_data)
        self.assertTrue(isinstance(price, float))

    def test_compute_imbalance(self):
        imbalance = self.processor.compute_imbalance(self.daily_data)
        expected = sum(entry["bid_size"] - entry["ask_size"] for entry in self.daily_data)
        self.assertEqual(imbalance, expected)

    def test_compute_vwap(self):
        self.processor.add_midquote_to_data(self.daily_data)
        vwap = self.processor.compute_vwap(self.daily_data, "09:30", "09:38")
        self.assertTrue(isinstance(vwap, float))

    def test_compute_terminal_price(self):
        self.processor.add_midquote_to_data(self.daily_data)
        price = self.processor.compute_terminal_price(self.daily_data)
        self.assertEqual(price, self.daily_data[-1]["mid_quote"])

    def test_filter_time_range(self):
        filtered = self.processor.filter_time_range(self.daily_data, "09:30", "09:34")
        for entry in filtered:
            self.assertTrue(mock_time_to_millis("09:30") <= entry["timestamp"] <= mock_time_to_millis("09:34"))

if __name__ == "__main__":
    unittest.main()
