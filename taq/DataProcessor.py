import os
from taq.Utils import time_to_millis
from taq.TAQQuotesReader import TAQQuotesReader
from taq.MyDirectories import BASE_PATH

class DataProcessor:
    def __init__(self, extract_dir):
        self.extract_dir = extract_dir

    def compute_midquote_returns(self, daily_data, interval=120):
        """Computes mid-quote returns at a given interval (default: 2 minutes)."""
        if len(daily_data) < interval:
            return []  # Not enough data points

        mid_quotes = [(entry["bid_price"] + entry["ask_price"]) / 2 for entry in daily_data]
        
        midquote_returns = []
        for i in range(interval, len(mid_quotes), interval):
            prev = mid_quotes[i - interval]
            curr = mid_quotes[i]
            midquote_returns.append((curr - prev) / prev)

        return midquote_returns

    def compute_total_daily_volume(self, daily_data):
        """Computes total daily traded volume."""
        return sum(entry["bid_size"] + entry["ask_size"] for entry in daily_data)

    def compute_arrival_price(self, daily_data):
        """Computes arrival price (average of first five mid-quotes)."""
        mid_quotes = [entry["mid_quote"] for entry in daily_data]
        return sum(mid_quotes[:5]) / len(mid_quotes[:5]) \
            if len(mid_quotes) >= 5 else sum(mid_quotes) / len(mid_quotes)
    
    def add_midquote_to_data(self, daily_data):
        """Adds mid-quote (average of bid and ask prices) to each entry in daily_data."""
        for entry in daily_data:
            entry["mid_quote"] = (entry["bid_price"] + entry["ask_price"]) / 2

    def compute_imbalance(self, daily_data):
        """Computes order imbalance between 9:30 and 3:30."""
        time_filtered_data = self.filter_time_range(daily_data, "09:30", "15:30")
        return sum(entry["bid_size"] - entry["ask_size"] for entry in time_filtered_data)

    # TODO: Implement compute_vwap method for trade data
    def compute_vwap(self, daily_data, start, end):
        """Computes VWAP for a given time range."""
        filtered_data = self.filter_time_range(daily_data, start, end)
        total_value = sum([p * v for p, v in zip(filtered_data["mid_quote"], filtered_data["volume"])])
        total_volume = sum(filtered_data["volume"])
        return total_value / total_volume if total_volume > 0 else 0

    def compute_terminal_price(self, daily_data):
        """Computes terminal price (average of last five mid-quotes at 4:00)."""
        return daily_data[-1]['mid_quote'] if daily_data else 0

    def filter_time_range(self, daily_data, start, end):
        """Filters data for a specific time range."""
        # Convert start and end time strings ("HH:MM") to timestamps in milliseconds from midnight
        start_millis = time_to_millis(start)
        end_millis = time_to_millis(end)

        return [entry for entry in daily_data if start_millis <= entry["timestamp"] <= end_millis]

    def save_results(self, stock, date, results):
        """Saves computed results to an output file."""
        output_dir = os.path.join(BASE_PATH, f"../output/{date}")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock}_results.txt")

        with open(output_file, "a") as f:
            f.write(f"{date}: {results}\n")