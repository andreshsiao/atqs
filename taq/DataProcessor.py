import os
from taq.Utils import time_to_millis 
from taq.MyDirectories import BASE_PATH

class DataProcessor:
    def __init__(self, extract_dir):
        # Initialize the DataProcessor with the directory containing extracted data
        self.extract_dir = extract_dir

    def compute_midquote_returns(self, daily_data, interval=120):
        """
        Computes mid-quote returns at a given interval (default: 2 minutes).
        Mid-quote is the average of bid and ask prices.
        Returns are calculated as the percentage change in mid-quote over the interval.
        """
        if len(daily_data) < interval:
            return []  # Not enough data points to compute returns

        # Calculate mid-quotes for each entry in the data
        mid_quotes = [(entry["bid_price"] + entry["ask_price"]) / 2 for entry in daily_data]
        
        midquote_returns = []
        # Compute returns at the specified interval
        for i in range(interval, len(mid_quotes), interval):
            prev = mid_quotes[i - interval]
            curr = mid_quotes[i]
            midquote_returns.append((curr - prev) / prev)

        return midquote_returns

    def compute_total_daily_volume(self, daily_data):
        """
        Computes the total daily traded volume.
        Volume is the sum of bid size and ask size for all entries in the data.
        """
        return sum(entry["bid_size"] + entry["ask_size"] for entry in daily_data)

    def compute_arrival_price(self, daily_data):
        """
        Computes the arrival price, which is the average of the first five mid-quotes.
        If there are fewer than five entries, it averages all available mid-quotes.
        """
        mid_quotes = [entry["mid_quote"] for entry in daily_data]
        return sum(mid_quotes[:5]) / len(mid_quotes[:5]) \
            if len(mid_quotes) >= 5 else sum(mid_quotes) / len(mid_quotes)
    
    def add_midquote_to_data(self, daily_data):
        """
        Adds a mid-quote field to each entry in the daily data.
        Mid-quote is calculated as the average of bid and ask prices.
        """
        for entry in daily_data:
            entry["mid_quote"] = (entry["bid_price"] + entry["ask_price"]) / 2
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