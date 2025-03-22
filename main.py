import os
from collections import defaultdict
import pandas as pd
from taq.MyDirectories import MyDirectories, BASE_PATH
from taq.DataProcessor import DataProcessor
from taq.TAQQuotesReader import TAQQuotesReader
from taq.Utils import extract_tar_files, extract_all_quotes, get_stock_list  # Importing from Utils

def main():
    # Define input directories
    tar_dir = MyDirectories.get_quotes_dir()
    extract_dir = os.path.join(tar_dir, "extracted")

    extract_tar_files(tar_dir, extract_dir)

    # Initialize data processor
    processor = DataProcessor(extract_dir)

    feature_matrices = {
        "2min_returns": defaultdict(dict),
        "total_volume": defaultdict(dict),
        "arrival_price": defaultdict(dict),
        "imbalance": defaultdict(dict),
        "terminal_price": defaultdict(dict),
    }

    # Process each extracted date folder
    for date_folder in sorted(os.listdir(extract_dir)):
        date_path = os.path.join(extract_dir, date_folder)
        if not os.path.isdir(date_path):
            continue  # Skip if not a folder

        # stock_list = get_stock_list(date_path)
        stock_list = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'JNJ', 'JPM']

        for stock in stock_list:
            print(f"Processing stock {date_folder}: {stock}")

            stock_file_path = os.path.join(date_path, f"{stock}_quotes.binRQ")
            reader = TAQQuotesReader(stock_file_path)  # Read binary data
            daily_data = extract_all_quotes(reader)

            processor.add_midquote_to_data(daily_data)

            # Compute required metrics
            two_minute_returns = processor.compute_midquote_returns(daily_data)
            total_volume = processor.compute_total_daily_volume(daily_data)
            arrival_price = processor.compute_arrival_price(daily_data)
            imbalance = processor.compute_imbalance(daily_data)
            terminal_price = processor.compute_terminal_price(daily_data)

            # Populate feature matrices
            feature_matrices["2min_returns"][stock][date_folder] = two_minute_returns
            feature_matrices["total_volume"][stock][date_folder] = total_volume
            feature_matrices["arrival_price"][stock][date_folder] = arrival_price
            feature_matrices["imbalance"][stock][date_folder] = imbalance
            feature_matrices["terminal_price"][stock][date_folder] = terminal_price

    # Save feature matrices to CSV
    output_dir = os.path.join(BASE_PATH, "../feature_matrices")
    os.makedirs(output_dir, exist_ok=True)

    for feature, matrix in feature_matrices.items():
        df = pd.DataFrame(matrix).T.sort_index()
        df.to_csv(os.path.join(output_dir, f"{feature}.csv"))

if __name__ == "__main__":
    main()