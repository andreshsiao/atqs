import os
from taq.MyDirectories import MyDirectories
from taq.DataProcessor import DataProcessor
from taq.TAQQuotesReader import TAQQuotesReader
from taq.Utils import extract_tar_files, extract_all_quotes, get_stock_list  # Importing from Utils

def main():
    # Define input directories
    tar_dir = MyDirectories.getQuotesDir()
    extract_dir = os.path.join(tar_dir, "extracted")

    extract_tar_files(tar_dir, extract_dir)

    # Initialize data processor
    processor = DataProcessor(extract_dir)

    # Process each extracted date folder
    for date_folder in sorted(os.listdir(extract_dir)):
        date_path = os.path.join(extract_dir, date_folder)
        if not os.path.isdir(date_path):
            continue  # Skip if not a folder

        stock_list = get_stock_list(date_path)

        for stock in stock_list:
            print(f"Processing stock {date_folder}: {stock}")

            stock_file_path = os.path.join(date_path, f"{stock}_quotes.binRQ")
            reader = TAQQuotesReader(stock_file_path)  # Read binary data
            daily_data = extract_all_quotes(reader)

            processor.add_midquote_to_data(daily_data)

            # Compute required metrics
            two_minute_returns = processor.compute_2min_midquote_returns(daily_data)
            total_volume = processor.compute_total_daily_volume(daily_data)
            arrival_price = processor.compute_arrival_price(daily_data)
            imbalance = processor.compute_imbalance(daily_data)
            # vwap_930_330 = processor.compute_vwap(daily_data, "09:30", "15:30")
            # vwap_930_400 = processor.compute_vwap(daily_data, "09:30", "16:00")
            terminal_price = processor.compute_terminal_price(daily_data)

            # Store or print computed values
            results = {
                "2min_returns": two_minute_returns,
                "total_volume": total_volume,
                "arrival_price": arrival_price,
                "imbalance": imbalance,
                # "VWAP_930_330": vwap_930_330,
                # "VWAP_930_400": vwap_930_400,
                "terminal_price": terminal_price,
            }
            processor.save_results(stock, date_folder, results)

if __name__ == "__main__":
    main()