import os
import tarfile

def extract_tar_files(tar_dir, extract_dir):
    """Extracts all tar files in the given directory if not already extracted."""
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    for tarfile_name in os.listdir(tar_dir):
        if tarfile_name.endswith(".tar") or tarfile_name.endswith(".tar.gz"):
            tar_path = os.path.join(tar_dir, tarfile_name)

            # Avoid re-extracting
            if not os.path.exists(os.path.join(extract_dir, tarfile_name.split('.')[0])):  
                print(f"Extracting {tarfile_name}...")
                with tarfile.open(tar_path, "r") as tar:
                    tar.extractall(extract_dir)

def get_stock_list(date_path):
    """Returns a list of available stock files for a given date directory."""
    return [f.split("quotes")[0][:-1] for f in os.listdir(date_path)]

def extract_all_quotes(reader):
    """
    Extracts all quotes from TAQQuotesReader into a structured list.
    """
    n = reader.getN()  # Number of records
    extracted_data = []

    for i in range(n):
        entry = {
            "timestamp": reader.getMillisFromMidn(i),
            "bid_size": reader.getBidSize(i),
            "bid_price": reader.getBidPrice(i),
            "ask_size": reader.getAskSize(i),
            "ask_price": reader.getAskPrice(i),
        }
        extracted_data.append(entry)

    return extracted_data

def time_to_millis(time_str):
        h, m = map(int, time_str.split(":"))
        return (h * 60 + m) * 60 * 1000