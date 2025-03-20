import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class MyDirectories:
    @staticmethod
    def get_quotes_dir():
        """Returns the path to the quotes directory."""
        return os.path.join(BASE_PATH, "../data/quotes")
    
    @staticmethod
    def get_trades_dir():
        """Returns the path to the trades directory."""
        return os.path.join(BASE_PATH, "../data/trades")