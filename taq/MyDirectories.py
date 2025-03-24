import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

class MyDirectories:
    @staticmethod
    def getQuotesDir():
        """Returns the path to the quotes directory."""
        return os.path.join(BASE_PATH, "../data/quotes/extracted")
    
    @staticmethod
    def getTradesDir():
        """Returns the path to the trades directory."""
        return os.path.join(BASE_PATH, "../data/trades/extracted")