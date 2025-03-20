import unittest

from taq import MyDirectories
from taq.TAQQuotesReader import TAQQuotesReader

class Test_TAQQuotesReader(unittest.TestCase):

    def test1(self):
        reader = TAQQuotesReader(MyDirectories.getQuotesDir() + '/20070920/IBM_quotes.binRQ')
        
        zz = list([
            reader.get_n(),
            reader.get_secs_from_epoc_to_midn(),
            reader.get_millis_from_midn(0),
            reader.get_ask_size(0),
            reader.get_ask_price(0),
            reader.get_bid_size(0),
            reader.get_bid_price(0)
        ])
        self.assertEqual(
            '[70166, 1190260800, 34210000, 1, 116.19999694824219, 38, 116.19999694824219]', 
            str(zz) 
       )


if __name__ == "__main__":
    unittest.main()
