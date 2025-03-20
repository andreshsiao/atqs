import unittest

from taq import MyDirectories
from taq.TAQTradesReader import TAQTradesReader

class Test_TAQTradesReader(unittest.TestCase):

    def test1(self):
        reader = TAQTradesReader(MyDirectories.get_trades_dir() + '/20070920/IBM_trades.binRT')
        
        zz = list([
            reader.get_n(),
            reader.get_secs_from_epoc_to_midn(),
            reader.get_millis_from_midn(0),
            reader.get_size(0),
            reader.get_price(0)
        ])

        self.assertEqual(
            '[25367, 1190260800, 34210000, 76600, 116.2699966430664]',
            str(zz)
        )


if __name__ == "__main__":
    unittest.main()
