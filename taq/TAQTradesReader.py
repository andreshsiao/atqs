import gzip
import struct

class TAQTradesReader(object):
    """
    This reader reads an entire compressed binary TAQ trades file into memory,
    uncompresses it, and gives its clients access to the contents of the file
    via a set of get methods.
    """

    def __init__(self, filePathName):
        """
        Do all of the heavy lifting here and give users getters for the results.
        """
        self.filePathName = filePathName
        with gzip.open(filePathName, "rb") as f:
            file_content = f.read()
            self._header = struct.unpack_from(">2i", file_content[0:8])
            
            # millis from midnight
            endI = 8 + (4 * self._header[1])
            self._ts = struct.unpack_from((">%di" % self._header[1]), file_content[8:endI])
            startI = endI

            
            endI = endI + (4 * self._header[1])
            self._s = struct.unpack_from((">%di" % self._header[1]), file_content[startI:endI])
            startI = endI

            
            endI = endI + (4 * self._header[1])
            self._p = struct.unpack_from((">%df" % self._header[1]), file_content[startI:endI])

    def get_n(self):
        return self._header[1]

    def get_secs_from_epoc_to_midn(self):
        return self._header[0]

    def get_price(self, index):
        return self._p[index]

    def get_millis_from_midn(self, index):
        return self._ts[index]

    def get_timestamp(self, index):
        return self.get_millis_from_midn(index)  # Compatibility (more intuitive)

    def get_size(self, index):
        return self._s[index]

    def rewrite(self, filePathName, tickerId):
        s = struct.Struct(">QHIf")
        out = gzip.open(filePathName, "wb")
        baseTS = self.get_secs_from_epoc_to_midn() * 1000
        for i in range(self.get_n()):
            ts = baseTS + self.getMillisFromMidn(i)
            out.write(s.pack(ts, tickerId, self.get_size(i), self.get_price(i)))
        out.close()
