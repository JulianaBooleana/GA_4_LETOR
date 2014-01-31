__author__ = 'julianagomes'


class Feature:


    Max = -1.0
    Min = 1.0

    def __init__(self,ma,mi,b,e = 0.1):
        self.Max = ma
        self.Min = mi
        self.e = e
        self.b = b


    def update_min_max(self,val):
        """
            if the value of a feature is higher than max val found --> update
            if the value of a feature is lower than a max val found --> update
        """
        if val > self.Max:
            self.Max = val
        if val < self.Min:
            self.Min = val


    def set_e(self,e):
        """
            sets the constant e
            used in the Bucket formula
        """
        self.e = e


    def set_b(self,b):
        """
            sets the constant b
            used in the Bucket formula
        """
        self.b = b


    def get_max(self):
        """
            returns the Max val that a feature takes
        """
        return self.Max


    def get_min(self):
        """
            returns the Min val that a feature takes
        """
        return self.Min


    def get_bucket(self,num):
        """
            Uniform sized buckets
            This formula was from page 203 of book:
            Managing Gigabytes: Compressing and Indexing Documents and Images
                                by Ian H. Witten, Alistair Moffat, Timothy C. Bell
        """
        c = (float(num - self.Min)/float((self.Max - self.Min) + self.e))*(2**self.b)
        c = round(c)
        return c