import numpy as np
from myarray import MyArray

class Module:
    def __init__(self):
        pass
    
    def __call__(self):
        pass
    
    def state_dict(self):
        return(self.__dict__)

    def load_dict(self, dict):
        for skey in self.__dict__.keys():
            for lkey, lval in dict.items():
                if skey == lkey:
                    self.__dict__[skey] = lval
    
    def eval(self):
        for val in self.__dict__.values():
            if type(val.weight) != type(None):
                val.weight = val.get_params().astype(np.float)

    def train(self):
        for val in self.__dict__.values():
            if type(val.weight) != type(None):
                val.weight = MyArray.from_array(val.get_params())

    def get_params(self):
        return [value.get_params() for value in self.__dict__.values()]