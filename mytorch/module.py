import numpy as np
from mytorch.tensor import MyTensor

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

    def get_params(self):
        params = [[value.get_params()["weight"], value.get_params()["bias"]] for value in self.__dict__.values() if type(value.get_params()) != type(None)]
        return sum(params, [])