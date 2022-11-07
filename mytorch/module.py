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
        params = []
        for layer in self.__dict__.values():
            param = layer.get_params()
            if type(param) == type(None):
                continue
            for p in param.values():
                if type(p) == type(None):
                    continue
                if type(p) == MyTensor:
                    params.append(p)
                    continue
        return params
    
    def train(self):
        for val in self.__dict__.values():
            val.train()

    def eval(self):
        for val in self.__dict__.values():
            val.eval()