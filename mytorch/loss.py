class MSELoss:
    def __init__(self):
        pass

    def __call__(self, y, t):
        return ((y-t)**2).sum()/y.a.shape[0]

class CrossEntropyLoss:
    def __init__(self, dim=1):
        self.dim = dim

    def __call__(self, y, t):
        return -(t*(y+1e-7).log()+(-t+1)*(-y+1+1e-7).log()).sum()/y.a.shape[0]/2