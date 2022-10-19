import numpy as np
from myarray import MyArray

class Loss:
    class MSELoss:
        def __init__(self):
            pass

        def __call__(self, y, t):
            return (y-t)**2
    
    class CrossEntropyLoss:
        def __init__(self):
            pass

        def __call__(self, y, t):
            return -np.sum(MyArray.Log(y + 1e-7) * t, axis=1)