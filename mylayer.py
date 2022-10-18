from myarray import MyArray
import numpy as np

class MyLayer:
    class Linear:
        def __init__(self, in_features, out_features, bias=True):
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            if bias:
                self.weight = MyArray.from_array(np.random.normal(size=(in_features+1, out_features)))
            else:
                self.weight = MyArray.from_array(np.random.normal(size=(in_features, out_features)))
        
        def __call__(self, inputs):
            if self.bias:
                inputs = np.concatenate((inputs, MyArray.from_array(np.ones((inputs.shape[0], 1)))),axis=1)
            return np.dot(inputs, self.weight)
        
        def get_params(self):
            return self.weight

    class ReLu:
        def __init__(self):
            pass

        def __call__(self, inputs):
            return np.array([i.ReLu() for i in inputs.reshape(-1)]).reshape(inputs.shape)

    class LeakeyReLu:
        def __init__(self):
            pass

        def __call__(self, inputs):
            return np.array([i.LeakeyReLu() for i in inputs.reshape(-1)]).reshape(inputs.shape)

    class Sigmoid:
        def __init__(self):
            pass
        
        def __call__(self, inputs):
            return np.array([i.Sigmoid() for i in inputs.reshape(-1)]).reshape(inputs.shape)

if __name__ == "__main__":
    linear = MyLayer.Linear(3, 2)
    relu = MyLayer.ReLu()
    a = MyArray.from_array(np.ones((4, 3)))
    b = linear(a)
    b = relu(b)
    print(b)