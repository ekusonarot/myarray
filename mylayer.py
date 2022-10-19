from myarray import MyArray
import numpy as np


class MyLayer:
    class Layer:
        def __init__(self):
            self.weight = None
            pass

        def __call__(self):
            pass

        def get_params(self):
            pass

    class Linear(Layer):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bias = bias
            if bias:
                self.weight = MyArray.from_array(np.random.normal(size=(in_features+1, out_features), loc=0, scale=1))
            else:
                self.weight = MyArray.from_array(np.random.normal(size=(in_features, out_features), loc=0, scale=1))
        
        def __call__(self, inputs):
            if self.bias:
                inputs = np.concatenate((inputs, MyArray.from_array(np.ones((inputs.shape[0], 1)))),axis=1)
            return np.dot(inputs, self.weight)
        
        def get_params(self):
            return self.weight

    class ReLu(Layer):
        def __init__(self):
            super().__init__()

        def __call__(self, inputs):
            return np.array([i.ReLu() for i in inputs.reshape(-1)]).reshape(inputs.shape)

    class LeakeyReLu(Layer):
        def __init__(self):
            super().__init__()

        def __call__(self, inputs):
            return np.array([i.LeakeyReLu() for i in inputs.reshape(-1)]).reshape(inputs.shape)

    class Sigmoid(Layer):
        def __init__(self):
            super().__init__()
        
        def __call__(self, inputs):
            return np.array([i.Sigmoid() for i in inputs.reshape(-1)]).reshape(inputs.shape)

    class Softmax(Layer):
        def __init__(self, dim=1):
            super().__init__()
            self.dim = dim
        
        def __call__(self, inputs):
            max = np.max(inputs, axis=self.dim, keepdims=True).astype(np.float)
            inputs = MyArray.Exp(inputs - max)
            sum = np.sum(inputs, axis=self.dim, keepdims=True)
            inputs = inputs/sum
            return inputs

if __name__ == "__main__":
    linear = MyLayer.Linear(3, 2)
    relu = MyLayer.ReLu()
    a = MyArray.from_array(np.ones((4, 3)))
    b = linear(a)
    b = relu(b)
    print(b)