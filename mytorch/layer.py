from mytorch.tensor import MyTensor as mt
import numpy as np


class Layer:
    def __init__(self):
        self.weight = None
        self.bias = None
        self.training = True

    def __call__(self):
        pass

    def get_params(self):
        pass

    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        
class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = mt(np.random.normal(size=(in_features, out_features), loc=0, scale=1))
        self.bias_flag = bias
        self.bias = bias
        if bias:
            self.bias = mt(np.zeros(out_features))
        else:
            self.bias = 0.
    
    def __call__(self, inputs):
        return mt.dot(inputs, self.weight) + self.bias
    
    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias
        if bias:
            self.bias = mt(np.random.normal(size=(1,out_channels,1,1), loc=0, scale=1))
        self.weight = mt(
            np.random.normal(size=(out_channels, in_channels*kernel_size*kernel_size),loc=0,scale=1)
        ).T()

    def __call__(self, inputs):
        input_col, out_h, out_w = inputs.im2col(self.kernel_size, self.kernel_size, self.stride, self.padding)
        return mt.dot(input_col, self.weight).reshape(inputs.a.shape[0], out_h, out_w, -1).transpose(0, 3, 1, 2) + self.bias

    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def __call__(self, inputs):
        input_col, out_h, out_w = inputs.im2col(self.kernel_size, self.kernel_size, self.stride, self.padding)
        input_col = input_col.reshape(-1,self.kernel_size**2)
        input_col = mt.max(input_col, axis=1)
        return input_col.reshape(inputs.a.shape[0], out_h, out_w, -1).transpose(0, 3, 1, 2)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, input):
        return input.reshape(input.a.shape[0],-1)

class BatchNorm1d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = mt(np.random.normal(size=(1,num_features), loc=0, scale=1))
        self.bias = mt(np.random.normal(size=(1,num_features), loc=0, scale=1))
        self.moving_mean = 0
        self.moving_std = 1
    
    def __call__(self, input):
        if self.training:
            mean = np.mean(input.a)
            std = np.std(input.a)
            input = (input-mean)/(std+self.eps)
            self.moving_mean = self.moving_mean*self.momentum + mean*(1-self.momentum)
            self.moving_std = self.moving_std*self.momentum + std*(1-self.momentum)
        else:
            input = (input-self.moving_mean)/(self.moving_std+self.eps)
        return self.weight*input+self.bias
    
    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}
        
class BatchNorm2d(Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = mt(np.random.normal(size=(1,num_features,1,1), loc=0, scale=1))
        self.bias = mt(np.random.normal(size=(1,num_features,1,1), loc=0, scale=1))
        self.moving_mean = 0
        self.moving_std = 1
    
    def __call__(self, input):
        if self.training:
            mean = np.mean(input.a, axis=(0,2,3), keepdims=True)
            std = np.std(input.a, axis=(0,2,3), keepdims=True)
            input = (input-mean)/(std+self.eps)
            self.moving_mean = self.moving_mean*self.momentum + mean*(1-self.momentum)
            self.moving_std = self.moving_std*self.momentum + std*(1-self.momentum)
        else:
            input = (input-self.moving_mean)/(self.moving_std+self.eps)
        return self.weight*input+self.bias
    
    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

class ReLu(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        return inputs.LeakeyReLu()

class LeakeyReLu(Layer):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, inputs):
        return inputs.LeakeyReLu(self.negative_slope)

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, inputs):
        return inputs.Sigmoid()

class Softmax(Layer):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def __call__(self, inputs):
        max = np.max(inputs.a, axis=self.dim, keepdims=True)
        inputs = (inputs - max).exp()
        sum = inputs.sum(axis=self.dim, keepdims=True)
        inputs = inputs/sum
        return inputs