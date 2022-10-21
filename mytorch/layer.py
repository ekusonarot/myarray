from mytorch.array import MyArray
import numpy as np


class Layer:
    def __init__(self):
        self.weight = None
        self.bias = None

    def __call__(self):
        pass

    def get_params(self):
        pass

    def im2col(self, input_data, kernel_size, stride=1, pad=0):
        if isinstance(kernel_size, tuple):
            filter_h = kernel_size[0]
            filter_w = kernel_size[1]
        else:
            filter_h = kernel_size
            filter_w = kernel_size
        
        # 入力データのサイズを取得
        N, C, H, W = input_data.shape
        
        # 出力データのサイズを計算
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        # パディング
        img = MyArray.from_array(np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant"))
        
        # 出力データの受け皿を初期化
        col = MyArray.from_array(np.zeros((N, C, filter_h, filter_w, out_h, out_w)))
        
        # 行方向のインデックス
        for y in range(filter_h):
            # 行方向の最大値を計算
            y_max = y + stride * out_h
            
            # 列方向のインデックス
            for x in range(filter_w):
                # 列方向の最大値を計算
                x_max = x + stride * out_w
                
                # フィルターのy,x要素に対応する入力データの要素を抽出
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        
        # 出力サイズに整形
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col, out_h, out_w
        
class Linear(Layer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = MyArray.from_array(np.random.normal(size=(in_features, out_features), loc=0, scale=1))
        self.bias_flag = bias
        self.bias = bias
        if bias:
            self.bias = MyArray.from_array(np.zeros(out_features))
        else:
            self.bias = 0.
    
    def __call__(self, inputs):
        return np.dot(inputs, self.weight) + self.bias
    
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
            self.bias = MyArray.from_array(np.random.normal(size=(1,out_channels,1,1), loc=0, scale=1))
        self.weight = MyArray.from_array(
            np.random.normal(size=(out_channels, in_channels*kernel_size*kernel_size),loc=0,scale=1)
        ).T

    def __call__(self, inputs):
        input_col, out_h, out_w = self.im2col(inputs, self.kernel_size, self.stride, self.padding)
        return np.dot(input_col, self.weight).reshape(inputs.shape[0], out_h, out_w, -1).transpose(0, 3, 1, 2) + self.bias

    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def __call__(self, inputs):
        input_col, out_h, out_w = self.im2col(inputs, self.kernel_size, self.stride, self.padding)
        input_col = input_col.reshape(-1,self.kernel_size**2)
        input_col = np.max(input_col, axis=1)
        return input_col.reshape(inputs.shape[0], out_h, out_w, -1).transpose(0, 3, 1, 2)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
    
    def __call__(self, input):
        return input.reshape((input.shape[0],)+(-1,))

class ReLu(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        return np.array([i.ReLu() for i in inputs.reshape(-1)]).reshape(inputs.shape)

class LeakeyReLu(Layer):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def __call__(self, inputs):
        return np.array([i.LeakeyReLu(self.negative_slope) for i in inputs.reshape(-1)]).reshape(inputs.shape)

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