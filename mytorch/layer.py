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
        self.bias = None
        self.weight = MyArray.from_array(
            np.random.normal(size=(out_channels, in_channels, kernel_size, kernel_size),loc=0,scale=1)
        )

    def __call__(self, inputs):
        col = self.__im2col(inputs)
        pass

    def get_params(self):
        return {"weight": self.weight, "bias": self.bias}

    def __im2col(self, images):
        stride = self.stride
        pad = self.padding
        filters = self.out_channels
        if images.ndim == 2:
            images = images.reshape(1, 1, *images.shape)
        elif images.ndim == 3:
            B, I_h, I_w = images.shape
            images = images.reshape(B, 1, I_h, I_w)
        B, C, I_h, I_w = images.shape
        if isinstance(filters, tuple):
            if len(filters) == 2:
                filters = (1, 1, *filters)
            elif len(filters) == 3:
                M, F_h, F_w = filters
                filters = (M, 1, F_h, F_w)
            _, _, F_h, F_w = filters
        else:
            if filters.ndim == 2:
                filters = filters.reshape(1, 1, *filters.shape)
            elif filters.ndim == 3:
                M, F_h, F_w = filters.shape
                filters = filters.reshape(M, 1, F_h, F_w)
            _, _, F_h, F_w = filters.shape
        
        if isinstance(stride, tuple):
            stride_ud, stride_lr = stride
        else:
            stride_ud = stride
            stride_lr = stride
        if isinstance(pad, tuple):
            pad_ud, pad_lr = pad
        elif isinstance(pad, int):
            pad_ud = pad
            pad_lr = pad
        elif pad == "same":
            pad_ud = 0.5*((I_h - 1)*stride_ud - I_h + F_h)
            pad_lr = 0.5*((I_w - 1)*stride_lr - I_w + F_w)
        pad_zero = (0, 0)
        
        O_h = int((I_h - F_h + 2*pad_ud)//stride_ud + 1)
        O_w = int((I_w - F_w + 2*pad_lr)//stride_lr + 1)
        
        result_pad = (pad_ud, pad_lr)
        pad_ud = int(np.ceil(pad_ud))
        pad_lr = int(np.ceil(pad_lr))
        pad_ud = (pad_ud, pad_ud)
        pad_lr = (pad_lr, pad_lr)
        images = np.pad(images, [pad_zero, pad_zero, pad_ud, pad_lr], \
                        "constant")
        
        cols = np.empty((B, C, F_h, F_w, O_h, O_w))
        for h in range(F_h):
            h_lim = h + stride_ud*O_h
            for w in range(F_w):
                w_lim = w + stride_lr*O_w
                cols[:, :, h, w, :, :] \
                    = images[:, :, h:h_lim:stride_ud, w:w_lim:stride_lr]
        
        results = []
        results.append(cols.transpose(1, 2, 3, 0, 4, 5).reshape(C*F_h*F_w, B*O_h*O_w))
        results.append((O_h, O_w))
        results.append(result_pad)
        return results

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

if __name__ == "__main__":
    linear = Linear(3, 2)
    relu = ReLu()
    a = MyArray.from_array(np.ones((4, 3)))
    b = linear(a)
    b = relu(b)
    print(b)