import numpy as np
import math

class MyTensor():
    def __init__(self, a, *parents, add=False):
        self.a = a
        if type(a) != np.ndarray:
            self.a = np.array(a)
        # parents = [(parent, deriv, self, other)]
        self.parents = parents
        for parent in parents:
            parent[0].child_count += 1
        self.grad = np.zeros(self.a.shape)
        self.add = add
        self.child_count = 0

    def backward(self):
        if self.a.shape != ():
            raise RuntimeError
        self.grad = np.array(1)
        nodes = [self]
        while True:
            next = []
            next_id = []
            for node in nodes:
                for parent in node.parents:
                    parent[0].child_count -= 1
                    parent[0].grad += parent[1](node.grad, parent[2], parent[3])
                    if 0 < parent[0].child_count:
                        continue
                    if id(parent[0]) not in next_id:
                        next.append(parent[0])
                        next_id.append(id(parent[0]))
            nodes = next
            if len(nodes) == 0:
                break

    def b(self, node, next, next_id):
        for parent in node.parents:
            parent[0].grad += parent[1](node.grad, parent[2], parent[3])
            if id(parent[0]) not in next_id:
                next.append(parent[0])
                next_id.append(id(parent[0]))

    def __str__(self) -> str:
        return str(self.a)
    
    def __repr__(self) -> str:
        return str(f"{self.a}<grad: {self.grad}>")
    
    def __deriv_add1__(grad, a, b):
        if grad.shape != a.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(a.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != a.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(a.shape)

    def __deriv_add2__(grad, a, b):
        if grad.shape != b.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(b.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != b.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad.reshape(b.shape)

    def __add__(self, other):
        if type(other) != MyTensor:
            self.a += other
            return self
        elif self.add:
            self.a += other.a
            self.parents += ((other, MyTensor.__deriv_add1__, self.a, other.a),)
            return self
        elif other.add:
            other.a += self.a
            other.parents += ((self, MyTensor.__deriv_add2__, self.a, other.a),)
            return other
        else:
            return MyTensor(self.a + other.a,
            (self, MyTensor.__deriv_add1__, self.a, other.a),
            (other, MyTensor.__deriv_add2__, self.a, other.a), add=True)

    def __deriv_sub1__(grad, a, b):
        if grad.shape != a.shape:
            return grad.sum(axis=0)
        return grad

    def __deriv_sub2__(grad, a, b):
        if grad.shape != b.shape:
            return -grad.sum(axis=0)
        return -grad

    def __sub__(self, other):
        if type(other) != MyTensor:
            self.a -= other
            return self
        elif self.add:
            self.a -= other.a
            self.parents += ((other, MyTensor.__deriv_sub2__, self.a, other.a),)
            return self
        elif other.add:
            other.a = self.a - other.a
            other.parents += ((self, MyTensor.__deriv_sub1__, self.a, other.a),)
            return other
        else:
            return MyTensor(self.a - other.a,
            (self, MyTensor.__deriv_sub1__, self.a, other.a),
            (other, MyTensor.__deriv_sub2__, self.a, other.a), add=True)

    def __deriv_div1__(grad, a, b):
        grad = 1/b*grad
        if grad.shape != a.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(a.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != a.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __deriv_div2__(grad, a, b):
        grad = -1*a/(b**2)*grad
        if grad.shape != b.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(b.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != b.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad
    
    def __truediv__(self, other):
        if type(self) == type(other):
            return MyTensor(self.a / other.a,
            (self, MyTensor.__deriv_div1__, self.a, other.a),
            (other, MyTensor.__deriv_div2__, self.a, other.a))
        return MyTensor(self.a / other, (self, MyTensor.__deriv_div1__, self.a, other))
        
    def __deriv_mul1__(grad, a, b):
        grad = b*grad
        if grad.shape != a.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(a.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != a.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __deriv_mul2__(grad, a, b):
        grad = a*grad
        if grad.shape != b.shape:
            for i in range(-1,-len(grad.shape)-1,-1):
                if i < -len(b.shape):
                    grad = grad.sum(axis=i, keepdims=True)
                elif grad.shape[i] != b.shape[i]:
                    grad = grad.sum(axis=i, keepdims=True)
        return grad

    def __mul__(self, other):
        if type(self) == type(other):
            return MyTensor(self.a * other.a,
            (self, MyTensor.__deriv_mul1__, self.a, other.a),
            (other, MyTensor.__deriv_mul2__, self.a, other.a))
        return MyTensor(self.a * other, (self, MyTensor.__deriv_mul1__, self.a, other))
    
    def __deriv_dot1__(grad, a, b):
        return np.dot(grad, b.T)

    def __deriv_dot2__(grad, a, b):
        return np.dot(a.T, grad)

    def dot(x, y):
        if type(x) == MyTensor and type(y) == MyTensor:
            r = np.dot(x.a, y.a)
            return MyTensor(r,
                (x, MyTensor.__deriv_dot1__, x.a, y.a),
                (y, MyTensor.__deriv_dot2__, x.a, y.a)
            )
        elif type(x) != MyTensor and type(y) != MyTensor:
            r = np.dot(x, y)
            return MyTensor(r)
        elif type(x) == MyTensor:
            r = np.dot(x.a, y)
            return MyTensor(r,
                (x, MyTensor.__deriv_dot1__, x.a, y)
            )
        else:
            r = np.dot(x, y.a)
            return MyTensor(r,
                (y, MyTensor.__deriv_dot1__, x, y.a)
            )
    
    def __deriv_sum__(grad, a, b):
        axis, keepdims = b
        size = a.shape[axis] if axis != None else None
        if size == None:
            return np.ones(a.shape)*grad
        elif keepdims != None:
            grad = np.repeat(grad, size, axis=axis)
        else:
            grad = np.expand_dims(grad, axis=axis)
            grad = np.repeat(grad, size, axis=axis)
        return grad

    def sum(self, axis=None, keepdims=False):
        x = self.a.sum(axis=axis, keepdims=keepdims)
        return MyTensor(x, 
            (self, MyTensor.__deriv_sum__, self.a, (axis, keepdims))
        )
        
    def __floordiv__(self, other):
        raise RuntimeError("derivative for floor_divide is not implemented")

    def __deriv_pow1__(grad, a, b):
        return b*(a**(b-1))*grad

    def __deriv_pow2__(grad, a, b):    
        return a**b*math.log(a)*grad

    def __pow__(self, other):
        if type(self) == type(other):
            return MyTensor(self.a ** other.a,
            (self, MyTensor.__deriv_pow1__, self.a, other.a),
            (other, MyTensor.__deriv_pow2__, self.a, other.a))
        return MyTensor(self.a ** other, (self, MyTensor.__deriv_pow1__, self.a, other))

    def __mod__(self, other):
        if type(self) == type(other):
            raise RuntimeError("the derivative for 'other' is not implemented")
        return MyTensor(self.a % other, (self, MyTensor.__deriv_add__, self.a, other))
    
    def __deriv_exp__(grad, a, b):
        return np.exp(a)*grad

    def exp(self):
        a = np.where(1e+2<self.a, 1e+2, self.a)
        return MyTensor(np.exp(a), (self, MyTensor.__deriv_exp__, a, None))
        
    def __deriv_log__(grad, a, b):
        return 1/a*grad

    def log(self):
        a = np.where(1e-7<self.a, self.a, 1e-7)
        return MyTensor(np.log(a), (self, MyTensor.__deriv_log__, a, None))
        
    def __pos__(self):
        return MyTensor(+self.a, (self, MyTensor.__deriv_add__, self.a, None))

    def __neg__(self):
        return MyTensor(-self.a, (self, MyTensor.__deriv_sub2__, None, self.a))

    def __lt__(self, other):
        if type(self) == type(other):
            return self.a < other.a
        return self.a < other

    def __le__(self, other):
        if type(self) == type(other):
            return self.a <= other.a
        return self.a <= other

    def __eq__(self, other):
        if type(self) == type(other):
            return self.a == other.a
        return self.a == other

    def __ne__(self, other):
        if type(self) == type(other):
            return self.a != other.a
        return self.a != other

    def __ge__(self, other):
        if type(self) == type(other):
            return self.a >= other.a
        return self.a >= other

    def __gt__(self, other):
        if type(self) == type(other):
            return self.a > other.a
        return self.a > other

    def __deriv_T__(grad, a, b):
        return grad.T

    def T(self):
        return MyTensor(self.a.T, 
            (self, MyTensor.__deriv_T__, self.a, None)
        )
    
    def grad(self):
        return self.grad

    def zero_grad(self):
        self.grad = np.zeros(self.a.shape)

    def __deriv_Sigmoid__(grad, a, b):
        return (1.-a)*a*grad

    def Sigmoid(self):
        x = np.where(self.a<-1e+2, -1e+2, self.a)
        x = 1./(1.+np.exp(-x))
        return MyTensor(x, (self, MyTensor.__deriv_Sigmoid__, x, None))

    def __deriv_LeakeyReLu__(grad, a, negative_slope):
        return np.where(0. < a, 1, negative_slope)*grad

    def LeakeyReLu(self, negative_slope=0.01):
        return MyTensor(np.where(0. < self.a, self.a, self.a*negative_slope),
            (self, MyTensor.__deriv_LeakeyReLu__, self.a, negative_slope)
        )

    def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        
        out_h = (H-filter_h+pad*2)//stride+1
        out_w = (W-filter_w+pad*2)//stride+1
        
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        
        for y in range(filter_h):     
            y_max = y + stride*out_h
            
            for x in range(filter_w):          
                x_max = x + stride*out_w
                
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:,:,y,x,:,:]
        return img[:, :, pad:pad+H, pad:pad+W]

    def __deriv_im2col__(col_grad, input_shape, args):
        grad = MyTensor.col2im(col_grad, input_shape, args[0], args[1], args[2], args[3])
        return grad

    def im2col(self, filter_h, filter_w, stride=1, pad=0, constant_values=0):
        input_data = self.a
        N, C, H, W = input_data.shape 
        
        out_h = (H-filter_h+pad*2)//stride+1
        out_w = (W-filter_w+pad*2)//stride+1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)],
                    'constant', constant_values=constant_values)
        
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            
            for x in range(filter_w):
                x_max = x + stride*out_w
                
                col[:, :, y, x, :, :] = img[:,:,y:y_max:stride,x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return MyTensor(col,
            (self, MyTensor.__deriv_im2col__, (N, C, H, W), (filter_h, filter_w, stride, pad))
        ), out_h, out_w
    
    def __deriv_reshape__(grad, a, b):
        return grad.reshape(a.shape)

    def reshape(self, *indices):
        return MyTensor(self.a.reshape(indices),
            (self, MyTensor.__deriv_reshape__, self.a, None)
        )

    def __deriv_transpose__(grad, a, b):
        index = [None]*len(a)
        for i in range(len(a)):
            index[a[i]] = i
        return grad.transpose(index)

    def transpose(self, *indices):
        return MyTensor(self.a.transpose(indices),
            (self, MyTensor.__deriv_transpose__, indices, None)
        )
    
    def max(input, axis=None):
        ind = np.argmax(input.a, axis)
        mask = np.zeros_like(input.a)
        if axis==1:
            for i in range(input.a.shape[0]):
                mask[i,ind[i]] = 1
        return (input*mask).sum(axis=axis, keepdims=True)
    