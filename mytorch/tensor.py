import numpy as np
import math

class MyTensor():
    def __init__(self, a, *parents, add=False):
        self.a = a
        if type(a) != np.ndarray:
            self.a = np.array(a)
        # parents = [(parent, deriv, self, other)]
        self.parents = parents
        self.grad = np.zeros(self.a.shape)
        self.add = add

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
                    parent[0].grad += parent[1](node.grad, parent[2], parent[3])
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
            return grad.sum(axis=0)
        return grad

    def __deriv_add2__(grad, a, b):
        if grad.shape != b.shape:
            return grad.sum(axis=0)
        return grad

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
        r = 1/b*grad
        if r.shape != a.shape:
            return r.sum(axis=-1, keepdims=True)
        return r

    def __deriv_div2__(grad, a, b):
        r = -1*a/(b**2)*grad
        if r.shape != b.shape:
            return r.sum(axis=-1, keepdims=True)
        return r
    
    def __truediv__(self, other):
        if type(self) == type(other):
            return MyTensor(self.a / other.a,
            (self, MyTensor.__deriv_div1__, self.a, other.a),
            (other, MyTensor.__deriv_div2__, self.a, other.a))
        return MyTensor(self.a / other, (self, MyTensor.__deriv_div1__, self.a, other))
        
    def __deriv_mul1__(grad, a, b):
        r = b*grad
        if r.shape != a.shape:
            return r.sum(axis=-1, keepdims=True)
        return r

    def __deriv_mul2__(grad, a, b):
        r = a*grad
        if r.shape != b.shape:
            return r.sum(axis=-1, keepdims=True)
        return r

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
        return MyTensor(np.log(self.a), (self, MyTensor.__deriv_log__, self.a, None))
        
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
        x = max(-1e+2, self.a)
        x = 1./(1.+math.exp(-x))
        return MyTensor(x, (self, MyTensor.__deriv_Sigmoid__, x, None))

    def __deriv_LeakeyReLu__(grad, a, negative_slope):
        return np.where(0. < a, 1, negative_slope)*grad

    def LeakeyReLu(self, negative_slope=0.01):
        return MyTensor(np.where(0. < self.a, self.a, self.a*negative_slope),
            (self, MyTensor.__deriv_LeakeyReLu__, self.a, negative_slope)
        )



def test1():
    x, y = 3., 2.
    def test(a, b):
        return a**2.-(b/2.+a)*3.+1.
    a = MyTensor(x)
    b = MyTensor(y)
    c = test(a,b)
    c.backward()
    print(a.grad)
    print(b.grad)
    a = torch.tensor(x, requires_grad=True)
    b = torch.tensor(y, requires_grad=True)
    c = test(a,b)
    c.backward()
    print(a.grad)
    print(b.grad)

def test2():
    x, y = [[1.,2.],[1., 2.]], [3., 4.]
    def test(a, b):
        return a*b-b/3+a**2+a
    a = MyTensor(x)
    b = MyTensor(y)
    c = test(a,b).sum()
    c.backward()
    print(MyTensor.grad(a))
    print(MyTensor.grad(b))
    a = torch.tensor(x, requires_grad=True)
    b = torch.tensor(y, requires_grad=True)
    c = test(a,b).sum()
    c.backward()
    print(a.grad)
    print(b.grad)

def test3():
    array1 = [[1.,1.,10.],[1.,3.,1.]]
    array2 = [[0.,0.,1.],[1.,0.,0.]]
    from layer import Softmax
    a = MyTensor(array1)
    b = MyTensor(array2)
    c = Softmax()(a)
    c = (b-c)**2
    c.sum().backward()
    print(MyTensor.grad(a))
    a = torch.tensor(array1, requires_grad=True)
    b = torch.tensor(array2, requires_grad=True)
    c = torch.nn.Softmax(dim=1)(a)
    c = (b-c)**2
    c.sum().backward()
    print(a.grad)

def test4():
    from layer import Conv2d
    conv = Conv2d(3, 2)
    a = MyTensor(np.ones((1,1,100,100)))
    conv(a)


if __name__ == "__main__":
    import sys
    import torch
    if "test1" == sys.argv[1]:
        test1()
    elif "test2" == sys.argv[1]:
        test2()
    elif "test3" == sys.argv[1]:
        test3()
    elif "test4" == sys.argv[1]:
        test4()
    