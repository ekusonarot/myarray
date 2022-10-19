import numpy as np
import math

class MyArray():
    def __init__(self, a, *parents):
        self.a = float(a)
        self.parents = parents
        self.grad = 0.

    def from_array(a):
        if type(a) != np.ndarray:
            a = np.array(a)
        b = np.array([MyArray(i) for i in a.reshape(-1)])
        b = b.reshape(a.shape)
        return b

    def backward(self):
        parents_grad = [(self.parents, 1.)]
        while True:
            next = []
            for parents, grad in parents_grad:
                for parent in parents:
                    parent[0].grad += parent[1](grad)
            next = [[(parent[0].parents, parent[0].grad) for parent in parents] for parents, _ in parents_grad ]
            parents_grad = sum(next, [])
            if len(parents_grad) == 0:
                break
                

    def __float__(self) -> float:
        return self.a

    def __str__(self) -> str:
        return str(f"{self.a:.4f}")
    
    def __repr__(self) -> str:
        return str(f"{self.a:.4f}")
    
    def __add__(self, other):
        if type(self) == type(other):
            return MyArray(self.a + other.a,
            (self, lambda grad: 1*grad),
            (other, lambda grad: 1*grad))
        self.a += other
        return self

    def __sub__(self, other):
        if type(self) == type(other):
            return MyArray(self.a - other.a,
            (self, lambda grad: 1*grad),
            (other, lambda grad: -1*grad))
        self.a -= other
        return self
    
    def __truediv__(self, other):
        if type(self) == type(other):
            return MyArray(self.a / other.a,
            (self, lambda grad: 1/other.a*grad),
            (other, lambda grad: -1*(self.a)/(other.a**2)*grad))
        return MyArray(self.a / other, (self, lambda grad: 1/other*grad))
        
    def __mul__(self, other):
        if type(self) == type(other):
            return MyArray(self.a * other.a,
            (self, lambda grad: other.a*grad),
            (other, lambda grad: self.a*grad))
        return MyArray(self.a * other, (self, lambda grad: other*grad))
    
    def __floordiv__(self, other):
        raise RuntimeError("derivative for floor_divide is not implemented")

    def __pow__(self, other):
        if type(self) == type(other):
            return MyArray(self.a ** other.a,
            (self, lambda grad: other.a*(self.a**(other.a-1))*grad),
            (other, lambda grad: self.a**other.a*math.log(self.a)*grad))
        return MyArray(self.a ** other, (self, lambda grad: other*(self.a**(other-1))*grad))

    def __mod__(self, other):
        if type(self) == type(other):
            raise RuntimeError("the derivative for 'other' is not implemented")
        return MyArray(self.a % other, (self, lambda grad: 1*grad))
    
    def exp(self):
        a = min(1e+2, self.a)
        return MyArray(math.exp(a), (self, lambda grad: math.exp(a)*grad))
        
    def log(self):
        return MyArray(math.log(self.a), (self, lambda grad: 1/self.a*grad))
        
    def __pos__(self):
        return MyArray(+self.a, (self, lambda grad: 1*grad))

    def __neg__(self):
        return MyArray(-self.a, (self, lambda grad: -1*grad))

    def __lt__(self, other):
        if type(self) == type(other):
            return MyArray(self.a < other.a)
        return MyArray(self.a < other)

    def __le__(self, other):
        if type(self) == type(other):
            return MyArray(self.a <= other.a)
        return MyArray(self.a <= other)

    def __eq__(self, other):
        if type(self) == type(other):
            return MyArray(self.a == other.a)
        return MyArray(self.a == other)

    def __ne__(self, other):
        if type(self) == type(other):
            return MyArray(self.a != other.a)
        return MyArray(self.a != other)

    def __ge__(self, other):
        if type(self) == type(other):
            return MyArray(self.a >= other.a)
        return MyArray(self.a >= other)

    def __gt__(self, other):
        if type(self) == type(other):
            return MyArray(self.a > other.a)
        return MyArray(self.a > other)
    
    def grad(array):
        if type(array) != np.ndarray:
            raise RuntimeError("not np.ndarray: {}".format(type(array)))
        grad = np.array([i.grad for i in array.reshape(-1,)]).reshape(array.shape)
        return grad

    def __zero_grad(self):
        self.grad = 0.

    def zero_grad(array):
        if type(array) != np.ndarray:
            raise RuntimeError("not np.ndarray: {}".format(type(array)))
        [i.__zero_grad() for i in array.reshape(-1,)]

    def ReLu(self):
        if 0 < self.a:
            return MyArray(self.a, (self, lambda grad: grad))
        else:
            return MyArray(0., (self, lambda grad: 0.))

    def Sigmoid(self):
        def sigmoid(x):
            x = max(-1e+2, x)
            return 1./(1.+math.exp(-x))
        return MyArray(sigmoid(self.a), (self, lambda grad: (1.-sigmoid(self.a))*sigmoid(self.a)*grad))

    def LeakeyReLu(self, negative_slope=0.01):
        if 0 < self.a:
            return MyArray(self.a, (self, lambda grad: grad))
        else:
            return MyArray(self.a*negative_slope, (self, lambda grad: negative_slope*grad))
    
    def Exp(array):
        if type(array) != np.ndarray:
            raise RuntimeError("not np.ndarray: {}".format(type(array)))
        return np.array([i.exp() for i in array.reshape(-1)]).reshape(array.shape)
    
    def Log(array):
        if type(array) != np.ndarray:
            raise RuntimeError("not np.ndarray: {}".format(type(array)))
        return np.array([i.log() for i in array.reshape(-1)]).reshape(array.shape)



def test1():
    x, y = 3., 2.
    def test(a, b):
        return a+a/b+b*a
    a = MyArray(x)
    b = MyArray(y)
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
        return a/b
    a = MyArray.from_array(x)
    b = MyArray.from_array(y)
    c = test(a,b).sum()
    c.backward()
    print(MyArray.grad(a))
    print(MyArray.grad(b))
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
    a = MyArray.from_array(array1)
    b = MyArray.from_array(array2)
    c = Softmax()(a)
    c = (b-c)**2
    c.sum().backward()
    print(MyArray.grad(a))
    a = torch.tensor(array1, requires_grad=True)
    b = torch.tensor(array2, requires_grad=True)
    c = torch.nn.Softmax(dim=1)(a)
    c = (b-c)**2
    c.sum().backward()
    print(a.grad)


if __name__ == "__main__":
    import sys
    import torch
    if "test1" == sys.argv[1]:
        test1()
    elif "test2" == sys.argv[1]:
        test2()
    elif "test3" == sys.argv[1]:
        test3()
    