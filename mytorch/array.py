from multiprocessing import Process
import numpy as np
import math

class MyArray():
    def __init__(self, a, *parents):
        self.a = float(a)
        # parents = [(parent, deriv, self, other)]
        self.parents = parents
        self.grad = 0.

    def from_array(a):
        if type(a) != np.ndarray:
            a = np.array(a)
        b = np.array([MyArray(i) if type(i)!=MyArray else i for i in a.reshape(-1)])
        b = b.reshape(a.shape)
        return b

    def backward(self):
        self.grad = 1
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
            '''
        while True:
            next = []
            next_id = []
            args = [(node, next, next_id) for node in nodes]
            p = [Process(target=self.b, args=arg) for arg in args]
            [i.start() for i in p]
            [i.join() for i in p]
            nodes = next
            if len(nodes) == 0:
                break
            '''

    def b(self, node, next, next_id):
        for parent in node.parents:
            parent[0].grad += parent[1](node.grad, parent[2], parent[3])
            if id(parent[0]) not in next_id:
                next.append(parent[0])
                next_id.append(id(parent[0]))

    def __float__(self) -> float:
        return self.a

    def __str__(self) -> str:
        return str(f"{self.a:.4f}")
    
    def __repr__(self) -> str:
        return str(f"{self.a:.4f}")
    
    def __deriv_add__(grad, a, b):
        return grad

    def __add__(self, other):
        if type(self) == type(other):
            return MyArray(self.a + other.a,
            (self, MyArray.__deriv_add__, self.a, other.a),
            (other, MyArray.__deriv_add__, self.a, other.a))
        self.a += other
        return self

    def __deriv_sub1__(grad, a, b):
        return grad

    def __deriv_sub2__(grad, a, b):
        return -grad

    def __sub__(self, other):
        if type(self) == type(other):
            return MyArray(self.a - other.a,
            (self, MyArray.__deriv_sub1__, self.a, other.a),
            (other, MyArray.__deriv_sub2__, self.a, other.a))
        self.a -= other
        return self

    def __deriv_div1__(grad, a, b):
        return 1/b*grad

    def __deriv_div2__(grad, a, b):
        return -1*a/(b**2)*grad
    
    def __truediv__(self, other):
        if type(self) == type(other):
            return MyArray(self.a / other.a,
            (self, MyArray.__deriv_div1__, self.a, other.a),
            (other, MyArray.__deriv_div2__, self.a, other.a))
        return MyArray(self.a / other, (self, MyArray.__deriv_div1__, self.a, other))
        
    def __deriv_mul1__(grad, a, b):
        return b*grad

    def __deriv_mul2__(grad, a, b):
        return a*grad

    def __mul__(self, other):
        if type(self) == type(other):
            return MyArray(self.a * other.a,
            (self, MyArray.__deriv_mul1__, self.a, other.a),
            (other, MyArray.__deriv_mul2__, self.a, other.a))
        return MyArray(self.a * other, (self, MyArray.__deriv_mul1__, self.a, other))
    
    def __floordiv__(self, other):
        raise RuntimeError("derivative for floor_divide is not implemented")

    def __deriv_pow1__(grad, a, b):
        return b*(a**(b-1))*grad

    def __deriv_pow2__(grad, a, b):    
        return a**b*math.log(a)*grad

    def __pow__(self, other):
        if type(self) == type(other):
            return MyArray(self.a ** other.a,
            (self, MyArray.__deriv_pow1__, self.a, other.a),
            (other, MyArray.__deriv_pow2__, self.a, other.a))
        return MyArray(self.a ** other, (self, MyArray.__deriv_pow1__, self.a, other))

    def __mod__(self, other):
        if type(self) == type(other):
            raise RuntimeError("the derivative for 'other' is not implemented")
        return MyArray(self.a % other, (self, MyArray.__deriv_add__, self.a, other))
    
    def __deriv_exp__(grad, a, b):
        return math.exp(a)*grad

    def exp(self):
        a = min(1e+2, self.a)
        return MyArray(math.exp(a), (self, MyArray.__deriv_exp__, a, None))
        
    def __deriv_log__(grad, a, b):
        return 1/a*grad

    def log(self):
        return MyArray(math.log(self.a), (self, MyArray.__deriv_log__, self.a, None))
        
    def __pos__(self):
        return MyArray(+self.a, (self, MyArray.__deriv_add__, self.a, None))

    def __neg__(self):
        return MyArray(-self.a, (self, MyArray.__deriv_sub2__, None, self.a))

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

    def __deriv_ReLu1__(grad, a, b):
        return grad

    def __deriv_ReLu2__(grad, a, b):
        return 0

    def __deriv_ReLu3__(grad, a, b):
        return b*grad

    def ReLu(self):
        if 0 < self.a:
            return MyArray(self.a, (self, MyArray.__deriv_ReLu1__, self.a, None))
        else:
            return MyArray(0., (self, MyArray.__deriv_ReLu2__, self.a, None))

    def __deriv_Sigmoid__(grad, a, b):
        return (1.-a)*a*grad

    def Sigmoid(self):
        x = max(-1e+2, self.a)
        x = 1./(1.+math.exp(-x))
        return MyArray(x, (self, MyArray.__deriv_Sigmoid__, x, None))

    def LeakeyReLu(self, negative_slope=0.01):
        if 0 < self.a:
            return MyArray(self.a, (self, MyArray.__deriv_ReLu1__, self.a, negative_slope))
        else:
            return MyArray(self.a*negative_slope, (self, MyArray.__deriv_ReLu3__, self.a, negative_slope))
    
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
        return a+a*b+(-a)
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

def test4():
    from layer import Conv2d
    conv = Conv2d(3, 2)
    a = MyArray.from_array(np.ones((1,1,100,100)))
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
    