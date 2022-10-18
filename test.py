from turtle import back
from cv2 import log
import numpy as np

class MyFloat():
    def __init__(self, a, *parents):
        self.a = float(a)
        self.parents = parents
        self.grad = 0

    def from_numpy(a):
        b = np.array([MyFloat(i) for i in a.reshape(-1)])
        b = b.reshape(a.shape)
        return b

    def backward(self):
        for parent in self.parents:
            parent[0].grad += parent[1](self.grad)

    
    def __str__(self) -> str:
        return str(self.a)
    
    def __repr__(self) -> str:
        return str(self.a)
    
    def __add__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a + other.a,
            (self, lambda grad: 1),
            (other, lambda grad: 1))
        return MyFloat(self.a + other, (self, lambda grad: 1))

    def __sub__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a - other.a,
            (self, lambda grad: 1),
            (other, lambda grad: -1))
        return MyFloat(self.a - other, (self, lambda grad: 1))
    
    def __truediv__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a / other.a,
            (self, lambda grad: 1/other.a*grad),
            (other, lambda grad: -1*(self.a)/(other.a**2)*grad))
        return MyFloat(self.a / other, (self, lambda grad: 1/other*grad))
        
    def __mul__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a * other.a,
            (self, lambda grad: other.a*grad),
            (other, lambda grad: self.a*grad))
        return MyFloat(self.a * other, (self, lambda grad: other*grad))
    
    def __floordiv__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a // other.a)
        return MyFloat(self.a // other)

    def __pow__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a ** other.a,
            (self, lambda grad: other.a*(self.a**other.a-1)*grad),
            (other, lambda grad: other.a**self.a*log(other.a)*grad))
        return MyFloat(self.a ** other, (self, lambda grad: 1/other*grad))

    def __mod__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a % other.a)
        return MyFloat(self.a % other)
        
    def __pos__(self):
        return MyFloat(+self.a, (self, lambda grad: 1))

    def __neg__(self):
        return MyFloat(-self.a, (self, lambda grad: -1))

    def __lt__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a < other.a)
        return MyFloat(self.a < other)

    def __le__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a <= other.a)
        return MyFloat(self.a <= other)

    def __eq__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a == other.a)
        return MyFloat(self.a == other)

    def __ne__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a != other.a)
        return MyFloat(self.a != other)

    def __ge__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a >= other.a)
        return MyFloat(self.a >= other)

    def __gt__(self, other):
        if type(self) == type(other):
            return MyFloat(self.a > other.a)
        return MyFloat(self.a > other)

if __name__ == "__main__":
    a = MyFloat(3)
    b = MyFloat(2)
    c = a-b
    c.backward()
    print(a.grad)
    print(b.grad)