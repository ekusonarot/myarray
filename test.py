import torch
from mytorch.module import Module
from mytorch.layer import Flatten, Linear, LeakeyReLu, MaxPool2d, Sigmoid, Conv2d, Softmax
from mytorch.loss import CrossEntropyLoss, MSELoss
from mytorch.optim import Adam
from mytorch.tensor import MyTensor
import numpy as np
import pickle
import sys

def test1():
    class Model(Module):
        def __init__(self):
            self.linear1 = Linear(2, 3)
            self.linear2 = Linear(3, 2)
            self.linear3 = Linear(2, 2)
            self.relu = LeakeyReLu()
            self.softmax= Softmax()
        def __call__(self, inputs):
            x = self.linear1(inputs)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = x.reshape(x.a.shape[0],-1).reshape(x.a.shape[0],x.a.shape[1])
            x = self.linear3(x)
            x = self.softmax(x)
            return x
    model = Model()

    celoss = CrossEntropyLoss()
    optim = Adam(params=model.get_params(), lr=1e-2)
    inputs = MyTensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = MyTensor([
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1]
    ])
    epoch=10000
    for e in range(epoch):
        optim.zero_grad()
        x = model(inputs)
        loss = celoss(x, targets)
        loss.sum().backward()
        optim.step()
        bar = int(e/epoch*40)
        print("\r[{}]{}".format("="*bar+"-"*(40-bar),loss.sum()), end="")
        del loss
    print("")    
    with open("state.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)
    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)

def test2():
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(1, 2, 3, 1, padding=3//2)
            self.maxpool1 = MaxPool2d(2, 2, 0)
            self.relu = LeakeyReLu(0.1)
            self.conv2 = Conv2d(2, 1, 3, 1, padding=3//2)
            self.flatten = Flatten()
            self.linear = Linear(196, 1)
            self.sigmoid = Sigmoid()

        def __call__(self, inputs):
            x = self.conv1(inputs)
            x = self.maxpool1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.flatten(x)
            x = self.linear(x)
            x = self.sigmoid(x)
            return x
    model = Model()
    optim = Adam(params = model.get_params(), lr=1e-7)
    mseloss = MSELoss()

    inputs = MyTensor(np.random.rand(16,1,28,28))
    targets = MyTensor([[0.],[1.],[1.],[0.],[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])

    epoch=10000
    for e in range(epoch):
        optim.zero_grad()
        x = model(inputs)
        loss = mseloss(x, targets)
        loss.sum().backward()
        optim.step()
        bar = int(e/epoch*40)
        print("\r[{}]{}".format("="*bar+"-"*(40-bar),loss.sum()), end="")
        del loss
    print("")
    with open("state.pkl", "wb") as f:
        pickle.dump(model.state_dict(), f)
    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)

def test3():
    import torch
    x = np.array([[0.3, 0.7, 0.5],[1.0, 0.0, 0.1]])
    y = np.array([[0.0, 1.0, 0.0],[1.0, 0.0, 0.0]])
    a = MyTensor(x)
    b = MyTensor(y)
    sigmoid = Sigmoid()
    celoss = MSELoss()
    c = sigmoid(a)
    #c = celoss(a,b)
    print(c)
    c.sum().backward()
    print("grad", a.grad)
    sigmoid = torch.nn.Sigmoid()
    celoss = torch.nn.MSELoss()
    a = torch.tensor(x, requires_grad=True)
    b = torch.tensor(y, requires_grad=True)
    print()
    c = sigmoid(a)
    #c = celoss(a,b)
    print(c)
    c.sum().backward()
    print("grad", a.grad)

def test4():
    import torch
    a = np.array([[2.,3.],[5.,5.]])
    x = MyTensor(a)
    y = x.sum(axis=1)
    print(y)
    y.sum().backward()
    print(x.grad)
    x = torch.tensor(a, requires_grad=True)
    y = x.sum(axis=1)
    print(y)
    y.sum().backward()
    print(x.grad)

def test5():
    a = np.array([[5., 6.],[1., 2.]])
    b = np.array([[3.], [4.]])
    x = MyTensor(a)
    y = MyTensor(b)
    z = (x/y).log()
    z.sum().backward()
    print(x.grad)
    print(y.grad)
    print()
    x = torch.tensor(a, requires_grad=True)
    y = torch.tensor(b, requires_grad=True)
    z = torch.log(x/y)
    z.sum().backward()
    print(x.grad)
    print(y.grad)

def test6():
    import torch
    x = np.array([[0.8, 0.2],[0.3, 0.7]])
    y = np.array([[1.0, 0.0],[1.0, 0.0]])
    a = MyTensor(x)
    b = MyTensor(y)
    c = a.exp()
    sum = c.sum(axis=1, keepdims=True)
    c = c/sum
    print("c",c)
    print("\nsum",sum)
    c.sum().backward()
    print("grad", a.grad)
    a = torch.tensor(x, requires_grad=True)
    b = torch.tensor(y, requires_grad=True)
    print()
    c = torch.exp(a)
    sum = c.sum(dim=1)
    c = c/sum
    print("c",c)
    print("\nsum",sum)
    c.sum().backward()
    print("grad", a.grad)

if __name__ == "__main__":
    arg = sys.argv[1]
    print(arg)
    if arg == "test1":
        test1()
    elif arg == "test2":
        test2()
    elif arg == "test3":
        test3()
    elif arg == "test4":
        test4()
    elif arg == "test5":
        test5()
    elif arg == "test6":
        test6()
