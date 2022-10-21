from mytorch.module import Module
from mytorch.layer import Flatten, Linear, LeakeyReLu, MaxPool2d, Sigmoid, Conv2d
from mytorch.loss import MSELoss
from mytorch.optim import Adam
from mytorch.array import MyArray
import numpy as np
import pickle
import sys

def test1():
    class Model(Module):
        def __init__(self):
            self.linear1 = Linear(2, 2)
            self.linear2 = Linear(2, 1)
            self.relu = LeakeyReLu()
            self.sigmoid= Sigmoid()
        def __call__(self, inputs):
            x = self.linear1(inputs)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.sigmoid(x)
            return x
    model = Model()
    model.eval()
    model.train()

    celoss = MSELoss()
    optim = Adam(params=model.get_params(), lr=1e-1)
    inputs = MyArray.from_array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = np.array([
        [0],
        [1],
        [1],
        [0]
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
        model.eval()
        pickle.dump(model.state_dict(), f)
    inputs = MyArray.from_array([
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]
        ])
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
            self.relu = LeakeyReLu()
            self.conv2 = Conv2d(2, 1, 3, 1, padding=3//2)
            self.flatten = Flatten()
            self.linear = Linear(14*14, 1)
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
    optim = Adam(model.get_params(), lr=1e-3)
    mseloss = MSELoss()

    inputs = MyArray.from_array(np.random.rand(16,1,28,28))
    targets = np.array([[0.],[1.],[1.],[0.],[1.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]])

    epoch=10
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
        model.eval()
        pickle.dump(model.state_dict(), f)
    inputs = inputs[:4]
    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)

if __name__ == "__main__":
    arg = sys.argv[1]
    print(arg)
    if arg == "test1":
        test1()
    elif arg == "test2":
        test2()
