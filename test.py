from mytorch.module import Module
from mytorch.layer import Linear, LeakeyReLu, Softmax
from mytorch.loss import CrossEntropyLoss, MSELoss
from mytorch.optim import Adam
from mytorch.array import MyArray
import numpy as np
import pickle

if __name__ == "__main__":
    celoss = MSELoss()
    class Model(Module):
        def __init__(self):
            self.linear1 = Linear(2, 4)
            self.linear2 = Linear(4, 2)
            self.relu = LeakeyReLu()
            self.softmax= Softmax()
        def __call__(self, inputs):
            x = self.linear1(inputs)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.softmax(x)
            return x
    model = Model()
    model.eval()
    model.train()
    optim = Adam(params=model.get_params(), lr=1e-2)
    inputs = MyArray.from_array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = np.array([
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0]
    ])
    epoch=1000
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

    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)