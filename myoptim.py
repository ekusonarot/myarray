from re import X
from myarray import MyArray

class MyOptim:
    class SGD:
        def __init__(self, params, lr):
            self.params = params
            self.lr = lr
        
        def step(self):
            for param in self.params:
                param -= (MyArray.grad(param)*self.lr)
                

        def zero_grad(self):
            for param in self.params:
                MyArray.zero_grad(param)

if __name__ == "__main__":
    from module import Module
    from mylayer import MyLayer
    import pickle
    class Model(Module):
        def __init__(self):
            self.linear1 = MyLayer.Linear(2, 16)
            self.linear2 = MyLayer.Linear(16, 8)
            self.linear3 = MyLayer.Linear(8, 4)
            self.linear4 = MyLayer.Linear(4, 1)
            self.relu = MyLayer.LeakeyReLu()
            self.sigmoid = MyLayer.Sigmoid()
        def __call__(self, inputs):
            x = self.linear1(inputs)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            x = self.relu(x)
            x = self.linear4(x)
            x = self.sigmoid(x)
            return x
    model = Model()
    model.eval()
    model.train()
    optim = MyOptim.SGD(params=[model.linear1.get_params(), model.linear2.get_params(), model.linear3.get_params(), model.linear4.get_params()], lr=1e-5)
    inputs = MyArray.from_array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = MyArray.from_array([
        [0],
        [1],
        [1],
        [1]
    ])
    epoch=100
    for e in range(epoch):
        optim.zero_grad()
        x = model(inputs)
        loss = (x-targets)**2
        loss.sum().backward()
        optim.step()
        bar = int(e/epoch*40)
        print("\r[{}]{}".format("="*bar+"-"*(40-bar),loss.sum()), end="")
        del loss
    
    with open("state.pkl", "wb") as f:
        model.eval()
        pickle.dump(model.state_dict(), f)

    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)