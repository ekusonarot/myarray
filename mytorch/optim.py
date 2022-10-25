from mytorch.tensor import MyTensor
import numpy as np

class Optim:
    def __init__(self, params, lr=1e-2):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for param in self.params:
            p = param
            if type(param) == dict:
                p = param["params"]
            MyTensor.zero_grad(p)

    def step(self):
        for i, param in enumerate(self.params):
            lr = self.lr
            p = param
            if type(param) == dict:
                p = param["params"]
                if "lr" in param:
                    lr = param["lr"]
            self.update(p, lr, i)
    
    def update(self, param, lr, i):
        pass

class SGD(Optim):
    def __init__(self, params, lr=1e-2, momentum=0.):
        super().__init__(params, lr)
        self.momentum = momentum
        self.past_w = [None] * len(params)

    def update(self, param, lr, i):
        grad = MyTensor.grad(param)
        if type(self.past_w[i]) == type(None):
            param -= (grad*lr)
            self.past_w[i] = (grad*lr)
        else:
            param -= (grad*lr+self.past_w[i]*self.momentum)
            self.past_w[i] = (grad*lr+self.past_w[i]*self.momentum)

class Adam(Optim):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-08):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [None] * len(params)
        self.v = [None] * len(params)
    
    def update(self, param, lr, i):
        grad = MyTensor.grad(param)
        if type(self.m[i]) == type(None):
            self.m[i] = self.betas[0]+grad*(1-self.betas[0])
            self.v[i] = self.betas[1]+grad**2*(1-self.betas[1])
        else:
            self.m[i] = self.m[i]*self.betas[0]+grad*(1-self.betas[0])
            self.v[i] = self.v[i]*self.betas[1]+grad**2*(1-self.betas[1])
        m = self.m[i]/(1-self.betas[0])
        v = self.v[i]/(1-self.betas[1])
        param -= lr*m/np.sqrt(v+self.eps)

if __name__ == "__main__":
    from module import Module
    from layer import MyLayer
    from loss import Loss
    import pickle
    celoss = Loss.CrossEntropyLoss()
    class Model(Module):
        def __init__(self):
            self.linear1 = MyLayer.Linear(2, 20)
            self.linear2 = MyLayer.Linear(20, 10)
            self.linear3 = MyLayer.Linear(10, 2)
            self.relu = MyLayer.LeakeyReLu()
            self.softmax= MyLayer.Softmax()
        def __call__(self, inputs):
            x = self.linear1(inputs)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            x = self.softmax(x)
            return x
    model = Model()
    model.eval()
    model.train()
    optim = Adam(params=model.get_params(), lr=1e-1)
    inputs = MyTensor([
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
    epoch=10
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