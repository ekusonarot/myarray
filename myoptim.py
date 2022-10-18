from myarray import MyArray

class MyOptim:
    class SGD:
        def __init__(self, params, lr):
            self.params = params
            self.lr = lr
        
        def step(self):
            for param in self.params:
                param -= MyArray.grad(param)*self.lr

        def zero_grad(self):
            for param in self.params:
                MyArray.zero_grad(param)

if __name__ == "__main__":
    from mylayer import MyLayer
    linear1 = MyLayer.Linear(2, 1)
    linear2 = MyLayer.Linear(4, 1)
    relu = MyLayer.LeakeyReLu()
    sigmoid = MyLayer.Sigmoid()
    
    optim = MyOptim.SGD(params=[linear1.get_params(), linear2.get_params()], lr=1e-1)

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
        [0]
    ])
    for e in range(1000):
        optim.zero_grad()
        x = linear1(inputs)
        #x = relu(x)
        #x = linear2(x)
        #x = sigmoid(x)
        print(x)
        loss = (x-targets)**2
        loss.sum().backward()
        optim.step()