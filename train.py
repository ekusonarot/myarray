from mytorch.module import Module
from mytorch.layer import BatchNorm1d, BatchNorm2d, Flatten, Linear, LeakeyReLu, MaxPool2d, Conv2d, Softmax
from mytorch.loss import CrossEntropyLoss
from mytorch.optim import Adam
from mytorch.tensor import MyTensor
from mytorch.dataloader import Dataloader
import numpy as np
import pickle
import sys

class Model(Module):
    def __init__(self):
        self.conv1 = Conv2d(1, 32, 3, 1, 3//2)
        self.bn1 = BatchNorm2d(32)
        self.relu1 = LeakeyReLu()
        self.maxpool1 = MaxPool2d(2, 2)

        self.conv2 = Conv2d(32, 16, 3, 1, 3//2)
        self.bn2 = BatchNorm2d(16)
        self.relu2 = LeakeyReLu()
        self.maxpool2 = MaxPool2d(2, 2)

        self.flatten = Flatten()
        self.linear = Linear(784, 15)
        self.bn3 = BatchNorm1d(15)
        self.softmax= Softmax()
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.bn3(x)
        x = self.softmax(x)
        return x

class Dataset:
    def __init__(self, img, target):
        self.img = img
        self.target = target
    
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        img = (img-img.min())/img.max()
        target = self.target[idx]
        return img, target

epoch = 1000
batch_size = 64
if __name__ == "__main__":
    model = Model()
    celoss = CrossEntropyLoss()
    optim = Adam(params=model.get_params(), lr=1e-2)

    train_data = np.load("./1_data/train_data.npy")
    train_label = np.load("./1_data/train_label.npy")
    dataset = Dataset(train_data, train_label)
    dataloader = Dataloader(dataset, batch_size, True)
    size = len(dataset)
    for e in range(epoch):
        progress = 0
        for x, target in dataloader:
            progress += batch_size
            optim.zero_grad()
            y = model(x)
            loss = celoss(y, target)
            loss.backward()
            optim.step()
            
            bar = int(progress/size*40)
            print("\r[{}]{}".format("="*bar+"-"*(40-bar),loss.sum()), end="")
            del loss
        print("")