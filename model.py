from mytorch.module import Module
from mytorch.layer import BatchNorm1d, BatchNorm2d, Flatten, Linear, LeakeyReLu, MaxPool2d, Conv2d, ReLu, Softmax

class Model(Module):
    def __init__(self):
        self.conv1 = Conv2d(1, 64, 3, padding=3//2, bayesian=True)
        self.conv2_1 = Conv2d(64, 16, 1, padding=0, bayesian=True)
        self.conv2_2 = Conv2d(16, 16, 3, padding=3//2, bayesian=True)
        self.conv2_3 = Conv2d(16, 64, 1, padding=0, bayesian=True)
        self.conv3_1 = Conv2d(64, 16, 1, padding=0, bayesian=True)
        self.conv3_2 = Conv2d(16, 16, 3, padding=3//2, bayesian=True)
        self.conv3_3 = Conv2d(16, 64, 1, padding=0, bayesian=True)
        self.conv4_1 = Conv2d(64, 16, 1, padding=0, bayesian=True)
        self.conv4_2 = Conv2d(16, 16, 3, padding=3//2, bayesian=True)
        self.conv4_3 = Conv2d(16, 64, 1, padding=0, bayesian=True)
        self.relu = LeakeyReLu()
        self.bn1 = BatchNorm2d(64, bayesian=True)
        self.bn2 = BatchNorm2d(64, bayesian=True)
        self.bn3 = BatchNorm2d(64, bayesian=True)
        self.bn4 = BatchNorm2d(64, bayesian=True)
        self.maxpool = MaxPool2d(2, 2)
        self.flatten = Flatten()
        self.linear1 = Linear(576, 64, bayesian=True)
        self.bn5 = BatchNorm1d(64)
        self.linear2 = Linear(64, 15, bayesian=True)
        self.softmax = Softmax()

    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        identity_mapping = x
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = identity_mapping + x
        x = self.maxpool(x)
        
        identity_mapping = x
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = identity_mapping + x
        x = self.maxpool(x)

        identity_mapping = x
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.bn4(x)
        x = identity_mapping + x

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x