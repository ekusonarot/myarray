import numpy as np
import random

class Dataloader:
    def __init__(self, dataset, batch_size=64, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.start = 0
        if shuffle:
            random.shuffle(self.indices)
    def __iter__(self):
        return self

    def __next__(self):
        if len(self.dataset) <= self.start:
            if self.shuffle:
                random.shuffle(self.indices)
            raise StopIteration()
        data = np.array([self.dataset[i] for i in self.indices[self.start:self.start+self.batch_size]])
        self.start += self.batch_size
        return data

if __name__ == "__main__":
    dataset = np.arange(64)
    dataloader = Dataloader(dataset, batch_size=4)
    for i in dataloader:
        print(i)