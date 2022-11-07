from mytorch.loss import CrossEntropyLoss
from mytorch.optim import Adam
from mytorch.dataloader import Dataloader
import numpy as np
import random
import pickle
from model import Model

class Dataset:
    def __init__(self, img, target, transform=None):
        self.img = img
        self.target = target
        self.transform = transform
    
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]
        if self.transform:
            img = self.transform(images = img.astype(np.uint8))
        img = (img-img.min())/img.max()
        target = self.target[idx]
        target = target*0.9+0.1/15.
        return img, target

def random_split(data, target, length):
    indices = list(range(len(data)))
    random.shuffle(indices)
    train_len = int(len(data)*length)
    val_len = len(data)-train_len
    train_data = [data[indices[i]] for i in range(train_len)]
    train_target = [target[indices[i]] for i in range(train_len)]
    val_data = [data[indices[i]] for i in range(train_len, train_len+val_len)]
    val_target = [target[indices[i]] for i in range(train_len, train_len+val_len)]
    return train_data, train_target, val_data, val_target

def accuracy(pred, target):
    y = np.argmax(pred.a, axis=1)
    t = np.argmax(target.a, axis=1)
    count = np.where(y == t, 1, 0)
    count = np.sum(count)
    return count/len(pred.a)

epoch = 500
batch_size = 32
lr = 1e-2
train_length = 0.8
path = "./weights/"


if __name__ == "__main__":
    import imgaug.augmenters as iaa
    augseq = iaa.Sequential([
        iaa.Crop(percent=(0, 0.03)),
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.1))),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-15, 15),
        shear=(-3, 3)
        ),
        ], random_order=True
    )
    model = Model()
    celoss = CrossEntropyLoss()
    optim = Adam(model.get_params(), lr=lr)

    train_data = np.load("./1_data/train_data.npy")
    train_label = np.load("./1_data/train_label.npy")
    train_x, train_y, val_x, val_y = random_split(train_data, train_label, train_length)
    train_dataset = Dataset(train_x, train_y, augseq)
    val_dataset = Dataset(val_x, val_y)
    dataloader = Dataloader(train_dataset, batch_size, True)
    val_dataloader = Dataloader(val_dataset)
    size = len(train_dataset)
    for e in range(epoch):
        progress = 0
        model.train()
        for x, target in dataloader:
            progress += batch_size
            optim.zero_grad()
            y = model(x)
            loss = celoss(y, target)
            loss.backward()
            optim.step()
            bar = int(progress/size*40)
            ac = accuracy(y, target)
            print("\r{}/{}[{}]loss: {} accuracy: {}".format(e,epoch,"="*bar+"-"*(40-bar),loss.sum(), ac), end="")
            del loss
        ac = 0
        loss = 0
        i = 0
        model.eval()
        for val_x, val_y in val_dataloader:
            i += 1
            y = model(val_x)
            ac += accuracy(y, val_y)
            loss += float(celoss(y, val_y).a)
        ac /= i+1e-7
        loss /= i+1e-7
        print(f"\nloss: {loss}, accuracy: {ac}")
        if (e+1)%10 == 0:
            with open(path+f"{e}_weights{ac}.pkl", "wb") as f:
                pickle.dump(model.state_dict(), f)