# myarray

pytorchに似せて作った機械学習ライブラリです．

# 機能

## Optimizer
- momentamSGD(params, lr, momentum)

| arg | 説明 |
| ---- | ---- |
| params | 更新するパラメータ |
| lr | 学習率 |
| momentum | モーメンタム |

- Adam(params, lr, betas, eps)

| arg | 説明 |
| ---- | ---- |
| params | 更新するパラメータ |
| lr | 学習率 |
| betas | 勾配の移動平均とその2乗を計算するための係数 |
| eps | 0除算を防ぐため |

## Layer
- Linear(in_features, out_features, bias=True)

| arg | 説明 |
| ---- | ---- |
| in_features | 入力数 |
| out_features | 出力数 |
| bias | バイアスの有無 |

- ReLu
- LeakeyReLu(negative_slope=0.01)

| arg | 説明 |
| ---- | ---- |
| negative_slope | 入力が負の時の傾き |

- Sigmoid
- Conv2d(未実装)

## Dataloader
- Dataloader(dataset, batch_size, shuffle)

| arg | 説明 |
| ---- | ---- |
| dataset | データセット |
| batch_size | バッチサイズ |
| shuffle | シャッフルの有無 |


## Module
- Module

    モデルを作る際に継承することで，パラメータのロードや保存が簡単になる．

| 関数 | 説明 |
| ---- | ---- |
| load_dict | 重みとバイアスをロード |
| state_dict | 重みとバイアスを出力 |
| eval | 評価モードに変更 |
| train | 学習モードに変更 |
| get_params | 重みとバイアスをリストで出力 |

## Loss
- MSELoss

    平均二乗誤差

- CrossEntropyLoss

    クロスエントロピー誤差

# 使い方
この例では，XOR(排他的論理和)を学習し，モデルを保存，モデルをロード，ロードしたモデルで予測する過程を説明する．
```
from mytorch.module import Module
from mytorch.layer import Linear, LeakeyReLu, Sigmoid
from mytorch.loss import MSELoss
from mytorch.optim import Adam
from mytorch.array import MyArray
import numpy as np
import pickle

if __name__ == "__main__":
    # モデルの構築
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

    # 損失関数の設定
    celoss = MSELoss()

    # 最適化の設定
    optim = Adam(params=model.get_params(), lr=1e-1)

    # 入力データと教師データ
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

    # トレーニング
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

    # モデルの保存   
    with open("state.pkl", "wb") as f:
        model.eval()
        pickle.dump(model.state_dict(), f)

    # テストデータ
    inputs = MyArray.from_array([
            [0, 1],
            [1, 1],
        ])
    
    # モデルのロードおよび予測
    with open("state.pkl", "rb") as f:
        model = Model()
        state = pickle.load(f)
        model.load_dict(state)
        pred = model(inputs)
        print(pred)
```