from nn import *
from autograd import *
import numpy as np

def loss(X, y, model, batch_size=None):


    ri = np.random.permutation(X.shape[0])[:batch_size]
    Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p*p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [((yi).__gt__(0)) == ((scorei.data).__gt__(0)) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


nn = MLP(3, [4, 4, 1])
print(nn)
print("number of parameters", len(nn.parameters()))

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets

for k in range(20):

    # forward
    total_loss, acc = loss(xs, ys, nn)

    # calculate loss (mean square error)
    nn.zero_grad()
    total_loss.backward()

    # backward (zero_grad + backward)
    ...

    # update
    learning_rate = 1.0 - 0.9*k/100
    for p in nn.parameters():
        p.data -= learning_rate * p.grad

    if k % 1 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")