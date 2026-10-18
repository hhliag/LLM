"""Check my autograd engine against PyTorch.

Builds one MLP with my Value/Neuron/Layer/MLP code, mirrors it in PyTorch with
the exact same weights and the same forward pass, backprops the same loss
through both, and compares every parameter gradient.
"""

import random

import torch

from nn import MLP

XS = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
YS = [1.0, -1.0, -1.0, 1.0]


def torch_mirror(model):
    """One torch scalar per parameter of `model`, same values, same layout."""
    params = []
    for layer in model.layers:
        layer_params = []
        for neuron in layer.neurons:
            w = [torch.tensor(float(wi.data), dtype=torch.double, requires_grad=True)
                 for wi in neuron.w]
            b = torch.tensor(float(neuron.b.data), dtype=torch.double, requires_grad=True)
            layer_params.append((w, b))
        params.append(layer_params)
    return params


def torch_forward(params, x):
    """Same forward pass as Neuron/Layer/MLP: tanh(w . x + b), layer by layer."""
    act = [torch.tensor(float(xi), dtype=torch.double) for xi in x]
    for layer_params in params:
        outs = []
        for w, b in layer_params:
            total = b
            for wi, xi in zip(w, act):
                total = total + wi * xi
            outs.append(torch.tanh(total))
        act = outs
    return act[0] if len(act) == 1 else act


def flatten(params):
    return [p for layer in params for w, b in layer for p in w + [b]]


def test_gradients_match():
    random.seed(1337)
    model = MLP(3, [4, 4, 1])
    params = torch_mirror(model)

    # my engine
    mine_pred = [model(x) for x in XS]
    mine_loss = sum([(pred - y) ** 2 for y, pred in zip(YS, mine_pred)])
    mine_loss.backward()

    # pytorch
    torch_pred = [torch_forward(params, x) for x in XS]
    torch_loss = sum([(pred - y) ** 2 for y, pred in zip(YS, torch_pred)])
    torch_loss.backward()

    assert abs(mine_loss.data - torch_loss.item()) < 1e-9, (
        f"loss differs: mine={mine_loss.data} torch={torch_loss.item()}"
    )

    mine_grads = [p.grad for p in model.parameters()]
    torch_grads = [p.grad.item() for p in flatten(params)]
    assert len(mine_grads) == len(torch_grads)

    worst = 0.0
    for i, (a, b) in enumerate(zip(mine_grads, torch_grads)):
        worst = max(worst, abs(a - b))
        assert abs(a - b) < 1e-9, f"param {i}: mine={a} torch={b}"

    return mine_loss.data, torch_loss.item(), len(mine_grads), worst


if __name__ == "__main__":
    loss_mine, loss_torch, n, worst = test_gradients_match()
    print(f"loss  mine={loss_mine:.12f}  torch={loss_torch:.12f}")
    print(f"compared {n} parameter gradients")
    print(f"max abs gradient difference: {worst:.3e}")
    print("OK: gradients match")
