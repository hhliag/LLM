# Zero to Hero

My work following Andrej Karpathy's [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
series — building an autograd engine and then a GPT from scratch.

## Contents

- **`draft.ipynb`** — my follow-along notebook. Scratch work, derivative checks
  against finite differences, graph visualisations, and the training loop, in
  the order I worked through the video.
- **`micrograd/engine.py`** — my `Value` class, lifted from the notebook: a
  scalar autograd engine that records a DAG of operations and backprops through
  it in reverse topological order.
- **`micrograd/nn.py`** — my `Neuron`, `Layer` and `MLP` classes, built on top
  of `Value`.
- **`micrograd/test_gradients.py`** — builds the same small MLP in my engine and
  in PyTorch with identical weights, and checks that every parameter gradient
  matches.

```
python micrograd/test_gradients.py
```

## Karpathy's original repos

- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

Local clones of both live outside this repo (in `~/Desktop/karpathy-reference/`)
and are gitignored, so everything here is my own code.
