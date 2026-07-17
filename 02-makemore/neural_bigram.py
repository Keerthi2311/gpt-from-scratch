"""
neural_bigram.py — the same bigram model, but LEARNED by gradient descent.

The counting model in bigram.py never trains: it reads probabilities straight
off the tallies. This file solves the identical problem the other way, as a
one-layer neural net, and the punchline of makemore part 1 is that the two
converge to the same distribution.

Forward pass, for a batch of current-characters x and next-characters y:

    onehot(x) @ W  ->  logits  ->  softmax  ->  probs
    loss = mean negative log likelihood of the true next char + L2 on W

W is 27x27. Row i of W, after softmax, plays the role of P[i] from the
counting model — but reached by minimizing loss instead of by counting.

The gradient is the one derived by hand in stage 01. There we showed that for
softmax followed by negative log likelihood,
    d(loss)/d(logit_j) = p_j - 1{j == target}
i.e. the predicted probabilities with 1 subtracted off the true class. That is
exactly `dlogits` below — the whole reason stage 01 built exp and log.

Needs numpy (the only dependency in this stage). Run:  python neural_bigram.py
"""

import numpy as np

from bigram import build_vocab, load_words


def build_dataset(words, stoi):
    """Flatten every name into (current, next) integer pairs."""
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            xs.append(stoi[a])
            ys.append(stoi[b])
    return np.array(xs), np.array(ys)


def train(words, steps=200, lr=50.0, reg=0.01, seed=2147483647, verbose=False):
    stoi, itos = build_vocab(words)
    n = len(itos)
    xs, ys = build_dataset(words, stoi)
    num = xs.shape[0]

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))

    # one-hot the inputs once; xenc @ W just selects rows of W, but writing it
    # as a matmul keeps the gradient dW = xenc.T @ dlogits obvious.
    xenc = np.zeros((num, n))
    xenc[np.arange(num), xs] = 1.0

    loss = None
    for k in range(steps):
        # forward: logits -> softmax -> probs
        logits = xenc @ W
        counts = np.exp(logits)
        probs = counts / counts.sum(axis=1, keepdims=True)

        # loss: mean NLL of the true next char, plus L2 regularization on W
        # (regularization is the gradient-descent equivalent of add-k
        # smoothing — it pulls W toward 0, i.e. probs toward uniform).
        data_loss = -np.log(probs[np.arange(num), ys]).mean()
        loss = data_loss + reg * (W**2).mean()

        # backward: dlogits = probs - onehot(y), the stage-01 result
        dlogits = probs.copy()
        dlogits[np.arange(num), ys] -= 1
        dlogits /= num
        dW = xenc.T @ dlogits + 2 * reg * W / W.size

        W -= lr * dW  # gradient descent step

        if verbose and (k % 20 == 0 or k == steps - 1):
            print(f"step {k:3d} | loss {loss:.4f}")

    return W, stoi, itos, float(loss)


def probs_from_W(W):
    """Softmax each row so W can be compared against the counting model's P."""
    counts = np.exp(W)
    return counts / counts.sum(axis=1, keepdims=True)


def sample(W, itos, rng):
    P = probs_from_W(W)
    out = []
    ix = 0
    while True:
        ix = rng.choice(len(itos), p=P[ix])
        if ix == 0:
            break
        out.append(itos[ix])
    return "".join(out)


if __name__ == "__main__":
    words = load_words()
    W, stoi, itos, loss = train(words, steps=200, verbose=True)
    print(f"\nfinal training loss: {loss:.4f}")
    rng = np.random.default_rng(2147483647)
    print("samples:", [sample(W, itos, rng) for _ in range(5)])
