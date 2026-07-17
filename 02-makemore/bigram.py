"""
bigram.py — a character-level bigram language model, the counting version.

Stage 2 of building GPT from first principles, following Karpathy's
"makemore". A bigram model asks one narrow question: given the current
character, what comes next? You can answer it without any training at all —
just count how often each pair of characters appears, normalize the counts
into probabilities, and sample from them.

The token '.' marks BOTH the start and end of a name and lives at index 0,
exactly as in makemore. So "emma" becomes the bigrams
    (. e) (e m) (m m) (m a) (a .)
and the single row P[stoi['.']] is the distribution over first letters.

Pure Python, no dependencies. Import it, or run it:  python bigram.py
"""

import math
import random


def load_words(path="names.txt"):
    return open(path, "r").read().splitlines()


def build_vocab(words):
    """'.' at index 0, then every character seen, sorted. Returns stoi, itos."""
    chars = sorted(set("".join(words)))
    itos = ["."] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def _multinomial(probs, rng):
    """Sample one index from a probability row using inverse-CDF sampling."""
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r < acc:
            return i
    return len(probs) - 1  # float rounding fell through; take the last bucket


class BigramCounts:
    """P(next | current) learned by counting, with add-k smoothing.

    smoothing (add-k) matters for more than tidiness: an unseen pair like
    ('j','q') has count 0, and log(0) = -inf would blow up the NLL the moment
    such a pair shows up in held-out data. Adding k to every count guarantees
    every probability is strictly positive.
    """

    def __init__(self, words, smoothing=1.0):
        self.stoi, self.itos = build_vocab(words)
        n = len(self.itos)
        self.N = [[0] * n for _ in range(n)]  # raw bigram counts
        for w in words:
            chs = ["."] + list(w) + ["."]
            for a, b in zip(chs, chs[1:]):
                self.N[self.stoi[a]][self.stoi[b]] += 1
        self.smoothing = smoothing
        self._build_probs()

    def _build_probs(self):
        n = len(self.itos)
        self.P = []
        for row in self.N:
            total = sum(row) + self.smoothing * n
            self.P.append([(c + self.smoothing) / total for c in row])

    def sample(self, rng):
        """Walk the chain from '.' until we land back on '.'. Returns one name."""
        out = []
        ix = 0
        while True:
            ix = _multinomial(self.P[ix], rng)
            if ix == 0:  # hit the end token
                break
            out.append(self.itos[ix])
        return "".join(out)

    def nll(self, words):
        """Average negative log likelihood per bigram — lower is better.

        This is the quality number. A model that assigned probability 1 to
        every actual next character would score 0. Uniform random over 27
        tokens scores log(27) ~ 3.30. The counting model lands around 2.45.
        """
        log_likelihood = 0.0
        count = 0
        for w in words:
            chs = ["."] + list(w) + ["."]
            for a, b in zip(chs, chs[1:]):
                p = self.P[self.stoi[a]][self.stoi[b]]
                log_likelihood += math.log(p)
                count += 1
        return -log_likelihood / count

    def top_bigrams(self, k=10):
        """The k most common (a -> b) pairs, for a sanity check on the counts."""
        pairs = []
        for i, row in enumerate(self.N):
            for j, c in enumerate(row):
                if c:
                    pairs.append((c, self.itos[i], self.itos[j]))
        pairs.sort(reverse=True)
        return pairs[:k]


if __name__ == "__main__":
    words = load_words()
    model = BigramCounts(words)
    print(f"{len(words)} names, vocab size {len(model.itos)}")
    print("top bigrams:", [(f"{a}{b}", c) for c, a, b in model.top_bigrams(5)])
    print(f"average NLL: {model.nll(words):.4f}")
    rng = random.Random(2147483647)
    print("samples:", [model.sample(rng) for _ in range(5)])
