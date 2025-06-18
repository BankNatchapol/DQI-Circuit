import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def get_optimal_w(m: int, l: int, p: int, r: int) -> np.ndarray:
    """
    Build sparse tridiagonal A via scipy.sparse.diags and extract its principal eigenvector.
    Pads result up to next power of 2.
    """
    # build the tridiagonal entries
    d    = (p - 2*r) / np.sqrt(r * (p - r))
    diag = np.arange(l + 1) * d
    off  = np.sqrt(np.arange(1, l + 1) * (m - np.arange(1, l + 1) + 1))

    # <-- use a tuple here, not a list -->
    A = diags([off, diag, off], offsets=(-1, 0, 1), format="csr")

    # get principal eigenvector
    _, vecs = eigsh(A, k=1, which="LA")
    w = vecs.flatten()
    w /= np.linalg.norm(w)

    # pad up to next power of two
    orig   = w.size
    target = 1 << math.ceil(math.log2(orig))
    return np.pad(w, (0, target - orig))
