from itertools import product
from typing import Callable, List, Tuple, Dict, Any
import numpy as np
import networkx as nx

def _enumerate_solutions(
    num_vars: int,
    score_fn: Callable[[Tuple[int, ...]], int]
) -> List[Tuple[Tuple[int, ...], int]]:
    """
    Generic brute-force over all 2^num_vars assignments.
    Returns a list of (bit_tuple, score) pairs, sorted descending by score.
    """
    results = []
    for bits in product([0,1], repeat=num_vars):
        results.append((bits, score_fn(bits)))
    # sort so best scores come first
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def max_xorsat_all_solutions(G: nx.Graph) -> Tuple[List[Dict[int,int]], int]:
    """
    Wrapper that scores by counting satisfied XOR edges.
    """
    def score(bits: Tuple[int,...]) -> int:
        return sum((bits[i] ^ bits[j]) == 1 for i, j in G.edges())

    raw = _enumerate_solutions(G.number_of_nodes(), score)
    best_score = raw[0][1]
    # collect all assignments tying that best_score
    best = [bits for bits, s in raw if s == best_score]
    return ([{i: bits[i] for i in range(len(bits))} for bits in best], best_score)


def brute_force_max(B: np.ndarray, v: np.ndarray) -> List[Tuple[str,int]]:
    """
    Wrapper that scores by the (-1)^(B·x + v) sum metric.
    """
    m, n = B.shape

    def score(bits: Tuple[int,...]) -> int:
        x = np.array(bits, dtype=int)
        return int(np.sum((-1)**(B.dot(x)+v)))

    raw = _enumerate_solutions(n, score)
    # return sorted ascending by bitstring integer (you can re-sort here if you like)
    # or leave descending by score—choose whichever API you prefer
    return [("".join(map(str,bits)), sc) for bits, sc in raw]
