"""
Cirq version of BPQM linear code utilities.
"""

# TODO: Implement Cirq-based BPQM linear code utilities here

import cirq
import re
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union, Set
from numpy.typing import NDArray

__all__ = ["CirqLinearCode"]

# Patterns for labeling
_LABEL_PATTERN = re.compile(r"^([a-zA-Z]+)(\d+)$")
_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _latex_label(name: str) -> str:
    """Convert a node name (e.g., 'x0') into a LaTeX-formatted label (e.g., '$x_{0}$')."""
    match = _LABEL_PATTERN.match(name)
    if not match:
        return name
    var, idx = match.groups()
    return rf"${var}_{{{idx}}}$"


def _unicode_label(name: str) -> str:
    """Convert a node name (e.g., 'x0') into a unicode-subscript label (e.g., 'x₀')."""
    match = _LABEL_PATTERN.match(name)
    if not match:
        return name
    var, idx = match.groups()
    return var + idx.translate(_SUBSCRIPT_MAP)


class CirqLinearCode:
    """
    Binary linear block code with enumeration, factor-graph construction,
    and visualization - Cirq version.

    Parameters
    ----------
    G : Optional[NDArray]
        Generator matrix of shape (k, n). If None, will infer k from H.
    H : NDArray
        Parity-check matrix of shape (n - k, n).
    """

    def __init__(self, G: Optional[NDArray] = None, H: Optional[NDArray] = None):
        if H is None:
            raise ValueError("Parity-check matrix H is required")
        self.H = np.asarray(H, dtype=int)
        self.n = self.H.shape[1]
        self.hk = H.shape[0]
        if G is not None:
            self.G = np.asarray(G, dtype=int)
            if self.G.shape[1] != self.n:
                raise ValueError("Generator matrix column count must match H")
            self.k = self.G.shape[0]
        else:
            self.G = None
            self.k = self.n - self.H.shape[0]

    def get_codewords(self) -> List[NDArray[np.int_]]:
        """Enumerate all codewords by recursively combining rows of G."""
        if self.G is None:
            raise ValueError("Generator matrix required to enumerate codewords")

        def _recurse(i: int) -> List[NDArray]:
            assert self.G is not None, "Generator matrix required to enumerate codewords"
            if i == self.k - 1:
                return [np.zeros(self.n, dtype=int), self.G[i]]
            prev = _recurse(i + 1)
            return prev + [(self.G[i] + cw) % 2 for cw in prev]

        return _recurse(0)

    def get_factor_graph(self) -> nx.Graph:
        """Build the bipartite factor graph: variable nodes x_i, check nodes c_j, and output nodes y_i."""
        graph = nx.Graph()
        # Add variable and output nodes
        for i in range(self.n):
            graph.add_node(f"x{i}", type="variable")
            graph.add_node(f"y{i}", type="output")
            graph.add_edge(f"x{i}", f"y{i}")
        # Add check nodes
        for j in range(self.H.shape[0]):
            graph.add_node(f"c{j}", type="check")
            for i in range(self.n):
                if self.H[j, i] != 0:
                    graph.add_edge(f"c{j}", f"x{i}")
        return graph

    def get_computation_graph(
        self,
        root: str,
        height: int,
        cloner: Optional[Any] = None,
        syndrome_mode: bool = False
    ) -> Tuple[nx.DiGraph, Dict[str, int], str]:
        """
        Unroll the factor graph for message passing.

        Parameters
        ----------
        root : str
            Name of the variable node to serve as root (e.g., `"x0"`).
        height : int
            Number of layers to expand on each side of the root.
        cloner : Optional[Any]
            Reserved for future cloning logic.
        syndrome_mode : bool
            If True, use syndrome-specific check node labeling.

        Returns
        -------
        Tuple[nx.DiGraph, Dict[str, int], str]
            The unrolled computation graph, a dictionary with variable occurrence counts,
            and the new root node label.
        """
        fg = self.get_factor_graph()
        directed = nx.DiGraph()
        var_occ = {v: 0 for v in fg.nodes if fg.nodes[v]["type"] == "variable"}
        check_occ = {c: 0 for c in fg.nodes() if fg.nodes[c]["type"] == "check"}
        check_counter = 0
        max_depth = 2 * height + 1

        def _expand(node: str, parent: Optional[str], depth: int) -> str:
            nonlocal check_counter
            if depth >= max_depth or fg.nodes[node]["type"] == "output":
                return "None"

            ntype = fg.nodes[node]["type"]
            if ntype == "variable":
                label = f"{node}_{var_occ[node]}"
                var_occ[node] += 1
                directed.add_node(label, type=ntype)
            else:
                if syndrome_mode:
                    occ = check_occ[node]
                    label = f"{node}_{occ}"
                    check_occ[node] += 1
                    directed.add_node(label,
                                      type=ntype,
                                      check_idx=int(node.lstrip("c")))
                else:
                    label = f"c{check_counter}"
                    check_counter += 1
                    directed.add_node(label, type=ntype)
            
            for neighbor in fg.neighbors(node):
                if neighbor == parent:
                    continue
                child = _expand(neighbor, node, depth + 1)
                if child != "None":
                    directed.add_edge(label, child)

            if ntype == "variable":
                out_label = label.replace("x", "y")
                directed.add_node(out_label, type="output")
                directed.add_edge(label, out_label)

            return label

        new_root = _expand(root, None, 0)
        return directed, var_occ, new_root

    # Add Cirq linear code methods here 