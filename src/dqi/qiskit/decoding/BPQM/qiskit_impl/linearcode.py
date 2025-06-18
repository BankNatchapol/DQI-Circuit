"""Utilities for binary linear block codes and their factor graphs."""
import re
import numpy as np
import networkx as nx
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union, Set
from numpy.typing import NDArray

__all__ = ["LinearCode"]

# Patterns for labeling
_LABEL_PATTERN = re.compile(r"^([a-zA-Z]+)(\d+)$")
_SUBSCRIPT_MAP = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _latex_label(name: str) -> str:
    """
    Convert a node name (e.g., 'x0') into a LaTeX-formatted label (e.g., '$x_{0}$').
    """
    match = _LABEL_PATTERN.match(name)
    if not match:
        return name
    var, idx = match.groups()
    return rf"${var}_{{{idx}}}$"


def _unicode_label(name: str) -> str:
    """
    Convert a node name (e.g., 'x0') into a unicode-subscript label (e.g., 'x₀').
    This is used for backends that do not support LaTeX rendering.
    """
    match = _LABEL_PATTERN.match(name)
    if not match:
        return name
    var, idx = match.groups()
    return var + idx.translate(_SUBSCRIPT_MAP)


class LinearCode:
    """
    Binary linear block code with enumeration, factor-graph construction,
    Matplotlib and PyVis visualizations, and BPQM unfolding.

    Parameters
    ----------
    G : Optional[NDArray]
        Generator matrix of shape (k, n). If None, will infer k from H.
    H : NDArray
        Parity-check matrix of shape (n - k, n).
    """

    def __init__(self, G: Optional[NDArray], H: NDArray):
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
        """
        Enumerate all codewords by recursively combining rows of G.

        Returns
        -------
        List[NDArray]
            List of length 2^k, each entry is a codeword of length n.

        Raises
        ------
        ValueError
            If G was not provided at initialization.
        """
        if self.G is None:
            raise ValueError("Generator matrix required to enumerate codewords")

        def _recurse(i: int) -> List[NDArray]:
            assert self.G, "Generator matrix required to enumerate codewords"
            if i == self.k - 1:
                return [np.zeros(self.n, dtype=int), self.G[i]]
            prev = _recurse(i + 1)
            return prev + [(self.G[i] + cw) % 2 for cw in prev]

        return _recurse(0)

    def get_factor_graph(self) -> nx.Graph:
        """
        Build the bipartite factor graph: variable nodes x_i,
        check nodes c_j, and output nodes y_i.

        Returns
        -------
        nx.Graph
            Undirected factor graph with node attribute 'type'.
        """
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

    def plot_factor_graph(
        self,
        backend: str = "matplotlib",
        latex_subscripts: bool = True,
        **kwargs
    ) -> None:
        """
        Render the factor graph using Matplotlib or PyVis.

        Parameters
        ----------
        backend : str
            'matplotlib' for static plots or 'pyvis' for interactive HTML.
        latex_subscripts : bool
            If True, use LaTeX labels in Matplotlib and unicode subscripts in PyVis
            (PyVis cannot render LaTeX directly).
        **kwargs
            Additional backend-specific parameters.
        """
        if backend == "matplotlib":
            self._plot_matplotlib(latex_subscripts=latex_subscripts, **kwargs)
        elif backend == "pyvis":
            self._plot_pyvis(latex_subscripts=latex_subscripts, **kwargs)
        else:
            raise ValueError("backend must be 'matplotlib' or 'pyvis'")

    def _plot_matplotlib(
        self,
        layout: str = "kamada_kawai",
        figsize: Tuple[int, int] = (6, 6),
        variable_color: str = "skyblue",
        check_color: str = "salmon",
        output_color: str = "lightgray",
        variable_size: int = 600,
        check_size: Optional[int] = None,
        output_size: Optional[int] = None,
        latex_subscripts: bool = True,
        title: Optional[str] = None,
        font_size: int = 12,
    ) -> None:
        """
        Internal method to draw factor graph with Matplotlib.
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        graph = self.get_factor_graph()
        pos_fn = {
            "kamada_kawai": nx.kamada_kawai_layout,
            "spring": nx.spring_layout,
            "circular": nx.circular_layout,
        }.get(layout, nx.spring_layout)
        pos = pos_fn(graph)

        # Prepare node groups
        vars_ = [n for n, d in graph.nodes(data=True) if d["type"] == "variable"]
        checks = [n for n, d in graph.nodes(data=True) if d["type"] == "check"]
        outs = [n for n, d in graph.nodes(data=True) if d["type"] == "output"]

        # Disable LaTeX engine in Matplotlib
        mpl.rcParams["text.usetex"] = False

        fig, ax = plt.subplots(figsize=figsize)
        # Draw nodes and edges
        nx.draw_networkx_nodes(graph, pos, vars_, node_color=variable_color,
                               node_shape="o", node_size=variable_size, ax=ax)
        nx.draw_networkx_nodes(graph, pos, checks, node_color=check_color,
                               node_shape="s", node_size=check_size or int(variable_size*1.2), ax=ax)
        nx.draw_networkx_nodes(graph, pos, outs, node_color=output_color,
                               node_shape="o", node_size=output_size or int(variable_size*0.8), ax=ax)
        nx.draw_networkx_edges(graph, pos, ax=ax)

        # Labels
        labels = {
            node: (_latex_label(node) if latex_subscripts else node)
            for node in graph.nodes()
        }
        nx.draw_networkx_labels(graph, pos, labels, font_size=font_size, ax=ax)

        if title:
            ax.set_title(title)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def _plot_pyvis(
        self,
        html_file: str = "factor_graph.html",
        width: str = "900px",
        height: str = "700px",
        cdn_resources: str = "remote",
        variable_color: str = "skyblue",
        check_color: str = "salmon",
        output_color: str = "lightgray",
        variable_size: int = 600,
        check_size: Optional[int] = None,
        output_size: Optional[int] = None,
        font_size: int = 12,
        latex_subscripts: bool = True,
    ) -> None:
        """
        Internal method to draw factor graph with PyVis.

        Note
        ----
        PyVis cannot render LaTeX, so unicode_subscript labels are used when
        `latex_subscripts=True`.
        """
        try:
            from pyvis.network import Network
        except ImportError as exc:
            raise ImportError("Install pyvis to use this backend: pip install pyvis") from exc

        graph = self.get_factor_graph()
        net = Network(width=width, height=height, cdn_resources=cdn_resources)
        net.from_nx(graph)

        # Compute positions once
        pos = nx.kamada_kawai_layout(graph)
        scale = 2000
        for node, (x, y) in pos.items():
            net.get_node(node).update({"x": x*scale, "y": y*scale, "physics": True})

        # Update styling and labels
        for node, data in graph.nodes(data=True):
            node_data = net.get_node(node)
            color, size = {
                "variable": (variable_color, variable_size),
                "check": (check_color, check_size or int(variable_size*1.2)),
                "output": (output_color, output_size or int(variable_size*0.8)),
            }[data["type"]]
            node_data.update({
                "color": color,
                "shape": "square" if data["type"] == "check" else "dot",
                "size": size / 10,
                "font": {"size": font_size},
                "label": (_unicode_label(node) if latex_subscripts else node),
            })

        net.save_graph(html_file)
    
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

        Returns
        -------
        Tuple[nx.DiGraph, Dict[str, int], str]
            The unrolled computation graph, a dictionary with variable occurrence counts,
            and the new root node label.
        """
        fg = self.get_factor_graph()
        directed = nx.DiGraph()
        var_occ = {v: 0 for v in fg.nodes if fg.nodes[v]["type"] == "variable"}
        check_occ = {c: 0 for c in fg.nodes() if fg.nodes[c]["type"]=="check"}
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
                if child!="None":
                    directed.add_edge(label, child)

            if ntype == "variable":
                out_label = label.replace("x", "y")
                directed.add_node(out_label, type="output")
                directed.add_edge(label, out_label)
            

            return label

        new_root = _expand(root, None, 0)
        return directed, var_occ, new_root
