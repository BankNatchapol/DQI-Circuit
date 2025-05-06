import random
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

from .solver import max_xorsat_all_solutions

def generate_graph_custom(num_nodes: int, num_edges: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Generate a random undirected graph with a specified number of nodes and edges.

    Args:
        num_nodes: Number of nodes in the graph.
        num_edges: Exact number of edges to include.
        seed: Optional seed for reproducibility.

    Returns:
        A NetworkX graph with the specified configuration.

    Raises:
        ValueError: If the number of edges exceeds the maximum possible for the given nodes.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    # Generate all possible edges (undirected, no self-loops).
    possible_edges = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    
    if num_edges > len(possible_edges):
        raise ValueError(f"Requested {num_edges} edges, but maximum is {len(possible_edges)}.")
    
    chosen_edges = random.sample(possible_edges, num_edges)
    G.add_edges_from(chosen_edges)
    return G


def get_max_xorsat_matrix(G: nx.Graph, v: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the constraint matrix and right-hand side vector for the Max-XORSAT problem.

    For each edge (i, j) in G, the constraint is x_i XOR x_j = b, where b is provided in v
    or defaults to 1.

    Args:
        G: The graph representing the problem.
        v: Optional list of right-hand side values for each edge.

    Returns:
        A tuple (A, b) where A is the binary constraint matrix and b is the RHS vector.

    Raises:
        ValueError: If the length of v does not match the number of edges.
    """
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    A = np.zeros((num_edges, num_nodes), dtype=int)
    
    if v is None:
        b = np.ones(num_edges, dtype=int)
    else:
        b = np.array(v, dtype=int)
        if len(b) != num_edges:
            raise ValueError("Length of v must match the number of edges.")
    
    for idx, (i, j) in enumerate(G.edges()):
        A[idx, i] = 1
        A[idx, j] = 1
        
    return A, b

def find_graph_with_target_max_solutions(
    num_nodes: int,
    num_edges: int,
    target_max_solutions: Optional[int] = None,
    seed: Optional[int] = None,
    max_iter: Optional[int] = None
) -> Tuple[Optional[nx.Graph], Optional[List[Dict[int, int]]], Optional[int], int]:
    """
    Search for a graph whose number of optimal assignments matches a target value.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        target_max_solutions: Desired count of optimal assignments, or None to accept any.
        seed: Optional seed for reproducibility.
        max_iter: Maximum iterations for searching, or None for unlimited.

    Returns:
        A tuple containing:
          - The graph meeting the criteria (or None if not found before max_iter).
          - A list of optimal assignments (or None if not found).
          - The maximum number of constraints satisfied (or None if not found).
          - The number of iterations performed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    iterations = 0
    # Continue until max_iter (if given) or forever
    while True:
        G = generate_graph_custom(num_nodes, num_edges)
        assignments, max_sat = max_xorsat_all_solutions(G)

        # If target_max_solutions is None, accept the first graph
        if target_max_solutions is None or len(assignments) == target_max_solutions:
            return G, assignments, max_sat, iterations

        iterations += 1
        # If we've hit a max_iter limit, stop
        if max_iter is not None and iterations >= max_iter:
            break

    return None, None, None, iterations
