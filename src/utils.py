#!/usr/bin/env python3

import time
from functools import wraps
from itertools import product
import random
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configure logging for the module.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def time_logger(func):
    """
    Decorator to log the execution time of a function.

    Usage:
        @time_logger
        def my_function(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed:.6f} seconds.")
        return result, elapsed 
    return wrapper


def combine_counts(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Consolidate counts by combining keys that share the same second part.

    Each key should be in the format 'first_part second_part'. The function returns a 
    dictionary with keys as the second part and values as the summed counts.

    Args:
        counts: Dictionary with keys in 'first_part second_part' format.

    Returns:
        A dictionary mapping the second part to the summed counts.
    """
    new_counts: Dict[str, int] = {}
    for key, value in counts.items():
        try:
            _, second_part = key.split()
        except ValueError:
            logger.error(f"Key '{key}' does not match the expected format.")
            continue
        new_counts[second_part] = new_counts.get(second_part, 0) + value
    return new_counts


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


def max_xorsat_all_solutions(G: nx.Graph) -> Tuple[List[Dict[int, int]], int]:
    """
    Exhaustively search for assignments that maximize the number of satisfied XOR constraints.

    For each edge (i, j), the constraint is satisfied if x_i XOR x_j equals 1.

    Args:
        G: The graph representing the Max-XORSAT problem.

    Returns:
        A tuple containing:
          - A list of dictionaries (node: value) representing optimal assignments.
          - The maximum number of constraints satisfied.
    """
    num_nodes = G.number_of_nodes()
    best_assignments: List[Tuple[int, ...]] = []
    max_satisfied = -1

    for bits in product([0, 1], repeat=num_nodes):
        count = sum(1 for (i, j) in G.edges() if (bits[i] ^ bits[j]) == 1)
        if count > max_satisfied:
            max_satisfied = count
            best_assignments = [bits]
        elif count == max_satisfied:
            best_assignments.append(bits)
    
    assignments_dict = [{i: bits[i] for i in range(num_nodes)} for bits in best_assignments]
    return assignments_dict, max_satisfied


def find_graph_with_target_max_solutions(
    num_nodes: int, 
    num_edges: int, 
    target_max_solutions: int, 
    seed: Optional[int] = None, 
    max_iter: int = 1000
) -> Tuple[Optional[nx.Graph], Optional[List[Dict[int, int]]], Optional[int], int]:
    """
    Search for a graph whose number of optimal assignments matches a target value.

    Args:
        num_nodes: Number of nodes.
        num_edges: Number of edges.
        target_max_solutions: Desired count of optimal assignments.
        seed: Optional seed for reproducibility.
        max_iter: Maximum iterations for searching.

    Returns:
        A tuple containing:
          - The graph meeting the criteria (or None if not found).
          - A list of optimal assignments (or None if not found).
          - The maximum number of constraints satisfied (or None if not found).
          - The number of iterations performed.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    iterations = 0
    while iterations < max_iter:
        G = generate_graph_custom(num_nodes, num_edges)
        assignments, max_sat = max_xorsat_all_solutions(G)
        if len(assignments) == target_max_solutions:
            return G, assignments, max_sat, iterations
        iterations += 1
        
    return None, None, None, iterations


@dataclass
class BruteForceResult:
    """
    Data structure to hold results from a brute-force evaluation.

    Attributes:
        full_binary_value: Integer representation of the full binary solution.
        full_label: String label in the format 'first_part second_part'.
        value: Objective value computed for the assignment.
    """
    full_binary_value: int
    full_label: str
    value: int


def brute_force_max(B: np.ndarray, v: np.ndarray, first_part: str = "000000") -> List[BruteForceResult]:
    """
    Evaluate the objective value for all assignments of the second part of a binary solution.

    Each assignment's result is stored in a BruteForceResult. The final list is sorted by the 
    integer value of the full binary string (first_part concatenated with the evaluated second part).

    Args:
        B: Binary matrix representing the constraints.
        v: Right-hand side vector.
        first_part: Fixed binary string for the first part of the solution.

    Returns:
        A sorted list of BruteForceResult objects.
    """
    num_vars = B.shape[1]
    results: List[BruteForceResult] = []
    
    for bits in product([0, 1], repeat=num_vars):
        x = np.array([bits])
        value = int(sum((-1) ** ((B @ x.T).T[0] + v)))
        second_part = ''.join(map(str, bits))
        full_label = f"{first_part} {second_part}".strip()
        full_binary_value = int(first_part + second_part, 2)
        results.append(BruteForceResult(full_binary_value, full_label, value))
    
    results.sort(key=lambda res: res.full_binary_value)
    return results


def draw_graph(G: nx.Graph, assignment: Optional[Dict[int, int]] = None) -> None:
    """
    Visualize a graph with nodes colored based on a provided binary assignment.

    If an assignment is provided, nodes with a value of 0 are colored pink and 1 are light blue.

    Args:
        G: The graph to visualize.
        assignment: Optional mapping of node to binary value.
    """
    pos = nx.spring_layout(G)
    if assignment:
        colors = ['pink' if assignment.get(node, 0) == 0 else 'lightblue' for node in G.nodes()]
        title = "Graph with Max-XORSAT Solution"
    else:
        colors = 'lightblue'
        title = "Graph"
    
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, edge_color='gray',
            node_size=700, font_size=12)
    plt.title(title)
    plt.show()


def plot_results_union_plotly(
    brute_force_results: List[BruteForceResult], 
    dqi_results: Dict[str, int], 
    plot_name: str = "Comparison of DQI and True Objective Values", 
    spline_smoothing: float = 1.0
) -> None:
    """
    Create a dual-axis Plotly chart comparing brute-force objective values with DQI probabilities.

    The left y-axis shows the smoothed objective values, while the right y-axis displays
    normalized DQI probabilities as a bar chart.

    Args:
        brute_force_results: List of brute-force evaluation results.
        dqi_results: Dictionary mapping solution labels to counts.
        plot_name: Title of the plot.
        spline_smoothing: Smoothing parameter for the objective value curve.
    """
    bf_dict: Dict[str, int] = {res.full_label: res.value for res in brute_force_results}
    union_keys = set(bf_dict.keys()).union(set(dqi_results.keys()))
    sorted_keys = sorted(union_keys, key=lambda k: int(k.replace(" ", ""), 2))
    
    bf_values = [bf_dict.get(k, 0) for k in sorted_keys]
    ext_counts = [dqi_results.get(k, 0) for k in sorted_keys]
    total_ext = sum(ext_counts)
    ext_probs = [count / total_ext if total_ext > 0 else 0 for count in ext_counts]
    
    max_prob = max(ext_probs) if ext_probs else 0
    dqi_y_max = round(max_prob * 1.25, 2)
    bf_y_max = max(bf_values)
    bf_y_min = -abs(bf_y_max)
    dqi_y_range = [-(dqi_y_max + dqi_y_max / 10), dqi_y_max + dqi_y_max / 10]
    bf_y_range = [bf_y_min + bf_y_min / 10, bf_y_max + bf_y_max / 10]
    
    # dtick values for consistent grid spacing.
    left_dtick = bf_y_max / bf_y_max if bf_y_max > 0 else 1
    right_dtick = round(dqi_y_max / bf_y_max if dqi_y_max > 0 else 1, 3)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=sorted_keys,
            y=bf_values,
            mode='lines',
            name='Objective Value',
            line=dict(shape='spline', smoothing=spline_smoothing),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ),
        secondary_y=False
    )
    
    if dqi_results:
        fig.add_trace(
            go.Bar(
                x=sorted_keys,
                y=ext_probs,
                name='DQI (Probability)',
                marker=dict(opacity=0.6)
            ),
            secondary_y=True
        )
    
    fig.update_xaxes(title_text="Binary Value of Solution", tickangle=-45)
    fig.update_yaxes(
        title_text="Objective Value", 
        range=bf_y_range,
        secondary_y=False, 
        tick0=0, 
        dtick=left_dtick, 
        showgrid=True, 
        gridcolor='lightblue'
    )
    
    if dqi_results:
        fig.update_yaxes(
            title_text="Probability (DQI)", 
            range=dqi_y_range, 
            secondary_y=True, 
            tick0=0, 
            dtick=right_dtick, 
            showgrid=True, 
            gridcolor='lightcoral'
        )
        
    fig.update_layout(
        title=plot_name,
        template="plotly_white",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': plot_name,
            'height': 500,
            'width': 1000,
            'scale': 6
        }
    }
    fig.show(config=config)
