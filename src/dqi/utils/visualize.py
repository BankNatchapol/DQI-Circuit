import networkx as nx

import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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
    brute_force_results: List[Tuple[str, int]],
    dqi_results: Dict[str, int],
    plot_name: str = "Comparison of DQI and True Objective Values",
    spline_smoothing: float = 1.0
) -> None:
    """
    Dual‐axis Plotly chart: brute‐force objective vs. DQI probability.

    - Normalizes and pads DQI labels, handling keys with optional prefixes "xxxx yyyy".
    - Expects brute_force_results as [(bitstring, value), ...].
    - Expects dqi_results keys either "xxxx yyyy" (prefix suffix) or just "suffix".
    """
    if not brute_force_results:
        raise ValueError("brute_force_results must be non-empty")

    # determine full bit‐length from brute‐force labels
    full_len = len(brute_force_results[0][0])

    # normalize DQI keys: remove spaces, pad left with zeros if too short
    norm_dqi: Dict[str, int] = {}
    for key, count in dqi_results.items():
        parts = key.split()
        if len(parts) == 2:
            prefix, suffix = parts
        else:
            prefix, suffix = "", parts[0]
        bits = suffix
        # pad suffix to full length
        if len(bits) < full_len:
            bits = "0" * (full_len - len(bits)) + bits
        # combine prefix bits (if any) + suffix
        full_bits = (prefix + " " + bits) if prefix else bits
        # if prefix present, brute-force has no such full_bits -> add zero entry later
        norm_dqi[full_bits] = norm_dqi.get(full_bits, 0) + count
    
    prefix = ""
    if dqi_results:
        key_split = list(dqi_results.keys())[0].split(" ")
        if len(key_split) == 2:
            prefix = "0"*len(key_split[0]) 
    # build brute‐force dict
    bf_dict = {f"{prefix} {label}".strip(): val for label, val in brute_force_results}
    # ensure brute‐force covers all norm_dqi keys (fill zeros)
    for bits in norm_dqi:
        if bits not in bf_dict:
            bf_dict[bits] = 0

    # union of all labels
    all_keys = set(bf_dict) | set(norm_dqi)
    sorted_keys = sorted(all_keys, key=lambda k: int(k.replace(" ", ""), 2))

    # prepare series
    bf_values  = [bf_dict.get(k, 0) for k in sorted_keys]
    ext_counts = [norm_dqi.get(k, 0) for k in sorted_keys]
    total_ext  = sum(ext_counts)
    ext_probs  = [(c / total_ext) if total_ext else 0 for c in ext_counts]

    # set up y‐ranges and dticks
    bf_max = max(bf_values) if bf_values else 0
    bf_min = -abs(bf_max)
    dqi_max = round(max(ext_probs) * 1.25, 2) if ext_probs else 0

    bf_range  = [bf_min + bf_min/10, bf_max + bf_max/10]
    dqi_range = [-(dqi_max + dqi_max/10), dqi_max + dqi_max/10]

    left_dtick  = bf_max / bf_max if bf_max else 1
    right_dtick = round(dqi_max / bf_max, 3) if bf_max else 1

    # plot
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=sorted_keys, y=bf_values, mode="lines",
            name="Objective Value",
            line=dict(shape="spline", smoothing=spline_smoothing),
            fill="tozeroy", fillcolor="rgba(0,0,255,0.2)"
        ),
        secondary_y=False
    )

    if norm_dqi:
        fig.add_trace(
            go.Bar(
                x=sorted_keys, y=ext_probs,
                name="DQI (Probability)",
                marker=dict(opacity=0.6)
            ),
            secondary_y=True
        )

    fig.update_xaxes(title_text="Binary Value of Solution", tickangle=-45)
    fig.update_yaxes(
        title_text="Objective Value",
        range=bf_range,
        secondary_y=False,
        tick0=0, dtick=left_dtick, showgrid=True, gridcolor="lightblue"
    )
    if norm_dqi:
        
        dqi_pos = round(max(ext_probs)*1.35, 3)
        symmetric_range = [-dqi_pos, dqi_pos]

        fig.update_yaxes(
            title_text="Probability (DQI)",
            # center zero in the middle
            range=symmetric_range,
            secondary_y=True,
            # only show positive tick-labels
            tickmode="array",
            tickvals=[i*right_dtick for i in range(int(dqi_pos/right_dtick)+3)],
            ticktext=[f"{i*right_dtick:.2f}" for i in range(int(dqi_pos/right_dtick)+3)],
            showgrid=True,
            gridcolor="lightcoral",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="black",
        )

    fig.update_layout(
        title=plot_name,
        template="plotly_white",
        font_color="black",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )

    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": plot_name,
            "height": 500,
            "width": 1400,
            "scale": 10
        }
    }
    fig.show(config=config)