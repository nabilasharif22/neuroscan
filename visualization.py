# ==========================================
# visualization.py (ROBUST VERSION)
# ==========================================

import plotly.graph_objects as go
from bug_checks import build_safe_graph, normalize_var


# ------------------------------------------
# 2. Extract + clean nodes
# ------------------------------------------
def extract_nodes(experiment):

    inputs = set()
    outputs = set()

    for var in experiment.get("manipulated_variables", []):
        normalized = normalize_var(var)
        if normalized:
            inputs.add(normalized)

    for var in experiment.get("measured_variables", []):
        normalized = normalize_var(var)
        if normalized:
            outputs.add(normalized)

    return inputs, outputs


# ------------------------------------------
# 3. Build graph safely
# ------------------------------------------
def build_graph(experiment):

    inputs, outputs = extract_nodes(experiment)

    links = experiment.get("model_links", [])

    edges = []
    all_nodes = set(inputs) | set(outputs)

    for link in links:
        exp_var = normalize_var(link.get("experiment_variable"))
        model_comp = normalize_var(link.get("model_component"))

        # Skip completely empty links
        if not exp_var and not model_comp:
            continue

        # Auto-add missing nodes
        if exp_var:
            all_nodes.add(exp_var)
        if model_comp:
            all_nodes.add(model_comp)

        edges.append({
            "source": exp_var,
            "target": model_comp,
            "type": link.get("relationship", "unknown"),
            "confidence": link.get("confidence", 0.5)
        })

    return inputs, outputs, all_nodes, edges


# ------------------------------------------
# 4. Layout (journal-style)
# ------------------------------------------
def compute_layout(inputs, outputs, all_nodes):

    pos = {}

    inputs = sorted(inputs)
    outputs = sorted(outputs)
    middle = sorted(all_nodes - set(inputs) - set(outputs))

    # Left column (inputs)
    for i, node in enumerate(inputs):
        pos[node] = (-1, i)

    # Middle column (latent / model)
    for i, node in enumerate(middle):
        pos[node] = (0, i)

    # Right column (outputs)
    for i, node in enumerate(outputs):
        pos[node] = (1, i)

    return pos


# ------------------------------------------
# 5. Draw edges safely
# ------------------------------------------
def make_edges(edges, pos):

    edge_x = []
    edge_y = []
    for edge in edges:

        src = edge["source"]
        tgt = edge["target"]

        if src not in pos or tgt not in pos:
            # Skip invalid edges (no crash)
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    return edge_x, edge_y


# ------------------------------------------
# 6. Draw nodes
# ------------------------------------------
def make_nodes(pos, inputs, outputs, selected_node=None):

    x = []
    y = []
    text = []
    color = []
    size = []

    for node, (px, py) in pos.items():

        x.append(px)
        y.append(py)
        text.append(node)

        if node in inputs:
            base_color = "royalblue"
        elif node in outputs:
            base_color = "darkorange"
        else:
            base_color = "seagreen"

        if selected_node and node == selected_node:
            color.append("crimson")
            size.append(24)
        else:
            color.append(base_color)
            size.append(18)

    return x, y, text, color, size


# ------------------------------------------
# 7. MAIN FUNCTION
# ------------------------------------------
def draw_experiment_diagram(experiment, rel_filter=None, selected_node=None):

    graph = build_safe_graph(experiment, rel_filter=rel_filter)
    inputs = graph["inputs"]
    outputs = graph["outputs"]
    all_nodes = graph["all_nodes"]
    edges = graph["edges"]

    pos = compute_layout(inputs, outputs, all_nodes)

    edge_x, edge_y = make_edges(edges, pos)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.8, color="gray"),
        hoverinfo="none",
        mode="lines",
    )

    normalized_selected = normalize_var(selected_node)
    node_x, node_y, node_text, node_color, node_size = make_nodes(
        pos,
        inputs,
        outputs,
        selected_node=normalized_selected,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='black')
        )
    )

    # Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Experiment ↔ Model Mapping",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white"
        )
    )

    return fig