# ==========================================
# visualization.py (ROBUST VERSION)
# ==========================================

import plotly.graph_objects as go
from bug_checks import build_safe_graph, normalize_var


RELATIONSHIP_COLORS = {
    "tests": "#2563eb",
    "correlates": "#7c3aed",
    "controls": "#0f766e",
    "modulates": "#d97706",
    "causes": "#dc2626",
    "unknown": "#6b7280",
}


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


def _centered_positions(nodes, x_value):

    positions = {}
    ordered_nodes = sorted(nodes)
    count = len(ordered_nodes)

    if count == 0:
        return positions

    spacing = 1.8
    midpoint = ((count - 1) / 2) * spacing
    for index, node in enumerate(ordered_nodes):
        positions[node] = (x_value, midpoint - index * spacing)

    return positions


def compute_semantic_layout(inputs, model_nodes, outputs, all_nodes):

    pos = {}
    pos.update(_centered_positions(inputs, -1.0))
    pos.update(_centered_positions(model_nodes, 0.0))

    remaining_outputs = sorted(set(outputs) - set(model_nodes))
    pos.update(_centered_positions(remaining_outputs, 1.0))

    unassigned = sorted(set(all_nodes) - set(pos))
    if unassigned:
        pos.update(_centered_positions(unassigned, 0.5))

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


def make_node_hover_text(pos, inputs, outputs, model_nodes):

    hover_text = []

    for node in pos:
        if node in inputs:
            role = "manipulated variable"
        elif node in outputs:
            role = "measured variable"
        elif node in model_nodes:
            role = "model component"
        else:
            role = "derived node"

        hover_text.append(f"{node}<br>Role: {role}")

    return hover_text


def make_edge_traces(edges, pos):

    edge_traces = []
    label_x = []
    label_y = []
    label_text = []
    annotations = []

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]

        if src not in pos or tgt not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        relationship = edge.get("type", "unknown")
        confidence = edge.get("confidence", 0.5)
        edge_kind = edge.get("kind", "input_to_model")
        color = RELATIONSHIP_COLORS.get(relationship, RELATIONSHIP_COLORS["unknown"])
        width = 1.5 + (3.0 * confidence)
        line_style = "dot" if edge_kind == "model_to_output" else "solid"
        direction_label = "model → outcome" if edge_kind == "model_to_output" else "input → model"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color, dash=line_style),
                opacity=0.35 + (0.6 * confidence),
                hoverinfo="text",
                text=[
                    f"{src} → {tgt}<br>Path: {direction_label}<br>Relationship: {relationship}<br>Confidence: {confidence:.2f}",
                    f"{src} → {tgt}<br>Path: {direction_label}<br>Relationship: {relationship}<br>Confidence: {confidence:.2f}",
                ],
                showlegend=False,
            )
        )

        label_x.append((x0 + x1) / 2)
        label_y.append(((y0 + y1) / 2) + 0.12)
        label_text.append(f"{relationship} · {confidence:.2f}")
        annotations.append(
            dict(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=max(1.5, width - 0.5),
                arrowcolor=color,
                opacity=0.55 + (0.35 * confidence),
            )
        )

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=12, color="#111111"),
        hoverinfo="skip",
        showlegend=False,
    )

    return edge_traces, label_trace, annotations


# ------------------------------------------
# 7. MAIN FUNCTION
# ------------------------------------------
def draw_experiment_diagram(experiment, rel_filter=None, selected_node=None):

    graph = build_safe_graph(experiment, rel_filter=rel_filter)
    inputs = graph["inputs"]
    outputs = graph["outputs"]
    model_nodes = graph.get("model_nodes", set())
    all_nodes = graph["all_nodes"]
    edges = graph["edges"]

    pos = compute_semantic_layout(inputs, model_nodes, outputs, all_nodes)

    edge_traces, edge_label_trace, annotations = make_edge_traces(edges, pos)

    normalized_selected = normalize_var(selected_node)
    node_x, node_y, node_text, node_color, node_size = make_nodes(
        pos,
        inputs,
        outputs,
        selected_node=normalized_selected,
    )
    node_hover_text = make_node_hover_text(pos, inputs, outputs, model_nodes)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        hovertext=node_hover_text,
        textfont=dict(size=13, color="#111111"),
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='black')
        )
    )

    # Figure
    fig = go.Figure(
        data=[*edge_traces, edge_label_trace, node_trace],
        layout=go.Layout(
            title="Experiment ↔ Model Mapping",
            title_x=0.5,
            showlegend=False,
            hovermode='closest',
            height=max(400, len(all_nodes) * 90),
            margin=dict(b=60, l=60, r=60, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.6, 1.6]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            annotations=annotations,
            font=dict(color="#111111"),
        )
    )

    fig.add_annotation(
        x=-1.0,
        y=max([value[1] for value in pos.values()], default=0) + 0.8,
        text="Inputs",
        showarrow=False,
        font=dict(size=13, color="royalblue"),
    )
    fig.add_annotation(
        x=0.0,
        y=max([value[1] for value in pos.values()], default=0) + 0.8,
        text="Model Components",
        showarrow=False,
        font=dict(size=13, color="seagreen"),
    )
    fig.add_annotation(
        x=1.0,
        y=max([value[1] for value in pos.values()], default=0) + 0.8,
        text="Outputs",
        showarrow=False,
        font=dict(size=13, color="darkorange"),
    )

    return fig