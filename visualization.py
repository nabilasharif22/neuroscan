# ==========================================
# visualization.py
# ==========================================

import plotly.graph_objects as go
from bug_checks import build_safe_graph, normalize_var

# --- Academic color palette ---
# Inspired by journal figure conventions (Nature / PNAS style)
NODE_COLORS = {
    "input":   "#1d4e8f",   # deep blue
    "model":   "#1a6b3c",   # forest green
    "output":  "#8b3a00",   # burnt sienna
    "default": "#4a4a4a",   # slate
    "highlight": "#9b1c1c", # deep crimson
}

NODE_BORDER = {
    "input":   "#0f2d55",
    "model":   "#0d3d20",
    "output":  "#5a2600",
    "default": "#222222",
    "highlight": "#6b0f0f",
}

RELATIONSHIP_COLORS = {
    "tests":      "#2563eb",   # blue
    "correlates": "#7c3aed",   # violet
    "controls":   "#0f766e",   # teal
    "modulates":  "#b45309",   # amber-brown
    "causes":     "#b91c1c",   # deep red
    "unknown":    "#6b7280",   # grey
}

BG_COLOR   = "#fafaf8"   # off-white, like paper
GRID_COLOR = "#ececec"
FONT_FAMILY = "'Georgia', 'Times New Roman', serif"
LABEL_FONT  = "'Inter', 'Helvetica Neue', Arial, sans-serif"


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
        exp_var   = normalize_var(link.get("experiment_variable"))
        model_comp = normalize_var(link.get("model_component"))
        if not exp_var and not model_comp:
            continue
        if exp_var:
            all_nodes.add(exp_var)
        if model_comp:
            all_nodes.add(model_comp)
        edges.append({
            "source": exp_var,
            "target": model_comp,
            "type": link.get("relationship", "unknown"),
            "confidence": link.get("confidence", 0.5),
        })
    return inputs, outputs, all_nodes, edges


# ------------------------------------------
# 4. Layout helpers
# ------------------------------------------
def _centered_positions(nodes, x_value, spacing=2.2):
    positions = {}
    ordered = sorted(nodes)
    n = len(ordered)
    if n == 0:
        return positions
    midpoint = ((n - 1) / 2) * spacing
    for i, node in enumerate(ordered):
        positions[node] = (x_value, midpoint - i * spacing)
    return positions


def compute_semantic_layout(inputs, model_nodes, outputs, all_nodes):
    pos = {}
    pos.update(_centered_positions(inputs,      -1.0))
    pos.update(_centered_positions(model_nodes,  0.0))
    remaining_outputs = sorted(set(outputs) - set(model_nodes))
    pos.update(_centered_positions(remaining_outputs, 1.0))
    unassigned = sorted(set(all_nodes) - set(pos))
    if unassigned:
        pos.update(_centered_positions(unassigned, 0.5))
    return pos


# ------------------------------------------
# 5. Draw edges (legacy helper, kept for compat)
# ------------------------------------------
def make_edges(edges, pos):
    edge_x, edge_y = [], []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src not in pos or tgt not in pos:
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
    x, y, text, color, border_color, size = [], [], [], [], [], []
    for node, (px, py) in pos.items():
        x.append(px)
        y.append(py)
        text.append(node)
        if selected_node and node == selected_node:
            color.append(NODE_COLORS["highlight"])
            border_color.append(NODE_BORDER["highlight"])
            size.append(30)
        elif node in inputs:
            color.append(NODE_COLORS["input"])
            border_color.append(NODE_BORDER["input"])
            size.append(24)
        elif node in outputs:
            color.append(NODE_COLORS["output"])
            border_color.append(NODE_BORDER["output"])
            size.append(24)
        else:
            color.append(NODE_COLORS["model"])
            border_color.append(NODE_BORDER["model"])
            size.append(24)
    return x, y, text, color, border_color, size


def make_node_hover_text(pos, inputs, outputs, model_nodes):
    hover_text = []
    for node in pos:
        if node in inputs:
            role = "Manipulated variable"
        elif node in outputs:
            role = "Measured variable"
        elif node in model_nodes:
            role = "Model component"
        else:
            role = "Derived node"
        hover_text.append(f"<b>{node}</b><br>{role}")
    return hover_text


def make_edge_traces(edges, pos):
    edge_traces = []
    label_x, label_y, label_text = [], [], []
    annotations = []

    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        if src not in pos or tgt not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        relationship = edge.get("type", "unknown")
        confidence   = edge.get("confidence", 0.5)
        edge_kind    = edge.get("kind", "input_to_model")
        color        = RELATIONSHIP_COLORS.get(relationship, RELATIONSHIP_COLORS["unknown"])

        # Width scales gently with confidence; never too thin or too thick
        width      = 1.2 + (2.2 * confidence)
        line_dash  = "dot" if edge_kind == "model_to_output" else "solid"
        path_label = "model → outcome" if edge_kind == "model_to_output" else "input → model"
        opacity    = 0.45 + (0.45 * confidence)

        edge_traces.append(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color, dash=line_dash),
                opacity=opacity,
                hoverinfo="text",
                text=[
                    f"<b>{src} → {tgt}</b><br>"
                    f"Path: {path_label}<br>"
                    f"Relationship: <i>{relationship}</i><br>"
                    f"Confidence: {confidence:.2f}",
                ] * 2,
                showlegend=False,
            )
        )

        # Midpoint label: small, italic, relationship + confidence
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2 + 0.14
        label_x.append(mx)
        label_y.append(my)
        label_text.append(f"<i>{relationship}</i> · {confidence:.2f}")

        # Arrow annotation on target end
        annotations.append(dict(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=max(1.2, width - 0.5),
            arrowcolor=color,
            opacity=opacity,
        ))

    label_trace = go.Scatter(
        x=label_x,
        y=label_y,
        mode="text",
        text=label_text,
        textfont=dict(size=10, color="#444444", family=LABEL_FONT),
        hoverinfo="skip",
        showlegend=False,
    )

    return edge_traces, label_trace, annotations


# ------------------------------------------
# 7. MAIN FUNCTION
# ------------------------------------------
def draw_experiment_diagram(experiment, rel_filter=None, selected_node=None):

    graph = build_safe_graph(experiment, rel_filter=rel_filter)
    inputs      = graph["inputs"]
    outputs     = graph["outputs"]
    model_nodes = graph.get("model_nodes", set())
    all_nodes   = graph["all_nodes"]
    edges       = graph["edges"]

    pos = compute_semantic_layout(inputs, model_nodes, outputs, all_nodes)

    edge_traces, edge_label_trace, annotations = make_edge_traces(edges, pos)

    normalized_selected = normalize_var(selected_node)
    x, y, text, color, border_color, size = make_nodes(
        pos, inputs, outputs, selected_node=normalized_selected
    )
    hover_text = make_node_hover_text(pos, inputs, outputs, model_nodes)

    node_trace = go.Scatter(
        x=x,
        y=y,
        mode="markers+text",
        text=text,
        textposition="bottom center",
        hoverinfo="text",
        hovertext=hover_text,
        textfont=dict(
            size=12,
            color="#1a1a1a",
            family=LABEL_FONT,
        ),
        marker=dict(
            size=size,
            color=color,
            line=dict(width=2, color=border_color),
            opacity=0.92,
        ),
    )

    # --- Column header y position ---
    max_y = max((v[1] for v in pos.values()), default=0)
    header_y = max_y + 1.1

    column_annotations = [
        dict(
            x=-1.0, y=header_y,
            text="<b>Inputs</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["input"], family=FONT_FAMILY),
            bgcolor="rgba(29,78,143,0.08)",
            bordercolor=NODE_COLORS["input"],
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=0.0, y=header_y,
            text="<b>Model Components</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["model"], family=FONT_FAMILY),
            bgcolor="rgba(26,107,60,0.08)",
            bordercolor=NODE_COLORS["model"],
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=1.0, y=header_y,
            text="<b>Outputs</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["output"], family=FONT_FAMILY),
            bgcolor="rgba(139,58,0,0.08)",
            bordercolor=NODE_COLORS["output"],
            borderwidth=1,
            borderpad=4,
        ),
    ]

    all_annotations = annotations + column_annotations

    fig = go.Figure(
        data=[*edge_traces, edge_label_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Experiment ↔ Model Mapping",
                x=0.5,
                font=dict(size=15, color="#1a1a1a", family=FONT_FAMILY),
            ),
            showlegend=False,
            hovermode="closest",
            height=max(420, len(all_nodes) * 100),
            margin=dict(b=70, l=50, r=50, t=70),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.75, 1.75],
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
            ),
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            annotations=all_annotations,
            font=dict(color="#1a1a1a", family=LABEL_FONT),
        ),
    )

    return fig