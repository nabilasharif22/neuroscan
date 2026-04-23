# ==========================================
# visualization.py
# ==========================================

import plotly.graph_objects as go
from bug_checks import build_safe_graph, normalize_var

# --- Academic color palette ---
# Inspired by journal figure conventions (Nature / PNAS style)
NODE_COLORS = {
    "input":   "#1e40af",
    "model":   "#166534",
    "output":  "#92400e",
    "default": "#5d6676",
    "highlight": "#b91c1c",
}

NODE_BORDER = {
    "input":   "#1e3a8a",
    "model":   "#14532d",
    "output":  "#78350f",
    "default": "#353c47",
    "highlight": "#991b1b",
}

RELATIONSHIP_COLORS = {
    "tests":      "#1d4ed8",
    "correlates": "#6b4ec6",
    "controls":   "#17766f",
    "modulates":  "#a16207",
    "causes":     "#b91c1c",
    "unknown":    "#747c89",
}

BG_COLOR   = "#ffffff"
GRID_COLOR = "#e6e9ef"
FONT_FAMILY = "'Source Serif 4', 'Georgia', 'Times New Roman', serif"
LABEL_FONT  = "'Inter', 'Helvetica Neue', Arial, sans-serif"
TEXT_COLOR = "#0b1220"


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
def _centered_positions(nodes, x_value, spacing=2.45):
    positions = {}
    ordered = sorted(nodes)
    n = len(ordered)
    if n == 0:
        return positions
    midpoint = ((n - 1) / 2) * spacing
    for i, node in enumerate(ordered):
        positions[node] = (x_value, midpoint - i * spacing)
    return positions


def _compute_vertical_spacing(inputs, model_nodes, outputs, all_nodes):
    max_column_size = max(len(inputs), len(model_nodes), len(outputs), 1)
    longest_label = max((len(node) for node in all_nodes), default=12)
    base_spacing = 2.45
    size_boost = max(0.0, (max_column_size - 4) * 0.18)
    label_boost = max(0.0, (longest_label - 16) * 0.03)
    return min(4.0, base_spacing + size_boost + label_boost)


def _wrap_node_label(node, width=16):
    label = str(node).replace("_", " ")
    words = label.split()
    if not words:
        return label

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "<br>".join(lines[:3])


def compute_semantic_layout(inputs, model_nodes, outputs, all_nodes):
    spacing = _compute_vertical_spacing(inputs, model_nodes, outputs, all_nodes)
    pos = {}
    pos.update(_centered_positions(inputs,      -1.0, spacing=spacing))
    pos.update(_centered_positions(model_nodes,  0.0, spacing=spacing))
    remaining_outputs = sorted(set(outputs) - set(model_nodes))
    pos.update(_centered_positions(remaining_outputs, 1.0, spacing=spacing))
    unassigned = sorted(set(all_nodes) - set(pos))
    if unassigned:
        pos.update(_centered_positions(unassigned, 0.5, spacing=spacing))
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
        text.append(_wrap_node_label(node))
        if selected_node and node == selected_node:
            color.append(NODE_COLORS["highlight"])
            border_color.append(NODE_BORDER["highlight"])
            size.append(32)
        elif node in inputs:
            color.append(NODE_COLORS["input"])
            border_color.append(NODE_BORDER["input"])
            size.append(26)
        elif node in outputs:
            color.append(NODE_COLORS["output"])
            border_color.append(NODE_BORDER["output"])
            size.append(26)
        else:
            color.append(NODE_COLORS["model"])
            border_color.append(NODE_BORDER["model"])
            size.append(26)
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

    total_edges = len(edges)
    show_edge_labels = total_edges <= 14

    for index, edge in enumerate(edges):
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
        width      = 1.35 + (1.9 * confidence)
        line_dash  = "dot" if edge_kind == "model_to_output" else "solid"
        path_label = "model → outcome" if edge_kind == "model_to_output" else "input → model"
        opacity    = 0.62 + (0.28 * confidence)

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
        if show_edge_labels:
            offset = 0.18 if index % 2 == 0 else -0.18
            my = (y0 + y1) / 2 + offset
            label_x.append(mx)
            label_y.append(my)
            label_text.append(f"{relationship} · {confidence:.2f}")

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
        textfont=dict(size=12, color=TEXT_COLOR, family=LABEL_FONT),
        hoverinfo="skip",
        showlegend=False,
        visible=show_edge_labels,
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
            size=13,
            color=TEXT_COLOR,
            family=LABEL_FONT,
        ),
        marker=dict(
            size=size,
            color=color,
            line=dict(width=2.2, color=border_color),
            opacity=0.98,
        ),
    )

    # --- Column header y position ---
    max_y = max((v[1] for v in pos.values()), default=0)
    header_y = max_y + 1.25

    column_annotations = [
        dict(
            x=-1.0, y=header_y,
            text="<b>Inputs</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["input"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=NODE_COLORS["input"],
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=0.0, y=header_y,
            text="<b>Model Components</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["model"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=NODE_COLORS["model"],
            borderwidth=1,
            borderpad=4,
        ),
        dict(
            x=1.0, y=header_y,
            text="<b>Outputs</b>",
            showarrow=False,
            font=dict(size=12, color=NODE_COLORS["output"], family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=NODE_COLORS["output"],
            borderwidth=1,
            borderpad=4,
        ),
    ]

    all_annotations = annotations + column_annotations

    band_shapes = [
        dict(type="rect", xref="x", yref="paper", x0=-1.35, x1=-0.65, y0=0, y1=1,
             fillcolor="rgba(30,64,175,0.03)", line=dict(width=0), layer="below"),
        dict(type="rect", xref="x", yref="paper", x0=-0.35, x1=0.35, y0=0, y1=1,
             fillcolor="rgba(22,101,52,0.03)", line=dict(width=0), layer="below"),
        dict(type="rect", xref="x", yref="paper", x0=0.65, x1=1.35, y0=0, y1=1,
             fillcolor="rgba(146,64,14,0.03)", line=dict(width=0), layer="below"),
    ]

    axis_guides = [
        dict(type="line", xref="x", yref="paper", x0=-1.0, x1=-1.0, y0=0, y1=1,
             line=dict(color="rgba(30,64,175,0.22)", width=1, dash="dot"), layer="below"),
        dict(type="line", xref="x", yref="paper", x0=0.0, x1=0.0, y0=0, y1=1,
             line=dict(color="rgba(22,101,52,0.22)", width=1, dash="dot"), layer="below"),
        dict(type="line", xref="x", yref="paper", x0=1.0, x1=1.0, y0=0, y1=1,
             line=dict(color="rgba(146,64,14,0.22)", width=1, dash="dot"), layer="below"),
    ]

    fig = go.Figure(
        data=[*edge_traces, edge_label_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Experiment ↔ Model Mapping",
                x=0.5,
                font=dict(size=16, color=TEXT_COLOR, family=FONT_FAMILY),
            ),
            showlegend=False,
            hovermode="closest",
            height=max(520, len(all_nodes) * 110),
            margin=dict(b=72, l=55, r=55, t=76),
            xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                range=[-1.75, 1.75],
                fixedrange=True,
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False,
                fixedrange=True,
            ),
            plot_bgcolor=BG_COLOR,
            paper_bgcolor=BG_COLOR,
            annotations=all_annotations,
            shapes=band_shapes + axis_guides,
            font=dict(color=TEXT_COLOR, family=LABEL_FONT),
        ),
    )

    return fig