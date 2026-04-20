# ==========================================
# INTERACTIVE PLOTLY DIAGRAM
# ==========================================

import plotly.graph_objects as go

def draw_experiment_diagram(exp, rel_filter, selected_node):

    fig = go.Figure()

    pos = {}

    # --------------------------------------
    # BUILD NODE POSITIONS
    # --------------------------------------
    nodes = []

    for v in exp["manipulated_variables"]:
        nodes.append((v, 0))

    for link in exp["model_links"]:
        nodes.append((link["model_component"], 1))

    for v in exp["measured_variables"]:
        nodes.append((v, 2))

    nodes = list(set(nodes))

    for i, (name, col) in enumerate(nodes):
        pos[name] = (col, -i)

    # --------------------------------------
    # DRAW NODES
    # --------------------------------------
    for name, col in nodes:

        style = dict(size=40, color="lightblue", opacity=0.8)

        if selected_node != "None":
            if name == selected_node:
                style = dict(size=55, color="red", opacity=1)
            else:
                style["opacity"] = 0.2

        fig.add_trace(go.Scatter(
            x=[pos[name][0]],
            y=[pos[name][1]],
            mode="markers+text",
            text=[name],
            marker=style
        ))

    # --------------------------------------
    # DRAW EDGES
    # --------------------------------------
    for link in exp["model_links"]:

        rel = link["relationship"]
        conf = link.get("confidence", 0.7)

        # default faded
        opacity = 0.1
        width = 1

        if rel in rel_filter:
            opacity = conf
            width = 2

        if selected_node != "None":
            if selected_node in [link["experiment_variable"], link["model_component"]]:
                opacity = 1
                width = 4

        fig.add_annotation(
            x=pos[link["model_component"]][0],
            y=pos[link["model_component"]][1],
            ax=pos[link["experiment_variable"]][0],
            ay=pos[link["experiment_variable"]][1],
            showarrow=True,
            arrowhead=2,
            arrowwidth=width,
            opacity=opacity,
            text=rel
        )

    # --------------------------------------
    # CLEAN LAYOUT
    # --------------------------------------
    fig.update_layout(
        title=exp["name"],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white"
    )

    return fig