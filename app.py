# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

import streamlit as st
from pipeline import analyze_text
from visualization import draw_experiment_diagram

st.set_page_config(layout="wide")
st.title("🧠 Neuro Model Mapper")

# ------------------------------------------
# MODE SELECTION
# ------------------------------------------
mode = st.radio(
    "Select Analysis Mode",
    ["Model Diagrams", "Experiment ↔ Model Mapping"]
)

# ------------------------------------------
# INPUT TEXT
# ------------------------------------------
text = st.text_area("Paste paper text here")

if st.button("Analyze"):

    results = analyze_text(text, mode)

    # --------------------------------------
    # FILTER PANEL (SIDEBAR)
    # --------------------------------------
    st.sidebar.header("🔍 Filters")

    rel_filter = st.sidebar.multiselect(
        "Relationship Type",
        ["tests", "correlates", "controls"],
        default=["tests"]
    )

    selected_node = st.sidebar.selectbox(
        "Highlight Node",
        ["None"] + results["all_nodes"]
    )

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------
    for exp in results["experiments"]:

        with st.expander(exp["name"]):

            fig = draw_experiment_diagram(
                exp,
                rel_filter=rel_filter,
                selected_node=selected_node
            )

            st.plotly_chart(fig, use_container_width=True)