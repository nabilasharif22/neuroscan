# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

import streamlit as st
from pipeline import analyze_text
from visualization import draw_experiment_diagram
from bug_checks import ALLOWED_RELATIONSHIPS

st.set_page_config(layout="wide")
st.title("🧠 Neuro Model Mapper")

# ------------------------------------------
# INPUT TEXT
# ------------------------------------------
text = st.text_area("Paste paper text here")

# --------------------------------------
# FILTER PANEL (SIDEBAR)
# --------------------------------------
st.sidebar.header("🔍 Filters")

rel_filter = st.sidebar.multiselect(
    "Relationship Type",
    sorted(ALLOWED_RELATIONSHIPS - {"unknown"}),
    default=["tests"],
)

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

if st.button("Analyze"):

    results = analyze_text(
        text,
        rel_filter=rel_filter,
        min_confidence=min_confidence,
    )

    issues = results.get("issues", [])
    if issues:
        st.warning("Data quality safeguards were applied to clean malformed input.")
        with st.expander("Show validation details"):
            for issue in issues:
                st.write(f"- {issue}")

    selected_node = st.sidebar.selectbox(
        "Highlight Node",
        ["None"] + results.get("all_nodes", [])
    )
    selected_node = None if selected_node == "None" else selected_node

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------
    experiments = results.get("experiments", [])

    if not experiments:
        st.info("No experiment-model links passed current filters.")

    for exp in experiments:

        with st.expander(exp["name"]):

            fig = draw_experiment_diagram(
                exp,
                rel_filter=rel_filter,
                selected_node=selected_node
            )

            st.plotly_chart(fig, use_container_width=True)