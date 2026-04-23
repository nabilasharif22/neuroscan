# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================

import io
import importlib
import time

import streamlit as st
from pipeline import analyze_text
from visualization import draw_experiment_diagram
from bug_checks import ALLOWED_RELATIONSHIPS
from llm import get_llm_status


def extract_uploaded_text(uploaded_file):
    if uploaded_file is None:
        return "", None

    file_name = (uploaded_file.name or "").lower()

    if file_name.endswith(".txt"):
        try:
            return uploaded_file.getvalue().decode("utf-8", errors="replace"), None
        except Exception as exc:
            return "", f"Could not read TXT file: {type(exc).__name__}"

    if file_name.endswith(".pdf"):
        try:
            pypdf_module = importlib.import_module("pypdf")
            pdf_reader_cls = getattr(pypdf_module, "PdfReader", None)
            if pdf_reader_cls is None:
                return "", "PDF support is unavailable because `pypdf` is not installed correctly."

            pdf_bytes = uploaded_file.getvalue()
            reader = pdf_reader_cls(io.BytesIO(pdf_bytes))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n\n".join(pages).strip(), None
        except Exception as exc:
            return "", f"Could not parse PDF: {type(exc).__name__}"

    return "", "Unsupported file type. Please upload a .pdf or .txt file."

st.set_page_config(layout="wide")
st.title("🧠 Neuro Model Mapper")

if "llm_status" not in st.session_state:
    st.session_state["llm_status"] = get_llm_status()

llm_status = st.session_state["llm_status"]
if llm_status.get("mode") == "mock":
    st.info("⚠️ Mock LLM mode is active — OpenAI API is NOT being used.")
if llm_status.get("api_configured") and llm_status.get("mode") == "mock":
    st.warning(f"OpenAI API is configured but currently failing. Using mock fallback. Reason: {llm_status.get('message', 'Unknown error')}")

st.sidebar.markdown("### 🤖 LLM Status")
if llm_status.get("mode") == "api":
    st.sidebar.success("API")
else:
    st.sidebar.warning("MOCK")
st.sidebar.caption(llm_status.get("message", ""))
if llm_status.get("api_configured") and llm_status.get("mode") == "mock":
    st.sidebar.error("API configured, but fallback is active")

st.sidebar.markdown("### 🗺️ Graph Legend")
st.sidebar.markdown("- 🔵 Input node: manipulated variable")
st.sidebar.markdown("- 🟢 Model node: latent/model component")
st.sidebar.markdown("- 🟠 Output node: measured variable")
st.sidebar.markdown("- Solid arrow: input → model link")
st.sidebar.markdown("- Dotted arrow: model → output link")
st.sidebar.markdown("- Edge color: relationship type")
st.sidebar.markdown("- Edge label: relationship · confidence")

# ------------------------------------------
# INPUT TEXT
# ------------------------------------------
uploaded_file = st.file_uploader(
    "Upload paper (.pdf or .txt)",
    type=["pdf", "txt"],
    accept_multiple_files=False,
)

text = st.text_area("Paste paper text here (optional)")

uploaded_text = ""
upload_error = None
if uploaded_file is not None:
    uploaded_text, upload_error = extract_uploaded_text(uploaded_file)
    if upload_error:
        st.error(upload_error)
    elif uploaded_text:
        with st.expander("Preview extracted upload text", expanded=False):
            preview_text = uploaded_text[:5000]
            st.text_area(
                "Extracted text preview",
                value=preview_text,
                height=220,
                disabled=True,
            )
            if len(uploaded_text) > len(preview_text):
                st.caption(f"Preview shows first {len(preview_text)} characters of {len(uploaded_text)} total.")
    else:
        st.warning("Uploaded file was parsed but no readable text was found.")

# --------------------------------------
# FILTER PANEL (SIDEBAR)
# --------------------------------------
st.sidebar.header("🔍 Filters")

_all_rels = sorted(ALLOWED_RELATIONSHIPS - {"unknown"})
rel_filter = st.sidebar.multiselect(
    "Relationship Type",
    _all_rels,
    default=_all_rels,
)

min_confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

if st.button("Analyze"):

    input_chunks = [uploaded_text.strip(), text.strip()]
    analysis_text = "\n\n".join(chunk for chunk in input_chunks if chunk)

    if not analysis_text:
        st.warning("Please upload a PDF/TXT file or paste text before running analysis.")
        st.stop()

    if uploaded_text and text.strip():
        st.caption("Using both uploaded content and pasted text for analysis.")
    elif uploaded_text:
        st.caption("Using uploaded file content for analysis.")
    else:
        st.caption("Using pasted text for analysis.")

    start_time = time.perf_counter()
    with st.spinner("Analyzing text and building graph..."):
        results = analyze_text(
            analysis_text,
            rel_filter=rel_filter,
            min_confidence=min_confidence,
        )
    elapsed_seconds = time.perf_counter() - start_time
    st.success(f"Analysis completed in {elapsed_seconds:.2f} seconds.")

    issues = results.get("issues", [])
    result_llm_status = results.get("llm_status", llm_status)
    st.session_state["llm_status"] = result_llm_status

    with st.expander("How to read this graph", expanded=False):
        st.markdown(
            "- **Columns:** Inputs (left), model components (middle), outputs (right).\n"
            "- **Arrows:** Solid lines are input→model links; dotted lines are model→output links.\n"
            "- **Relationship colors:** Each edge color represents tests/correlates/controls/modulates/causes.\n"
            "- **Confidence values:** Labels like `modulates · 0.78` show extraction certainty in [0,1].\n"
            "- **Filters:** Relationship and minimum confidence filters hide links that don't match your selection."
        )

    if result_llm_status.get("api_configured") and result_llm_status.get("mode") == "mock":
        raw_reason = result_llm_status.get('message', 'Unknown error')
        if "RateLimitError" in raw_reason and "TPD" in raw_reason:
            human_reason = "Groq daily token limit reached (100k/day on free tier). Try again in ~15 min or upgrade at console.groq.com/settings/billing."
        elif "RateLimitError" in raw_reason:
            human_reason = "API rate limit reached. Please wait a moment and try again."
        else:
            human_reason = raw_reason
        st.error(f"API call failed during analysis; mock fallback used. Reason: {human_reason}")
    if result_llm_status.get("mode") == "mock":
        st.caption("Current extraction source: Mock data (no API calls).")
    else:
        _msg = result_llm_status.get("message", "")
        if "Groq" in _msg:
            st.caption(f"Current extraction source: Groq API (model: {result_llm_status.get('model', '')}).")
        else:
            st.caption(f"Current extraction source: OpenAI API (model: {result_llm_status.get('model', '')}).")

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

    for index, exp in enumerate(experiments):

        with st.expander(exp["name"]):

            fig = draw_experiment_diagram(
                exp,
                rel_filter=rel_filter,
                selected_node=selected_node
            )

            chart_key = f"plot_{index}_{exp.get('name', 'experiment')}"
            st.plotly_chart(fig, width="stretch", key=chart_key)