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

st.set_page_config(
    page_title="NeuroScan — Computational Model Extractor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom academic CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
}

/* Page title */
h1 { font-family: 'Lora', Georgia, serif !important; font-weight: 600 !important; color: #1a1a2e !important; letter-spacing: -0.3px; }
h2, h3 { font-family: 'Lora', Georgia, serif !important; color: #1a1a2e !important; }

/* Subtle top rule under title */
.block-container { padding-top: 2rem !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #f4f4f0 !important; border-right: 1px solid #ddd; }
section[data-testid="stSidebar"] .stMarkdown p { font-size: 0.82rem; color: #444; }

/* Buttons */
div.stButton > button {
    background: #1d4e8f !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.6rem !important;
    letter-spacing: 0.3px;
    font-size: 0.9rem !important;
}
div.stButton > button:hover { background: #163d72 !important; }

/* Expander header */
details summary { font-weight: 600; font-size: 0.9rem; color: #1a1a2e; }

/* File uploader */
[data-testid="stFileUploader"] { border: 1.5px dashed #b0b0b0 !important; border-radius: 6px !important; padding: 0.6rem !important; background: #fafaf8 !important; }

/* Caption text */
.stCaption { color: #666 !important; font-size: 0.77rem !important; }

/* Divider */
hr { border: none; border-top: 1px solid #ddd; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- Header ---
col_logo, col_title = st.columns([0.06, 0.94])
with col_logo:
    st.markdown("<div style='font-size:2.4rem;line-height:1;padding-top:6px'>🧠</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(
        "<h1 style='margin:0;font-size:1.9rem'>NeuroScan</h1>"
        "<p style='margin:0;color:#555;font-size:0.88rem;font-family:Inter,sans-serif'>"
        "Computational model structure extractor for neuroscience papers</p>",
        unsafe_allow_html=True,
    )
st.markdown("<hr>", unsafe_allow_html=True)

if "llm_status" not in st.session_state:
    st.session_state["llm_status"] = get_llm_status()

llm_status = st.session_state["llm_status"]

# --- Sidebar ---
st.sidebar.markdown(
    "<p style='font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;"
    "color:#888;font-weight:600;margin-bottom:4px'>Extraction Engine</p>",
    unsafe_allow_html=True,
)
if llm_status.get("mode") == "api":
    st.sidebar.success(f"✓ API — {llm_status.get('model', '')}")
else:
    st.sidebar.warning("⚠ Mock mode (no API key)")
st.sidebar.caption(llm_status.get("message", ""))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;"
    "color:#888;font-weight:600;margin-bottom:6px'>Graph Legend</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<div style='font-size:0.82rem;line-height:1.9'>"
    "<span style='color:#1d4e8f'>⬤</span>&nbsp; <b>Input</b> — manipulated variable<br>"
    "<span style='color:#1a6b3c'>⬤</span>&nbsp; <b>Model node</b> — latent component<br>"
    "<span style='color:#8b3a00'>⬤</span>&nbsp; <b>Output</b> — measured variable<br>"
    "<span style='color:#b91c1c'>—</span>&nbsp; <b>causes</b>&emsp;"
    "<span style='color:#b45309'>—</span>&nbsp; <b>modulates</b><br>"
    "<span style='color:#2563eb'>—</span>&nbsp; <b>tests</b>&emsp;&emsp;"
    "<span style='color:#0f766e'>—</span>&nbsp; <b>controls</b><br>"
    "<span style='color:#7c3aed'>—</span>&nbsp; <b>correlates</b><br>"
    "― solid = input→model &nbsp;|&nbsp; ··· dotted = model→output"
    "</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------
# INPUT PANEL
# ------------------------------------------
st.markdown("#### Upload or paste a paper")

col_upload, col_paste = st.columns([1, 1], gap="large")
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload paper (.pdf or .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
with col_paste:
    text = st.text_area(
        "Or paste text directly",
        placeholder="Paste abstract, methods, or full paper text here…",
        height=140,
        label_visibility="visible",
    )

uploaded_text = ""
upload_error = None
if uploaded_file is not None:
    uploaded_text, upload_error = extract_uploaded_text(uploaded_file)
    if upload_error:
        st.error(upload_error)
    elif uploaded_text:
        with st.expander("Preview extracted text", expanded=False):
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
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;"
    "color:#888;font-weight:600;margin-bottom:6px'>Filters</p>",
    unsafe_allow_html=True,
)

_all_rels = sorted(ALLOWED_RELATIONSHIPS - {"unknown"})
rel_filter = st.sidebar.multiselect(
    "Relationship types",
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

st.markdown("<br>", unsafe_allow_html=True)
analyze_clicked = st.button("⚡ Run Analysis", use_container_width=False)

if analyze_clicked:

    input_chunks = [uploaded_text.strip(), text.strip()]
    analysis_text = "\n\n".join(chunk for chunk in input_chunks if chunk)

    if not analysis_text:
        st.warning("Please upload a PDF/TXT file or paste text before running analysis.")
        st.stop()

    if uploaded_text and text.strip():
        st.caption("Source: uploaded file + pasted text combined.")
    elif uploaded_text:
        st.caption("Source: uploaded file.")
    else:
        st.caption("Source: pasted text.")

    start_time = time.perf_counter()
    with st.spinner("Segmenting paper and extracting model structure…"):
        results = analyze_text(
            analysis_text,
            rel_filter=rel_filter,
            min_confidence=min_confidence,
        )
    elapsed_seconds = time.perf_counter() - start_time
    st.success(f"Extraction complete — {elapsed_seconds:.2f} s")

    issues = results.get("issues", [])
    result_llm_status = results.get("llm_status", llm_status)
    st.session_state["llm_status"] = result_llm_status

    # API / rate-limit notices
    if result_llm_status.get("api_configured") and result_llm_status.get("mode") == "mock":
        raw_reason = result_llm_status.get("message", "Unknown error")
        if "RateLimitError" in raw_reason and "TPD" in raw_reason:
            human_reason = (
                "Groq daily token limit reached (100k/day on free tier). "
                "Try again in ~15 min or upgrade at console.groq.com/settings/billing."
            )
        elif "RateLimitError" in raw_reason:
            human_reason = "API rate limit reached. Please wait a moment and try again."
        else:
            human_reason = raw_reason
        st.error(f"API call failed; mock fallback used. Reason: {human_reason}")

    _msg = result_llm_status.get("message", "")
    if result_llm_status.get("mode") == "mock":
        st.caption("Extraction source: mock data.")
    elif "Groq" in _msg:
        st.caption(f"Extraction source: Groq — {result_llm_status.get('model', '')}")
    else:
        st.caption(f"Extraction source: OpenAI — {result_llm_status.get('model', '')}")

    if issues:
        with st.expander("⚠ Validation notices", expanded=False):
            for issue in issues:
                st.markdown(f"<span style='font-size:0.82rem;color:#666'>• {issue}</span>", unsafe_allow_html=True)

    selected_node = st.sidebar.selectbox(
        "Highlight node",
        ["None"] + results.get("all_nodes", [])
    )
    selected_node = None if selected_node == "None" else selected_node

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------
    experiments = results.get("experiments", [])

    if not experiments:
        st.info("No experiment-model links passed the current filters.")

    for index, exp in enumerate(experiments):

        exp_name = exp.get("name", f"Experiment {index + 1}")
        manip    = exp.get("manipulated_variables", [])
        measured = exp.get("measured_variables", [])

        # Render experiment card with metadata row
        st.markdown(
            f"<div style='margin-top:1.4rem;padding-bottom:2px;border-bottom:2px solid #1d4e8f'>"
            f"<span style='font-family:Lora,Georgia,serif;font-size:1.05rem;font-weight:600;"
            f"color:#1a1a2e'>{exp_name}</span></div>",
            unsafe_allow_html=True,
        )
        meta_parts = []
        if manip:
            meta_parts.append(
                "<span style='color:#1d4e8f'><b>Manipulated:</b></span> "
                + ", ".join(f"<i>{v}</i>" for v in manip)
            )
        if measured:
            meta_parts.append(
                "<span style='color:#8b3a00'><b>Measured:</b></span> "
                + ", ".join(f"<i>{v}</i>" for v in measured)
            )
        if meta_parts:
            st.markdown(
                "<div style='font-size:0.82rem;color:#444;margin:6px 0 10px'>"
                " &nbsp;|&nbsp; ".join(meta_parts) + "</div>",
                unsafe_allow_html=True,
            )

        fig = draw_experiment_diagram(
            exp,
            rel_filter=rel_filter,
            selected_node=selected_node,
        )
        chart_key = f"plot_{index}_{exp_name}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Graph reading guide (collapsed by default)
        with st.expander("How to read this figure", expanded=False):
            st.markdown(
                "<div style='font-size:0.82rem;color:#444;line-height:1.75'>"
                "<b>Columns:</b> Inputs (left) · Model components (centre) · Outputs (right)<br>"
                "<b>Arrows:</b> Solid = input→model link · Dotted = model→output link<br>"
                "<b>Edge colour:</b> encodes relationship type (see legend in sidebar)<br>"
                "<b>Edge label:</b> <i>relationship</i> · confidence [0–1]<br>"
                "<b>Confidence</b> reflects LLM extraction certainty, not statistical significance"
                "</div>",
                unsafe_allow_html=True,
            )