"""
Streamlit UI for the RAG Pipeline.

Start with (while the FastAPI server is running on port 8000):
    streamlit run streamlit_app.py

The UI calls the FastAPI backend at API_BASE_URL (default: http://localhost:8000).
Set the API_BASE_URL environment variable to point at a deployed instance.

Layout:
  Sidebar   — settings (top-k, reranker toggle, verification toggle, API URL)
  Main area — question input, answer with highlighted citations, confidence
              traffic-light, collapsible source cards
"""

import os
import re

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
QUERY_URL = f"{API_BASE_URL}/api/v1/query"
HEALTH_URL = f"{API_BASE_URL}/api/v1/health"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Pipeline",
    page_icon="🔍",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .confidence-high   { color: #2ecc71; font-weight: bold; }
    .confidence-medium { color: #f39c12; font-weight: bold; }
    .confidence-low    { color: #e74c3c; font-weight: bold; }
    .citation-badge    { background: #3498db; color: white;
                         border-radius: 4px; padding: 1px 6px;
                         font-size: 0.75em; font-weight: bold; }
    .verified-yes   { color: #2ecc71; }
    .verified-no    { color: #e74c3c; }
    .verified-unk   { color: #95a5a6; }
    .source-card    { border-left: 3px solid #3498db; padding-left: 0.75em; margin-bottom: 0.5em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")

    api_url_input = st.text_input("API Base URL", value=API_BASE_URL)
    if api_url_input != API_BASE_URL:
        API_BASE_URL = api_url_input.rstrip("/")
        QUERY_URL = f"{API_BASE_URL}/api/v1/query"
        HEALTH_URL = f"{API_BASE_URL}/api/v1/health"

    st.divider()

    top_n = st.slider("Top-N results (reranked)", min_value=1, max_value=10, value=5)
    use_reranker = st.toggle("Use cross-encoder reranker", value=True)
    skip_verification = st.toggle("Skip citation verification (faster)", value=False)

    st.divider()

    # Health check
    if st.button("Check API health"):
        try:
            r = requests.get(HEALTH_URL, timeout=5)
            data = r.json()
            if data.get("pipeline_ready"):
                st.success("API is healthy and pipeline is ready")
            else:
                st.warning("API is up but pipeline is not ready")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")

    st.markdown("---")
    st.caption("RAG Pipeline · Groq llama3-70b · ChromaDB + BM25")


# ── Helper functions ──────────────────────────────────────────────────────────

def _confidence_badge(score: float) -> str:
    if score >= 0.7:
        cls, label = "confidence-high", "HIGH"
    elif score >= 0.4:
        cls, label = "confidence-medium", "MEDIUM"
    else:
        cls, label = "confidence-low", "LOW"
    return f'<span class="{cls}">{label} ({score:.2f})</span>'


def _verification_icon(verified) -> str:
    if verified is True:
        return '<span class="verified-yes">✓ verified</span>'
    if verified is False:
        return '<span class="verified-no">✗ unsupported</span>'
    return '<span class="verified-unk">? unknown</span>'


def _highlight_citations(answer: str, num_sources: int) -> str:
    """Wrap [n] tokens in a styled badge span."""
    def replace(m):
        n = int(m.group(1))
        if 1 <= n <= num_sources:
            return f'<sup class="citation-badge">[{n}]</sup>'
        return m.group(0)
    return re.sub(r"\[(\d+)\]", replace, answer)


def _call_api(question: str) -> dict | None:
    payload = {
        "question": question,
        "top_n": top_n,
        "use_reranker": use_reranker,
        "skip_verification": skip_verification,
    }
    try:
        r = requests.post(QUERY_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot connect to the API at {QUERY_URL}. "
            "Make sure the FastAPI server is running: "
            "`uvicorn src.api.main:app --reload --port 8000`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("RAG Pipeline — Document Q&A")
st.caption(
    "Ask questions about your ingested documents. "
    "Answers are grounded in retrieved chunks and include citation verification."
)

question = st.text_area(
    "Your question",
    placeholder="What does the document say about…?",
    height=100,
)

ask_btn = st.button("Ask", type="primary", use_container_width=True)

if ask_btn and question.strip():
    with st.spinner("Retrieving and generating…"):
        data = _call_api(question.strip())

    if data:
        # ── Answer ────────────────────────────────────────────────────────────
        st.subheader("Answer")
        highlighted = _highlight_citations(data["answer"], len(data["cited_sources"]))
        st.markdown(highlighted, unsafe_allow_html=True)

        # ── Confidence ────────────────────────────────────────────────────────
        conf = data["confidence"]
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Composite Confidence", f"{conf['composite_score']:.2f}")
        with col2:
            st.metric("Retrieval Quality", f"{conf['retrieval_confidence']:.2f}")
        with col3:
            st.metric("Citation Coverage", f"{conf['citation_coverage']:.2f}")
        with col4:
            st.metric("Completeness", f"{conf['completeness_score']:.2f}")

        # Traffic-light badge
        st.markdown(
            f"Confidence level: {_confidence_badge(conf['composite_score'])}",
            unsafe_allow_html=True,
        )

        # ── Cited sources ─────────────────────────────────────────────────────
        if data["cited_sources"]:
            st.markdown("---")
            st.subheader(f"Sources ({len(data['cited_sources'])})")
            for src in data["cited_sources"]:
                with st.expander(
                    f"[{src['citation_number']}] {src['source']}  —  "
                    f"score: {src['score']:.3f}",
                    expanded=False,
                ):
                    ver_html = _verification_icon(src["verified"])
                    st.markdown(
                        f"**Status:** {ver_html}  \n"
                        f"**Reason:** {src['verification_reason']}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f'<div class="source-card">{src["content"]}</div>', unsafe_allow_html=True)

        # ── Meta ──────────────────────────────────────────────────────────────
        st.caption(
            f"Model: {data['model']} · Latency: {data['latency_ms']:.0f} ms"
        )

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")
