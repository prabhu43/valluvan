"""Valluvan — Streamlit chat interface.

A conversational UI over the Thirukkural RAG pipeline (rag/rag.py). Ask a
life/ethics question; Valluvan answers grounded in the retrieved kurals, citing
the couplets it used. Every answer shows the source kurals (Tamil + English) and
its retrieval/LLM telemetry, and can be rated 👍/👎 (logged via app/storage.py
for Phase 9 monitoring).

Run:  make app      (i.e. streamlit run app/streamlit_app.py)
Needs Qdrant up (`make qdrant-up`) and a GROQ_API_KEY in .env.
"""

import sys
from pathlib import Path

# Streamlit runs this file as a script, so the repo root isn't on sys.path by
# default; add it so the `rag` and `app` packages import cleanly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.storage import log_feedback, log_interaction  # noqa: E402
from rag.rag import (  # noqa: E402
    DEFAULT_PROMPT,
    LLM_MODEL,
    PROMPTS,
    RETRIEVAL_MODE,
    answer,
)
from rag.search import SEARCHERS  # noqa: E402

st.set_page_config(page_title="Valluvan — Thirukkural Wisdom", page_icon="📜")

# Starter questions shown on the welcome screen, grouped by the Thirukkural's
# three sections (paals). Clicking one submits it like a typed question.
SUGGESTIONS = {
    "அறம் · Virtue & ethics": [
        "How can I control my anger?",
        "Why is honesty important, and what happens when I lie?",
        "What does the Thirukkural say about being kind to guests?",
        "How should I treat people who wronged me?",
    ],
    "பொருள் · Wealth, work & leadership": [
        "What makes a good leader?",
        "How should I handle money and avoid poverty?",
        "Why does the Thirukkural warn against laziness?",
        "How do I choose the right friends?",
    ],
    "இன்பம் · Love & relationships": [
        "What does the Thirukkural say about true love between partners?",
        "How does separation from a loved one feel?",
    ],
}


@st.cache_resource(show_spinner=False)
def _warm():
    """Pre-load embedding/rerank models once so the first query isn't slow."""
    from rag.search import _clients

    _clients()
    return True


def _ask(question: str, mode: str, prompt: str, limit: int) -> dict:
    result = answer(question, mode=mode, prompt=prompt, limit=limit)
    result["interaction_id"] = log_interaction(question, result)
    return result


def _render_sources(kurals: list[dict]) -> None:
    with st.expander(f"📖 Sources — {len(kurals)} kural(s) cited"):
        for k in kurals:
            st.markdown(
                f"**Kural {k['kural_no']}** · _{k['adhigaram_en']}_ "
                f"({k['section_en']})"
            )
            st.markdown(
                f"<div style='font-size:1.05rem'>{k['kural_ta']}</div>",
                unsafe_allow_html=True,
            )
            if k.get("transliteration"):
                st.caption(k["transliteration"])
            st.markdown(f"**Translation:** {k['translation_en']}")
            st.markdown(f"**Meaning:** {k['explanation_en']}")
            if k.get("rerank_score") is not None:
                st.caption(f"rerank score: {k['rerank_score']:.2f}")
            st.divider()


def _render_telemetry(r: dict) -> None:
    bits = [
        f"`{r.get('provider')}/{r.get('model')}`",
        f"retrieval: `{r.get('retrieval_mode')}`",
        f"prompt: `{r.get('prompt_variant')}`",
        f"{r.get('latency_s')}s",
        f"{r.get('total_tokens')} tokens",
    ]
    if r.get("rewritten_query"):
        bits.append(f"rewritten: _{r['rewritten_query']}_")
    st.caption(" · ".join(bits))


def _render_feedback(r: dict) -> None:
    iid = r["interaction_id"]
    current = st.session_state.feedback.get(iid)
    col_up, col_down, col_msg = st.columns([1, 1, 6])
    if col_up.button("👍", key=f"up_{iid}", disabled=current is not None):
        st.session_state.feedback[iid] = 1
        log_feedback(iid, 1)
        st.rerun()
    if col_down.button("👎", key=f"down_{iid}", disabled=current is not None):
        st.session_state.feedback[iid] = -1
        log_feedback(iid, -1)
        st.rerun()
    if current == 1:
        col_msg.caption("Thanks for the 👍")
    elif current == -1:
        col_msg.caption("Thanks — noted the 👎")


def _render_assistant(msg: dict) -> None:
    r = msg["result"]
    st.markdown(r["answer"])
    _render_sources(r["kurals"])
    _render_telemetry(r)
    _render_feedback(r)


def _render_suggestions() -> None:
    """Starter-question chips inside a collapsible panel pinned at the top.

    Expanded on the welcome screen; auto-collapses once a conversation starts
    but stays available (click the header to reopen).
    """
    with st.expander(
        "💡 Suggested questions", expanded=not st.session_state.messages
    ):
        for section, questions in SUGGESTIONS.items():
            st.markdown(f"**{section}**")
            cols = st.columns(2)
            for i, q in enumerate(questions):
                if cols[i % 2].button(
                    q, key=f"suggest_{q}", use_container_width=True
                ):
                    st.session_state.pending = q
                    st.rerun()


# --- Session state ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}

# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.selectbox(
        "Retrieval mode",
        list(SEARCHERS.keys()),
        index=list(SEARCHERS.keys()).index(RETRIEVAL_MODE)
        if RETRIEVAL_MODE in SEARCHERS
        else 0,
        help="Default `rerank` won the retrieval evaluation.",
    )
    prompt = st.selectbox(
        "Prompt variant",
        list(PROMPTS.keys()),
        index=list(PROMPTS.keys()).index(DEFAULT_PROMPT)
        if DEFAULT_PROMPT in PROMPTS
        else 0,
        help="Default `concise` won the LLM evaluation.",
    )
    limit = st.slider("Kurals retrieved (k)", 3, 10, 5)
    st.caption(f"LLM: `{LLM_MODEL}`")
    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(
        "Valluvan answers only from the 1,330 kurals of the Thirukkural and "
        "cites the couplets it uses."
    )

# --- Header ----------------------------------------------------------------
st.title("📜 Valluvan")
st.caption(
    "Timeless wisdom from the **Thirukkural** of Thiruvalluvar — ask about life, "
    "ethics, love, or leadership, and get guidance grounded in the couplets."
)
_warm()

# --- Suggested questions ---------------------------------------------------
# Pinned at the top, always available: expanded on the welcome screen, collapsed
# (but reopenable) once a conversation is underway.
_render_suggestions()

# --- Replay history --------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            _render_assistant(msg)
        else:
            st.markdown(msg["content"])

# --- Handle new input ------------------------------------------------------
# A question can come from a clicked suggestion (session_state.pending) or the
# chat input box; both flow through the same path below.
typed = st.chat_input("Ask Valluvan a question…")
question = st.session_state.pop("pending", None) or typed

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Thirukkural…"):
            try:
                result = _ask(question, mode=mode, prompt=prompt, limit=limit)
            except Exception as e:  # noqa: BLE001 - surface errors to the user
                st.error(f"Something went wrong: {e}")
                st.stop()
        st.session_state.messages.append({"role": "assistant", "result": result})
        _render_assistant(st.session_state.messages[-1])
