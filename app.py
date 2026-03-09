import base64
import re
from collections import Counter

import graphviz
import penman
import requests
import streamlit as st

st.set_page_config(page_title="AMR Parsing Demo", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 720px; margin: auto; }
h1, h2, h3, .stCaption { text-align: center; }
div[data-testid="stGraphVizChart"] { display: flex; justify-content: center; }
</style>
""", unsafe_allow_html=True)

st.title("AMR Parsing Demo")

MODAL_API_URL = st.secrets.get("MODAL_API_URL", "")

if not MODAL_API_URL:
    st.error(
        "Set `MODAL_API_URL` in `.streamlit/secrets.toml` "
        "(the URL from `modal deploy demo/modal_api.py`)"
    )
    st.stop()

text_col, lang_col = st.columns((4, 1))
sentence = text_col.text_input("Input sentence", placeholder="The boy wants to go.")
lang_col.selectbox("Language", ["English"], index=0, disabled=True)

if sentence and st.button("Parse AMR"):
    with st.spinner("Generating AMR... (first request may take ~2 min for GPU cold start)"):
        try:
            resp = requests.post(
                MODAL_API_URL, json={"sentence": sentence}, timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"API error: {e}")
            st.stop()

    amr_string = data.get("amr", "")
    if not amr_string or not amr_string.startswith("("):
        st.warning("Could not generate a valid AMR graph.")
        if amr_string:
            st.code(amr_string, language="text")
        st.stop()

    try:
        graph = penman.decode(amr_string)
    except Exception as e:
        st.warning(f"Failed to parse AMR: {e}")
        st.code(amr_string, language="text")
        st.stop()

    # --- Graph visualization ---
    st.subheader("Graph visualization")

    viz = graphviz.Digraph(
        graph_attr={"rankdir": "TB", "size": "8,10", "dpi": "72"},
        node_attr={
            "color": "#3aafa9",
            "style": "rounded,filled",
            "shape": "box",
            "fontcolor": "white",
            "fontsize": "12",
            "margin": "0.15,0.08",
        },
        edge_attr={"fontsize": "10"},
    )

    nodename_c = Counter(
        triple[2] for triple in graph.triples if triple[1] == ":instance"
    )
    nodenames = {
        triple[0]: triple[2]
        for triple in graph.triples
        if triple[1] == ":instance"
    }

    # Disambiguate duplicate concept names (e.g. two "go-01" nodes)
    str_c: Counter = Counter()
    for var in nodenames:
        name = nodenames[var]
        if nodename_c[name] > 1:
            str_c[name] += 1
            nodenames[var] = f"{name} ({str_c[name]})"

    def node_label(var: str) -> str:
        return nodenames.get(var, var)

    for triple in graph.triples:
        if triple[1] != ":instance":
            viz.edge(node_label(triple[0]), node_label(triple[2]), label=triple[1])

    st.graphviz_chart(viz)

    # --- Download graph as PNG ---
    png_bytes = viz.pipe(format="png")
    b64 = base64.b64encode(png_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="amr-graph.png">Download graph as PNG</a>'
    st.markdown(href, unsafe_allow_html=True)

    # --- Reasoning ---
    raw = data.get("raw", "")
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        with st.expander("Model reasoning", expanded=True):
            st.markdown(reasoning)

    # --- PENMAN representation ---
    st.subheader("PENMAN representation")
    st.code(amr_string, language="text")
