"""Web UI (bonus): search, chat, bid records, comparison and go/no-go — a thin client over the REST API."""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="RFP Intelligence", layout="wide")


def call(method: str, path: str, timeout: float = 600, **kwargs: Any) -> Any:
    try:
        resp = httpx.request(method, f"{API}{path}", timeout=timeout, **kwargs)
        body = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        st.error(f"API unreachable at {API}: {exc}")
        st.stop()
    if not body.get("success"):
        st.error(body.get("error") or f"HTTP {resp.status_code}")
        st.stop()
    return body["data"]


def show_citations(citations: list[dict[str, Any]]) -> None:
    for c in citations:
        with st.expander(f"{c.get('file')} — page {c.get('page')}"):
            st.write(c.get("quote") or c.get("text", ""))


health = call("GET", "/health", timeout=30)
with st.sidebar:
    st.header("RFP Intelligence")
    st.caption(f"API: {API}")
    st.write("LLM configured:", "✅" if health["llm_configured"] else "❌ set LLM_* in .env")
    st.write("Indexed bids:", ", ".join(health["bids"]) or "none")
    folder = st.text_input("Index a bid folder (under DATA_ROOT)", placeholder="Bid1")
    if st.button("Index") and folder:
        st.json(call("POST", "/index", json={"folder": folder}))

tab_search, tab_chat, tab_bids, tab_compare = st.tabs(["🔎 Search", "💬 Ask", "📄 Bid records", "⚖️ Compare / Go-No-Go"])

with tab_search:
    q = st.text_input("Query", placeholder="bid bond requirement")
    c1, c2, c3, c4, c5 = st.columns(5)
    bid = c1.selectbox("Bid", ["(all)"] + health["bids"])
    doc_type = c2.selectbox("Doc type", ["(any)", "rfp", "addendum", "specs", "affidavit", "bid_page", "other"])
    mode = c3.selectbox("Mode", ["hybrid", "vector", "bm25"])
    top_k = c4.number_input("Top k", 1, 50, 5)
    rerank = c5.checkbox("Rerank", True)
    if q:
        params: dict[str, Any] = {"q": q, "top_k": top_k, "mode": mode, "rerank": rerank}
        if bid != "(all)":
            params["bid_id"] = bid
        if doc_type != "(any)":
            params["doc_type"] = doc_type
        data = call("GET", "/search", params=params)
        st.caption(f"{len(data['results'])} results in {data['latency_ms']} ms")
        for i, r in enumerate(data["results"], 1):
            score = r["rerank_score"] if r["rerank_score"] is not None else r["score"]
            with st.expander(f"[{i}] {score:.3f} · {r['bid_id']} · {r['citation']['file']} p.{r['citation']['page']}"
                             f" · {r['doc_type']}"):
                st.text(r["text"])

with tab_chat:
    st.session_state.setdefault("history", [])
    for turn in st.session_state.history:
        st.chat_message("user").write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            st.caption(f"answered by {turn['answered_by']} · bids {', '.join(turn['bids'])}")
            show_citations(turn["citations"])
    if question := st.chat_input("Ask about the bids, e.g. 'What is the due date for Bid1 after all addendums?'"):
        st.chat_message("user").write(question)
        with st.spinner("Agents working..."):
            st.session_state.history.append(call("POST", "/ask", json={"question": question}))
        st.rerun()

with tab_bids:
    bid_sel = st.selectbox("Bid", health["bids"], key="bid_record")
    if bid_sel and st.button(f"Run extraction for {bid_sel}"):
        with st.spinner("Extracting (multi-agent run)..."):
            run = call("POST", "/extract", json={"folder": bid_sel})
        st.success(f"Run {run['run']['run_id']}: {run['record']['validation']}")
    if bid_sel and call("GET", "/bids")[bid_sel]["extracted"]:
        rec = call("GET", f"/bids/{bid_sel}/record")
        st.subheader("Summary")
        st.write(rec["fields"].get("Bid Summary", {}).get("value"))
        st.write("Validation:", rec["validation"])
        rows = [{"field": k, "value": json.dumps(v["value"]) if isinstance(v["value"], (dict, list)) else v["value"],
                 "confidence": v["confidence"],
                 "sources": "; ".join(f"{s['file']} p.{s['page']}" for s in v["sources"]),
                 "notes": v["notes"]} for k, v in rec["fields"].items() if k != "Bid Summary"]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.subheader("Addendum changes")
        st.dataframe(pd.DataFrame([{**{k: c[k] for k in ("addendum_number", "field", "description")},
                                    "source": f"{c['source']['file']} p.{c['source']['page']}"}
                                   for c in rec["addendum_changes"]]), hide_index=True)
        if rec["processing_errors"]:
            st.warning("\n".join(rec["processing_errors"]))
        st.download_button("Download JSON", json.dumps(rec, indent=2), f"{bid_sel}.json")

with tab_compare:
    aspect = st.text_input("Compare aspect (optional)", placeholder="warranty requirements")
    if st.button("Compare all bids"):
        data = call("POST", "/compare", json={"question": aspect or None})
        if "markdown" in data:
            st.markdown(data["markdown"])
        else:
            st.write(data["answer"])
            show_citations(data["citations"])
    gng_bid = st.selectbox("Go / No-Go for", health["bids"], key="gng")
    if gng_bid and st.button("Recommend"):
        res = call("POST", "/go-no-go", json={"bid_id": gng_bid})
        st.subheader(res["decision"])
        st.write(res["explanation"])
        st.dataframe(pd.DataFrame(res["rules"]), hide_index=True)
