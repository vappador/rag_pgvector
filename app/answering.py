# app/answering.py
from __future__ import annotations
from typing import List, Dict
import textwrap
import os

from app.rerankers import Candidate

USE_LLM_SYNTH = os.environ.get("USE_LLM_SYNTH", "0") == "1"
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")

def _format_citation(c: Candidate) -> str:
    sym = f" [{c.symbol}]" if c.symbol else ""
    return f"{c.repo}:{c.path}:{c.start_line}-{c.end_line}{sym}"

def _make_context_block(cands: List[Candidate], max_chars: int = 6000) -> str:
    blocks = []
    used = 0
    for c in cands:
        head = _format_citation(c)
        body = c.content.strip()
        block = f"### {head}\n{body}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block); used += len(block)
    return "\n".join(blocks)

def _rule_based_summarize(q: str, cands: List[Candidate]) -> Dict:
    """Deterministic synthesis that quotes + cites sources."""
    if not cands:
        return {"answer": "No relevant code found in the index.", "citations": []}
    # Simple heuristic: prefer chunks where the symbol name appears in the query
    boosted = sorted(
        cands,
        key=lambda c: (1 if (c.symbol and c.symbol.split('#')[-1].lower() in q.lower()) else 0, c.score),
        reverse=True,
    )
    top = boosted[:3]
    parts = []
    cites = []
    for c in top:
        cites.append(_format_citation(c))
        snippet = textwrap.shorten(c.content.strip(), width=600, placeholder="…")
        parts.append(f"- From `{_format_citation(c)}`:\n```\n{snippet}\n```")
    answer = "Here’s what the code shows:\n" + "\n".join(parts)
    return {"answer": answer, "citations": cites}

def synthesize_answer(q: str, cands: List[Candidate]) -> Dict:
    """
    Always provenance-first. If using an LLM, prompt it to quote + cite explicitly.
    """
    if not USE_LLM_SYNTH:
        return _rule_based_summarize(q, cands)

    try:
        import ollama  # type: ignore
        context = _make_context_block(cands)
        prompt = (
            "You are assisting with a codebase Q&A. Use ONLY the provided snippets to answer.\n"
            "Rules:\n"
            "1) If the snippets do not contain the answer, say 'I don’t have enough code context to answer.'\n"
            "2) When you answer, quote small code bits and include a citation after each point using the format repo:path:start-end.\n"
            "3) Do NOT invent files or symbols; do NOT reference external knowledge.\n"
            f"Question: {q}\n\n---\nSnippets:\n{context}\n---\n"
            "Respond with Markdown. Keep it concise and structured."
        )
        resp = ollama.chat(model=LLM_MODEL, messages=[{"role":"user","content":prompt}])
        text = resp["message"]["content"].strip()
        # Optional: enforce at least one known citation string
        known = {_format_citation(c) for c in cands}
        if not any(k in text for k in known):
            # Fall back to rule-based to guarantee citations
            return _rule_based_summarize(q, cands)
        return {"answer": text, "citations": sorted(list(known))}
    except Exception:
        return _rule_based_summarize(q, cands)
