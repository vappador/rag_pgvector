# app/rerankers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple, Dict

@dataclass
class Candidate:
    chunk_id: int
    repo: str
    path: str
    start_line: int
    end_line: int
    content: str
    symbol: Optional[str]
    score: float  # initial hybrid score

def _pack_pair(q: str, c: Candidate) -> str:
    # Short and consistent input format for cross-encoders/LLM judges
    head = f"{c.repo}:{c.path}:{c.start_line}-{c.end_line}"
    sym  = f" [{c.symbol}]" if c.symbol else ""
    return f"Q: {q}\nD: {head}{sym}\n{c.content}"

# ---------------- Cross-encoder (SentenceTransformers) ----------------
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError("Install sentence-transformers to use CrossEncoderReranker") from e
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, cands: List[Candidate], top_k: int = 10) -> List[Candidate]:
        # batch score
        pairs = [(query, c.content) for c in cands]
        ce_scores = self.model.predict(pairs)  # higher is better
        out = []
        for c, s in zip(cands, ce_scores):
            # blend with initial score (hybrid) to stabilize
            blended = 0.3 * c.score + 0.7 * float(s)
            out.append(Candidate(**{**c.__dict__, "score": blended}))
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]

# ---------------- LLM judge via Ollama ----------------
class LLMJudgeReranker:
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model
        try:
            import ollama  # type: ignore
            self._ollama = ollama
        except Exception as e:
            raise RuntimeError("Install ollama python package to use LLMJudgeReranker") from e

    def rerank(self, query: str, cands: List[Candidate], top_k: int = 10) -> List[Candidate]:
        out = []
        for c in cands:
            prompt = (
                "You are ranking code snippets for relevance.\n"
                "Score 0..1 how well the snippet ANSWERS the question, with preference for exact symbols, APIs, and clear code.\n"
                f"Question:\n{query}\n---\nSnippet:\n{_pack_pair(query, c)}\n---\n"
                "Return only a JSON: {\"score\": <float 0..1>}."
            )
            try:
                resp = self._ollama.chat(model=self.model, messages=[{"role":"user","content":prompt}])
                text = resp["message"]["content"].strip()
                score = 0.0
                # ultra-robust parse
                import json as _json, re
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    score = float(_json.loads(m.group(0)).get("score", 0.0))
                blended = 0.4 * c.score + 0.6 * score
                out.append(Candidate(**{**c.__dict__, "score": blended}))
            except Exception:
                out.append(c)  # fall back to original
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]
