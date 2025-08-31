# app/search.py (example changes)
from typing import List
from app.rerankers import Candidate, CrossEncoderReranker, LLMJudgeReranker

def _to_candidates(rows) -> List[Candidate]:
    # rows include: chunk_id, repo, path, start_line, end_line, content, symbol, hybrid_score
    return [
        Candidate(
            chunk_id=r["chunk_id"],
            repo=r["repo"],
            path=r["path"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            content=r["content"],
            symbol=r.get("symbol"),
            score=float(r["hybrid_score"])
        )
        for r in rows
    ]

def rerank_candidates(query: str, rows, rerank: bool, reranker_kind: str, top_rerank: int):
    cands = _to_candidates(rows)
    if not rerank or not cands:
        return cands[:top_rerank]
    if reranker_kind == "judge":
        rr = LLMJudgeReranker()
    else:
        rr = CrossEncoderReranker()
    return rr.rerank(query, cands, top_k=top_rerank)
