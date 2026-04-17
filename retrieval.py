"""
retrieval.py — FAISS + BM25 Hybrid Retrieval Module
Faithful port of Cell 5 (build_store) + Cell 6 (retrieve/rerank helpers).

Separation rationale:
  - app.py       → Flask routes only
  - retrieval.py → all vector-store + keyword-search logic (this file)
  - agents.py    → ReAct agent loops
  - evaluation.py→ offline evaluation matrix
"""

import re
import numpy as np
from typing import Dict, List, Optional, Any

# ── Lazy imports (heavyweight; only loaded when build_store is called) ─────────
def _faiss():
    import faiss
    return faiss

def _bm25():
    from rank_bm25 import BM25Okapi
    return BM25Okapi

# ── Section helpers (Cell 5) ───────────────────────────────────────────────────

def _section_aliases(title: str) -> set:
    aliases = set()
    t = title.lower().strip()
    aliases.add(t)
    for w in t.split():
        if len(w) > 3:
            aliases.add(w)
    words = t.split()
    if len(words) >= 2:
        aliases.add(' '.join(words[:2]))
    return aliases


def split_sections_smart(text: str) -> Dict[str, Any]:
    """
    Splits raw PDF text into named sections using progressively looser
    regex patterns (Cell 5).  Returns {alias: {title, body}} dict.
    """
    patterns = [
        r'(?m)^\s*(\d{1,2})\.\s+([A-Z][A-Z\s&/\-()\d]{4,120})\s*$',
        r'(?m)^\s*(\d{1,2})\.\s+([A-Z][a-zA-Z\s&/\-()\d]{4,120})\s*$',
        r'(?m)^\s*([A-Z][A-Z\s&/\-()]{4,80})\s*$',
        r'(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,8})\s*$',
    ]
    best_matches: list = []
    for pat in patterns:
        matches = list(re.finditer(pat, text))
        if len(matches) >= 2:
            best_matches = matches
            break

    sections: Dict[str, Any] = {}
    if len(best_matches) >= 2:
        for i, m in enumerate(best_matches):
            title = m.group(m.lastindex).strip()
            start = m.start()
            end   = best_matches[i + 1].start() if i + 1 < len(best_matches) else len(text)
            body  = text[start:end].strip()
            if len(body) < 30:
                continue
            for alias in _section_aliases(title):
                sections[alias] = {'title': title, 'body': body}
    return sections


# ── Chunking (Cell 5) ──────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = 300, overlap: int = 80) -> List[str]:
    """Word-level sliding-window chunker (Cell 5)."""
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        chunk = ' '.join(words[start:start + size])
        if len(chunk) > 100:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ── Index builder (Cell 5) ────────────────────────────────────────────────────

def build_store(text: str, emb_model) -> Dict[str, Any]:
    """
    Builds the FAISS + BM25 dual index for a single document (Cell 5).
    Returns a store dict: {chunks, faiss_idx, bm25_idx}
    """
    faiss = _faiss()
    BM25Okapi = _bm25()

    chunks = chunk_text(text)
    embs   = emb_model.encode(chunks, show_progress_bar=False).astype('float32')
    faiss.normalize_L2(embs)

    idx = faiss.IndexFlatIP(embs.shape[1])
    idx.add(embs)

    bm25 = BM25Okapi([c.lower().split() for c in chunks])

    return {'chunks': chunks, 'faiss_idx': idx, 'bm25_idx': bm25}


# ── Section retrieval helpers (Cell 6) ────────────────────────────────────────

def get_section(sections: Dict, *keywords) -> Optional[str]:
    """Finds the best matching section body for a set of keywords (Cell 6)."""
    if not sections:
        return None
    for kw in keywords:
        kl = kw.lower().strip()
        if kl in sections:
            return sections[kl]['body']
        for key, val in sections.items():
            if isinstance(key, str) and kl in val['title'].lower():
                return val['body']
    best_section, best_score = None, 0
    seen_titles: set = set()
    for key, val in sections.items():
        title = val['title']
        if title in seen_titles:
            continue
        seen_titles.add(title)
        body_lower = val['body'].lower()
        score = sum(1 for kw in keywords if kw.lower() in body_lower)
        if score > best_score:
            best_score = score
            best_section = val['body']
    return best_section if best_score > 0 else None


def keyword_slice(text: str, start_kws: list, stop_kws: Optional[list] = None,
                  max_chars: int = 1200) -> Optional[str]:
    """Extracts a slice of text anchored by keyword positions (Cell 6)."""
    tl = text.lower()
    sp = None
    for kw in start_kws:
        p = tl.find(kw.lower())
        if p != -1 and (sp is None or p < sp):
            sp = p
    if sp is None:
        return None
    sp = max(0, sp - 80)
    ep = len(text)
    if stop_kws:
        for kw in stop_kws:
            p = tl.find(kw.lower(), sp + 50)
            if p != -1 and p < ep:
                ep = p
    lines = [ln.strip() for ln in text[sp:ep].splitlines()
             if ln.strip() and len(ln.strip()) > 3]
    return '\n'.join(lines)[:max_chars] or None


# ── Hybrid retrieval (Cell 6) ─────────────────────────────────────────────────

def hybrid_retrieve(query: str, store: Dict, emb_model, top_k: int = 6) -> List[Dict]:
    """
    FAISS (dense cosine) + BM25 (sparse keyword) with Reciprocal Rank Fusion.
    Faithful port of Cell 6 hybrid_retrieve().
    """
    faiss = _faiss()

    q_emb = emb_model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = store['faiss_idx'].search(q_emb, top_k * 2)
    sem = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    raw  = store['bm25_idx'].get_scores(query.lower().split())
    top  = np.argsort(raw)[::-1][:top_k * 2]
    bm25 = [(int(i), float(raw[i])) for i in top]

    # Reciprocal Rank Fusion
    rrf: Dict[int, float] = {}
    for ranked in [sem, bm25]:
        for rank, (idx, _) in enumerate(ranked, 1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

    fused  = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    chunks = store['chunks']
    return [{'text': chunks[i], 'score': round(s, 4)} for i, s in fused]


def rerank(query: str, candidates: List[Dict], reranker, top_k: int = 4) -> List[Dict]:
    """Cross-encoder reranker (Cell 6)."""
    if not candidates:
        return []
    pairs     = [[query, c['text']] for c in candidates]
    ce_scores = reranker.predict(pairs)
    ranked    = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)
    return [{**c, 'ce_score': round(float(sc), 4)} for c, sc in ranked[:top_k]]


def build_context(chunks: List[Dict], max_chars: int = 1500) -> str:
    """Assembles retrieved chunks into a numbered context string (Cell 6)."""
    parts, total = [], 0
    for i, c in enumerate(chunks):
        snippet = c['text'][:600]
        if total + len(snippet) > max_chars:
            break
        parts.append(f'[{i+1}] {snippet}')
        total += len(snippet)
    return '\n\n'.join(parts)


def build_combined_context(query: str, pdf_path: str,
                           all_pdf_texts: Dict, all_pdf_sections: Dict,
                           all_pdf_stores: Dict, emb_model, reranker,
                           field_keywords: Optional[list] = None,
                           max_chars: int = 2000) -> str:
    """
    Combines section-aware lookup + RAG retrieval into a single context string.
    Faithful port of Cell 6 build_combined_context().
    """
    text  = all_pdf_texts[pdf_path]
    secs  = all_pdf_sections.get(pdf_path, {})
    store = all_pdf_stores[pdf_path]
    words = len(text.split())
    parts: List[str] = []

    if words <= 2500:
        return text[:4000]

    kws = field_keywords or query.lower().split()[:5]
    sec_txt = get_section(secs, *kws)
    if not sec_txt:
        sec_txt = keyword_slice(text, kws, max_chars=1000)
    if sec_txt:
        parts.append(f'[SECTION]\n{sec_txt[:900]}')

    retrieved = hybrid_retrieve(query, store, emb_model, top_k=6)
    reranked  = rerank(query, retrieved, reranker, top_k=4)
    rag_ctx   = build_context(reranked, max_chars=1200)
    if rag_ctx:
        parts.append(f'[RAG CHUNKS]\n{rag_ctx}')

    return '\n\n'.join(parts)[:max_chars]
