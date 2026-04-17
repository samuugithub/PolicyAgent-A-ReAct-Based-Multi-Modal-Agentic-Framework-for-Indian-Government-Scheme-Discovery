"""
llm.py — LLM / Groq Client Layer
=================================
Owns all Groq API interaction, token-budget management, and
prompt-level utilities extracted from app.py.

Separation rationale:
  app.py    → Flask routes only
  llm.py    → Groq client + token budget + prompt utilities  ← THIS FILE
  retrieval.py → FAISS + BM25 hybrid retrieval
  agents.py → ReAct agent loops (imports from llm + retrieval)
  evaluation.py → offline evaluation matrix
"""

import os
import re
import time
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FAST  = 'llama-3.1-8b-instant'
MODEL_SMART = 'llama-3.3-70b-versatile'

CHAT_SYSTEM = (
    'You are an assistant for Indian government welfare schemes stored in a master Excel file on Google Drive.\n'
    'ABSOLUTE RULES — violating any rule is FORBIDDEN:\n'
    '1. Answer ONLY using facts that are EXPLICITLY present in the CONTEXT block provided to you.\n'
    '2. If the context is empty, missing, or does not contain the answer, you MUST respond with:\n'
    '   "This information is not available in the uploaded scheme documents."\n'
    '   Do NOT attempt to answer from general knowledge.\n'
    '3. NEVER invent, assume, or recall facts from training data about any scheme.\n'
    '4. NEVER answer questions about schemes not present in the context.\n'
    '5. Start with ONE direct sentence answering the question.\n'
    '6. Follow with bullet points (max 6) for details — only facts from context.\n'
    '7. Always state exact values for amounts, dates, seat counts as they appear in the context.\n'
    '8. End with: Source: [scheme name from context]\n'
    '9. Never use vague language like "it may be", "usually", "typically", "generally".\n'
    '10. If asked about a scheme that is NOT in the context, say:\n'
    '    "This scheme is not in the uploaded documents. I can only answer questions about: [list scheme names from context]"'
)

EXTRACT_SYSTEM = (
    'You are a precise government policy document analyst. '
    'Answer ONLY using facts explicitly stated in the Context. '
    'Use short bullet points starting with bullet. '
    'If a field has zero info, reply: Not available'
)

FIELD_QUESTIONS = {
    'Eligibility':            'What are the exact eligibility criteria? List every condition explicitly mentioned.',
    'Benefits':               'What is the exact financial benefit, scholarship amount, or assistance provided? Include all numbers.',
    'Documents Required':     'What documents must be submitted? List only documents explicitly named.',
    'Application Process':    'What are the exact steps to apply? List each step in order.',
    'Deadline':               'What are the exact application dates, deadline, or last date mentioned?',
    'Implementing Authority': 'Which ministry, directorate, or authority implements or administers this scheme?',
    'No. of Scholarships':    'How many scholarships, seats, or awards are available? Give the exact number.',
    'Eligible Courses':       'Which courses, programmes, or fields of study are eligible?',
}

FIELD_KEYWORDS = {
    'Eligibility':            ['eligibility','eligible','who can apply','criteria','conditions','requirements','income limit','qualification'],
    'Benefits':               ['benefits','financial','amount','scholarship','assistance','quantum','award','stipend','grant','allowance','rupees'],
    'Documents Required':     ['documents','enclosures','submission','attach','certificates','proof','papers','required documents','annexure'],
    'Application Process':    ['application','procedure','how to apply','apply','submit','process','steps','method','form','portal','online'],
    'Deadline':               ['deadline','last date','closing date','date','timeline'],
    'Implementing Authority': ['implementing','authority','ministry','directorate','department','director','administered','nodal'],
    'No. of Scholarships':    ['number of scholarship','seats','quota','how many','total'],
    'Eligible Courses':       ['eligible course','programme','stream','degree','master','bachelor','diploma','subjects','disciplines'],
}

# ── Token Budget (Cell 2) ──────────────────────────────────────────────────────
_token_log:  list = []
TOKEN_LIMIT: int  = 9000
WINDOW_SEC:  int  = 62

# ── Groq client (initialised by init_groq) ────────────────────────────────────
groq_client = None
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()


def _est(text: str) -> int:
    """Rough token estimate: chars // 4."""
    return max(1, len(text) // 4)


def _used() -> int:
    """Tokens used in the current rolling window."""
    now = time.time()
    _token_log[:] = [(ts, t) for ts, t in _token_log if now - ts < WINDOW_SEC]
    return sum(t for _, t in _token_log)


def _wait_if_needed(prompt: str, max_new: int = 250) -> None:
    """Block until the token budget has room for this call."""
    est = _est(prompt) + max_new
    while True:
        used = _used()
        if used + est < TOKEN_LIMIT:
            break
        if _token_log:
            wait = WINDOW_SEC - (time.time() - _token_log[0][0]) + 1
        else:
            wait = 1
        time.sleep(max(1, wait))
        now = time.time()
        _token_log[:] = [(ts, t) for ts, t in _token_log if now - ts < WINDOW_SEC]


def _log_tokens(prompt: str, max_new: int) -> None:
    _token_log.append((time.time(), _est(prompt) + max_new))


# ── Client initialisation ──────────────────────────────────────────────────────

def init_groq(api_key: Optional[str] = None):
    """Initialise the Groq client.  Returns (success: bool, message: str)."""
    global groq_client, GROQ_API_KEY
    try:
        from groq import Groq
        key = api_key or GROQ_API_KEY
        if not key:
            return False, 'GROQ_API_KEY not set'
        GROQ_API_KEY = key
        groq_client = Groq(api_key=key)
        return True, 'Groq client initialized'
    except Exception as e:
        return False, str(e)


# ── Core LLM call ─────────────────────────────────────────────────────────────

def groq_call(prompt: str, model: Optional[str] = None,
              max_tokens: int = 350, temperature: float = 0.1,
              system: Optional[str] = None) -> str:
    """Single-turn LLM call with automatic token-budget throttle and retry."""
    if not groq_client:
        return 'Error: Groq client not initialized.'
    model  = model  or MODEL_FAST
    system = system or CHAT_SYSTEM
    _wait_if_needed(prompt, max_tokens)
    try:
        resp = groq_client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user',   'content': prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        _log_tokens(prompt, max_tokens)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if '429' in err:
            time.sleep(15)
            try:
                resp = groq_client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user',   'content': prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                _log_tokens(prompt, max_tokens)
                return resp.choices[0].message.content.strip()
            except Exception as e2:
                return f'Error: {e2}'
        return f'Error: {e}'


# ── Query normaliser ───────────────────────────────────────────────────────────

def normalize_query(question: str) -> str:
    """Rewrite the user's question into a concise policy-document search query."""
    _wait_if_needed(question, 40)
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{
                'role': 'system',
                'content': (
                    'Rewrite the question into a short precise search query (max 12 words) '
                    'using formal policy document terminology. Return only the rewritten query.'
                ),
            }, {'role': 'user', 'content': question}],
            temperature=0.0,
            max_tokens=40,
        )
        _log_tokens(question, 40)
        return resp.choices[0].message.content.strip()
    except Exception:
        return question


# ── Answer cleaner ────────────────────────────────────────────────────────────

def clean_answer(text: str) -> str:
    """Strip boilerplate / hedge phrases from LLM output."""
    if not text:
        return 'Not available'
    BAD = [
        r'not specified', r'not mentioned', r'not provided', r'not available',
        r'no information', r'note:', r'presumably', r'can be inferred',
    ]
    lines = []
    for ln in text.splitlines():
        ln_clean = ln.strip().lstrip('*-').lstrip('\u2022').strip()
        if not ln_clean:
            continue
        if any(re.search(p, ln_clean.lower()) for p in BAD):
            continue
        lines.append(ln_clean)
    result = '\n'.join(lines).strip()
    return result if result else 'Not available'


# Backward-compat alias (app.py used _clean_answer)
_clean_answer = clean_answer
