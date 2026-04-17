"""
agents.py — Agent Layer (Doc Agent → Retrieval Agent → Response Agent)
=======================================================================
Faithful port of the ReAct agent design from the notebook:

  DocAgent        → classify_intent + think_and_plan   (THINK phase)
  RetrievalAgent  → act_retrieve + observe              (ACT + OBSERVE phase)
  ResponseAgent   → synthesize + run_llm_agent_loop     (RESPOND phase)

All heavyweight Groq calls go through llm.groq_call.
All retrieval calls go through retrieval.hybrid_retrieve / build_combined_context.

State is injected by app.py via bind_app_state() after initialisation, which
avoids circular imports while keeping the module fully testable in isolation.
"""

import re
import json
import dataclasses
from typing import Any, Dict, List, Optional

# ── LLM layer imports ──────────────────────────────────────────────────────────
from llm import (
    groq_call, _wait_if_needed, _log_tokens,
    MODEL_FAST, MODEL_SMART,
    FIELD_QUESTIONS, FIELD_KEYWORDS,
    CHAT_SYSTEM,
)

# ── App-state references (populated by bind_app_state) ────────────────────────
# These mirror the globals in app.py.  Call bind_app_state() once models are
# loaded so agents always see the live lists/dicts.
master_data:        list = []
all_pdf_paths:      list = []
all_pdf_texts:      dict = {}
all_pdf_sections:   dict = {}
all_pdf_stores:     dict = {}
emb_model                = None
reranker                 = None


def bind_app_state(state: dict) -> None:
    """
    Inject live app globals into this module so agents see current data.

    Call this in app.py every time the relevant globals change, e.g.:

        import agents
        agents.bind_app_state({
            'master_data':      master_data,
            'all_pdf_paths':    all_pdf_paths,
            'all_pdf_texts':    all_pdf_texts,
            'all_pdf_sections': all_pdf_sections,
            'all_pdf_stores':   all_pdf_stores,
            'emb_model':        emb_model,
            'reranker':         reranker,
        })
    """
    g = globals()
    for k, v in state.items():
        if k in g:
            g[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# DOC AGENT — THINK phase (classify intent + build retrieval plan)
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_SYS = (
    'You classify user intent for a government scheme chatbot.\n'
    'Intent options:\n'
    '  personal_eligibility_check -- user mentions personal details (state, income,\n'
    '                                 caste, course, age) and asks if they qualify.\n'
    '                                 Also use when user mentions any state/region\n'
    '                                 and asks can I apply / am I eligible.\n'
    '  factual_lookup             -- asks for a specific fact: amount, deadline, docs\n'
    '  overview                   -- wants a general summary of a scheme\n'
    '  recommend_scheme           -- asks which scheme is best for their situation\n'
    '  compare_schemes            -- asks to compare multiple schemes\n'
    '  off_topic                  -- zero relevance to any government/welfare scheme\n'
    'KEY RULE: state/income/caste/course + apply/eligible -> personal_eligibility_check.\n'
    'off_topic is ONLY for questions with zero connection to welfare schemes.\n'
    'Examples:\n'
    '  "I am from Kerala, can I apply?" -> personal_eligibility_check, state: Kerala\n'
    '  "My income is 3 lakhs, am I eligible?" -> personal_eligibility_check\n'
    '  "What documents are needed?" -> factual_lookup, field: documents\n'
    '  "What is the weather?" -> off_topic\n'
    'Also extract: entities, target_scheme (else "all"), field\n'
    'Reply as JSON only, no markdown:\n'
    '{"intent":"...","entities":"...","target_scheme":"...","field":"..."}'
)

_PLANNER_SYS = (
    'You are a retrieval planner for a government scheme AI agent.\n'
    'Given intent and user question, output a JSON retrieval plan:\n'
    '{"fields":[...], "keywords":[...], "reason":"..."}\n'
    'fields: list from [Eligibility, Benefits, Documents Required, Application Process,'
    ' Deadline, Implementing Authority, No. of Scholarships, Eligible Courses]\n'
    'keywords: 3-6 short search terms\nNo markdown, JSON only.'
)


class DocAgent:
    """
    THINK phase: understands the user's question and builds a retrieval plan.
    Corresponds to Cell 8 (classify_intent) + Cell 9a (think_and_plan).
    """

    @staticmethod
    def classify_intent(question: str):
        """Returns (intent, entities, target_scheme, field)."""
        from llm import groq_client  # late import — avoids stale reference
        if not groq_client:
            return 'factual_lookup', '', 'all', ''
        _wait_if_needed(question, 80)
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL_FAST,
                messages=[{'role': 'system', 'content': _INTENT_SYS},
                          {'role': 'user',   'content': question}],
                temperature=0.0, max_tokens=80,
            )
            _log_tokens(question, 80)
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json|```$', '', raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            return (
                parsed.get('intent',        'factual_lookup'),
                parsed.get('entities',      ''),
                parsed.get('target_scheme', 'all'),
                parsed.get('field',         ''),
            )
        except Exception:
            return 'factual_lookup', '', 'all', ''

    @staticmethod
    def think_and_plan(question: str, intent: str,
                       entities: str, target_scheme: str, field: str) -> dict:
        """Produces {fields, keywords, reason} retrieval plan."""
        from llm import groq_client
        if intent in ('recommend_scheme', 'compare_schemes', 'overview'):
            return {
                'fields':   list(FIELD_QUESTIONS.keys()),
                'keywords': (entities.split(', ') if entities else []) + ['scheme', 'eligibility'],
                'reason':   f'Full data needed for {intent}',
            }
        _wait_if_needed(question, 120)
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {'role': 'system', 'content': _PLANNER_SYS},
                    {'role': 'user',   'content':
                        f'Question: {question}\nIntent: {intent}\n'
                        f'Entities: {entities}\nField hint: {field}'},
                ],
                temperature=0.0, max_tokens=120,
            )
            _log_tokens(question, 120)
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json|```$', '', raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception:
            return {
                'fields':   [field or 'Eligibility'],
                'keywords': entities.split(', ') if entities else [question[:40]],
                'reason':   'fallback plan',
            }

    @staticmethod
    def resolve_scope(target_scheme: str) -> list:
        """Narrows the list of PDFs to search based on intent target."""
        if not all_pdf_paths:
            return []
        if not target_scheme or target_scheme.lower() in ('all', '', 'any', 'each'):
            return all_pdf_paths
        ts     = target_scheme.lower()
        scores = []
        for p in all_pdf_paths:
            import os
            fname = os.path.basename(p).lower().replace('.pdf', '').replace('_', ' ')
            score = sum(1 for w in ts.split() if w in fname)
            scores.append((score, p))
        scores.sort(reverse=True)
        return [scores[0][1]] if scores[0][0] > 0 else all_pdf_paths


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL AGENT — ACT + OBSERVE phase
# ══════════════════════════════════════════════════════════════════════════════

_FIELD_TRIGGERS = {
    'Eligibility':            ['eligible','eligibility','qualify','qualification','who can','criteria',
                               'requirement','restriction','conditions','income','caste','age','am i',
                               'can i','apply','who is','entitled','minimum','maximum','limit'],
    'Benefits':               ['benefit','benefits','amount','money','scholarship','stipend','grant',
                               'rupees','how much','financial','fund','award','prize','assistance',
                               'support','payment','allowance','quantum','value'],
    'Documents Required':     ['document','documents','certificate','certificates','proof','paper',
                               'attach','attachment','submit','need to bring','required','enclosure'],
    'Application Process':    ['apply','application','how to','procedure','steps','step','process',
                               'portal','fill','form','online','offline','register','method'],
    'Deadline':               ['deadline','last date','when','date','closing','by when','timeline'],
    'Implementing Authority': ['authority','ministry','department','who runs','administered','contact',
                               'office','directorate','nodal','implementing','managed by'],
    'No. of Scholarships':    ['how many','seats','quota','number of','total slots','available','count'],
    'Eligible Courses':       ['course','courses','programme','degree','stream','subject',
                               'field of study','discipline','branch','undergraduate','postgraduate'],
}


class RetrievalAgent:
    """
    ACT + OBSERVE phase: fetches relevant context then judges sufficiency.
    Corresponds to Cell 9a (act_retrieve) + Cell 10 (observe).
    """

    @staticmethod
    def select_fields(question: str) -> list:
        q_lower = question.lower()
        q_words = set(q_lower.split())
        selected = []
        for field, kws in _FIELD_TRIGGERS.items():
            matched = any(kw in q_lower or set(kw.split()) & q_words for kw in kws)
            if matched:
                selected.append(field)
        return selected if selected else list(_FIELD_TRIGGERS.keys())

    @staticmethod
    def excel_retrieve(query: str, top_k: int = None, target_scheme: str = 'all') -> list:
        """
        Retrieval from master Excel KB.
        If target_scheme is specified, filters to matching schemes first.
        If no match found, falls back to all schemes (so user gets a response).
        """
        fields = RetrievalAgent.select_fields(query)
        for f in ('Eligibility', 'Benefits'):
            if f not in fields:
                fields.append(f)

        # ── Scheme filtering: score each row against target_scheme ────────────
        def _scheme_score(row, target: str) -> int:
            """Returns match score between row scheme name and target string."""
            if not target or target.lower() in ('all', '', 'any', 'each'):
                return 1  # no filter
            name = str(row.get('Scheme Name', row.get('PDF', ''))).lower()
            pdf  = str(row.get('PDF', '')).lower().replace('.pdf', '').replace('_', ' ')
            t    = target.lower()
            # exact substring match
            if t in name or t in pdf:
                return 10
            # word-level match
            t_words = [w for w in t.split() if len(w) > 2]
            score   = sum(1 for w in t_words if w in name or w in pdf)
            return score

        # Score all rows; filter to best-matching scheme if target given
        if target_scheme and target_scheme.lower() not in ('all', '', 'any', 'each'):
            scored = [(row, _scheme_score(row, target_scheme)) for row in master_data]
            best_score = max(s for _, s in scored) if scored else 0
            if best_score > 0:
                rows_to_use = [row for row, s in scored if s == best_score]
            else:
                rows_to_use = master_data  # fallback: no match found, return all
        else:
            rows_to_use = master_data

        observations = []
        for row in rows_to_use:
            scheme_name = row.get('Scheme Name', row.get('PDF', 'Unknown'))
            fname       = row.get('PDF', '')
            for field in fields:
                val = str(row.get(field, '')).strip()
                if val and val.lower() != 'not available' and len(val) > 10:
                    observations.append({
                        'scheme': scheme_name, 'pdf': fname,
                        'field': field, 'text': val[:800], 'source': 'excel_kb',
                    })
        return observations

    @staticmethod
    def act_retrieve(plan: dict, pdf_paths_in_scope: list, question: str) -> list:
        """
        Hybrid retrieval: cached Excel fields → section lookup → FAISS+BM25 RAG.
        Corresponds to Cell 9a act_retrieve().
        """
        from retrieval import build_combined_context

        observations    = []
        fields_needed   = plan.get('fields', list(FIELD_QUESTIONS.keys()))
        kws             = plan.get('keywords', [])

        if not pdf_paths_in_scope and master_data:
            return RetrievalAgent.excel_retrieve(question)

        for pdf_path in pdf_paths_in_scope:
            import os
            fname       = os.path.basename(pdf_path)
            master_row  = next((r for r in master_data if r['PDF'] == fname), {})
            # Use proper Scheme Name from Excel, fall back to filename-derived name
            scheme_name = master_row.get('Scheme Name') or fname.replace('.pdf', '').replace('_', ' ')

            for field in fields_needed:
                cached = master_row.get(field, '').strip()
                if cached and cached.lower() != 'not available' and len(cached) > 20:
                    observations.append({
                        'scheme': scheme_name, 'pdf': fname,
                        'field': field, 'text': cached, 'source': 'cached',
                    })
                elif pdf_path in all_pdf_stores:
                    fkws = FIELD_KEYWORDS.get(field, [field.lower()]) + kws
                    ctx  = build_combined_context(
                        query=question,
                        pdf_path=pdf_path,
                        all_pdf_texts=all_pdf_texts,
                        all_pdf_sections=all_pdf_sections,
                        all_pdf_stores=all_pdf_stores,
                        emb_model=emb_model,
                        reranker=reranker,
                        field_keywords=fkws[:6],
                    )
                    if ctx.strip():
                        observations.append({
                            'scheme': scheme_name, 'pdf': fname,
                            'field': field, 'text': ctx[:800], 'source': 'rag',
                        })
        return observations

    @staticmethod
    def observe(observations: list, intent: str, plan: dict):
        """
        Checks whether retrieved observations are sufficient to answer.
        Returns (sufficient: bool, missing_fields: list).
        """
        if not observations:
            return False, plan.get('fields', [])
        retrieved = {obs['field'] for obs in observations if len(obs['text']) > 30}
        needed    = set(plan.get('fields', []))
        missing   = list(needed - retrieved)
        if intent == 'personal_eligibility_check' and 'Eligibility' not in retrieved:
            return False, ['Eligibility']
        sufficient = len(retrieved) >= max(1, len(needed) * 0.6)
        return sufficient, missing


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE AGENT — SYNTHESIZE phase
# ══════════════════════════════════════════════════════════════════════════════

_SYS_FACTUAL = (
    'You are an assistant for Indian government welfare schemes loaded from a master Excel on Google Drive.\n'
    'ABSOLUTE RULES — no exceptions:\n'
    '1. Answer ONLY from facts EXPLICITLY present in the CONTEXT block below. NEVER use training knowledge.\n'
    '2. If the context does not contain the answer, respond ONLY with:\n'
    '   "This information is not available in the uploaded scheme documents."\n'
    '3. NEVER mention, describe, or hint at schemes not present in the context.\n'
    '4. NEVER invent amounts, dates, eligibility criteria, or document names.\n'
    '5. First line: direct one-sentence answer.\n'
    '6. Then: bullet points with exact specifics (amounts, dates, names, steps) from context.\n'
    '7. End with: Source: [scheme name from context]\n'
    '8. If asked about a scheme not in context: "This scheme is not in our uploaded documents."'
)
_SYS_ELIGIBILITY = (
    'You are checking if a user qualifies for a scheme FROM THE CONTEXT ONLY.\n'
    'ABSOLUTE RULES — no exceptions:\n'
    '1. Read ONLY the eligibility criteria stated in the CONTEXT block. NEVER use training knowledge.\n'
    '2. If context has no eligibility info, say: "Eligibility criteria not found in uploaded documents."\n'
    '3. NEVER invent criteria. If a criterion is not in the context, say UNKNOWN.\n'
    '4. Give a clear VERDICT first: ELIGIBLE / NOT ELIGIBLE / NEED MORE INFO\n'
    '5. List each criterion from context and whether the user meets it (YES / NO / UNKNOWN).\n'
    '6. State what additional info is needed if verdict is NEED MORE INFO.'
)
_SYS_RECOMMEND = (
    'You are recommending Indian government schemes to a user BASED ON CONTEXT ONLY.\n'
    'ABSOLUTE RULES — no exceptions:\n'
    '1. ONLY recommend schemes explicitly listed in the CONTEXT block. NEVER suggest others.\n'
    '2. NEVER use training knowledge about any scheme.\n'
    '3. Rank schemes: BEST FIT first based on context data.\n'
    '4. For each scheme state: why it fits, exact eligibility criteria, exact benefit amount from context.\n'
    '5. If no scheme in context matches, say: "No matching scheme found in the uploaded documents."'
)
_SYS_COMPARE = (
    'You are comparing Indian government schemes BASED ON CONTEXT ONLY.\n'
    'ABSOLUTE RULES — no exceptions:\n'
    '1. ONLY compare schemes present in the CONTEXT block. NEVER add outside knowledge.\n'
    '2. NEVER invent values. If a field is missing from context, write: Not available in documents.\n'
    '3. Use a side-by-side structure.\n'
    '4. Compare: Eligibility, Benefits, Documents, Deadline, Authority.\n'
    '5. Highlight the most important differences using only context facts.'
)


class ResponseAgent:
    """
    RESPOND phase: synthesises retrieved observations into a final answer.
    Corresponds to Cell 10 (synthesize) in the notebook.
    """

    @staticmethod
    def synthesize(question: str, intent: str,
                   entities: str, observations: list) -> str:
        if not observations:
            return 'No relevant information found in the uploaded scheme documents.'

        # ── ANTI-HALLUCINATION GUARD: drop weak/empty observations ──────────
        observations = [o for o in observations if len(o.get('text', '').strip()) > 30]
        if not observations:
            return 'This information is not available in the uploaded scheme documents.'

        schemes_seen = list(dict.fromkeys(obs['scheme'] for obs in observations))

        # ── Build context strictly from retrieved observations only ──────────
        ctx_parts    = []
        for scheme in schemes_seen:
            s_obs = [obs for obs in observations if obs['scheme'] == scheme]
            ctx   = f'--- SCHEME: {scheme} ---\n'
            for obs in s_obs:
                ctx += f'[{obs["field"]}]\n{obs["text"][:500]}\n\n'
            ctx_parts.append(ctx)
        context = '\n'.join(ctx_parts)[:3500]

        # ── Inject available scheme list so LLM knows what NOT to invent ────
        all_scheme_names = list(dict.fromkeys(
            str(r.get('Scheme Name', r.get('PDF', ''))) for r in master_data
        )) if master_data else schemes_seen
        scheme_list_note = (
            f'\n\n[ONLY THESE SCHEMES EXIST IN OUR DATABASE: {", ".join(all_scheme_names)}]\n'
            '[DO NOT answer about any scheme not listed above or not present in context above.]'
        )

        if intent == 'personal_eligibility_check':
            sys_p = _SYS_ELIGIBILITY
            task  = (
                f'User message: "{question}"\nUser details: {entities or "see message above"}\n\n'
                'Check eligibility step by step and give a verdict:'
            )
        elif intent == 'recommend_scheme':
            sys_p = _SYS_RECOMMEND
            task  = (
                f'User request: "{question}"\nUser profile: {entities or "general applicant"}\n'
                f'Available schemes: {", ".join(schemes_seen)}\n\nRank and recommend:'
            )
        elif intent == 'compare_schemes':
            sys_p = _SYS_COMPARE
            task  = f'Compare: {", ".join(schemes_seen)}\nUser question: {question}'
        elif intent == 'overview':
            sys_p = _SYS_FACTUAL
            task  = (
                'Give a complete overview covering:\n'
                '1. Purpose\n2. Who is eligible\n3. What benefits (exact amounts)\n'
                '4. Documents required\n5. How to apply\n6. Deadline'
            )
        else:
            sys_p = _SYS_FACTUAL
            task  = f'Question: {question}\n\nAnswer directly and specifically:'

        prompt = (
            f'Context from uploaded scheme documents:\n{context}'
            f'{scheme_list_note}\n\n{task}'
        )
        return groq_call(prompt, model=MODEL_SMART, max_tokens=600, system=sys_p)


# ══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY (Cell 9b)
# ══════════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY = {
    'speech_tool': {
        'name': 'speech_tool',
        'description': (
            'Converts raw audio bytes into text. '
            'Returns the recognized text and a confidence score (0.0-1.0). '
            'Use this FIRST when user_input is audio/voice. '
            'If confidence < 0.5, consider calling it again or asking clarification.'
        ),
        'parameters': {'lang': 'Language code: "kn" for Kannada, "hi" for Hindi (default "kn")'},
    },
    'translation_tool': {
        'name': 'translation_tool',
        'description': (
            'Translates Kannada or Hindi text into English. '
            'Use this when recognized text is in a non-English language. '
            'Skip this tool if the text is already in English.'
        ),
        'parameters': {
            'text':     'The native-language text to translate',
            'src_lang': 'Source language: "kn" Kannada, "hi" Hindi',
        },
    },
    'scheme_matching_tool': {
        'name': 'scheme_matching_tool',
        'description': (
            'Searches all loaded government scheme documents for information '
            'relevant to an English query. Returns top matching text chunks. '
            'Use this once you have a clear English question. '
            'Returns success=False if no scheme matches the query.'
        ),
        'parameters': {'query': 'The English-language question', 'top_k': 'Number of top matches (default 3)'},
    },
    'excel_query_tool': {
        'name': 'excel_query_tool',
        'description': (
            'Queries the master Excel knowledge base (extracted scheme data). '
            'Use this when NO PDF documents are loaded in the session. '
            'You can specify which fields to retrieve and optionally filter by scheme name. '
            'Returns structured field-value pairs for all matching schemes.'
        ),
        'parameters': {
            'fields':  'List of fields e.g. [Eligibility, Benefits, ...]',
            'schemes': 'Optional scheme name filter []',
            'query':   'The user question',
        },
    },
}

EXCEL_TOOL_FIELDS = list(FIELD_QUESTIONS.keys())


def _format_tools_for_prompt() -> str:
    lines = []
    for t in TOOL_REGISTRY.values():
        params = ', '.join(f'{k}: {v}' for k, v in t['parameters'].items())
        lines.append(f'  - {t["name"]}({params})\n    {t["description"]}')
    return '\n'.join(lines)


def excel_query_tool(fields=None, schemes=None, query='', target_scheme='all') -> dict:
    if not master_data:
        return {'success': False, 'data': [], 'schemes_found': [], 'error': 'master_data is empty.'}
    wanted_fields = list(fields) if fields else list(EXCEL_TOOL_FIELDS)
    for f in ('Eligibility', 'Benefits'):
        if f not in wanted_fields:
            wanted_fields = [f] + wanted_fields

    # Build the effective scheme filter: prefer explicit 'schemes' arg,
    # then target_scheme extracted from intent classifier
    effective_target = 'all'
    if schemes:
        effective_target = ' '.join(schemes)
    elif target_scheme and target_scheme.lower() not in ('all', '', 'any', 'each'):
        effective_target = target_scheme

    if effective_target and effective_target.lower() not in ('all', '', 'any', 'each'):
        # Score all rows and keep only best-matching ones
        def _score(row):
            name = str(row.get('Scheme Name', row.get('PDF', ''))).lower()
            pdf  = str(row.get('PDF', '')).lower().replace('.pdf', '').replace('_', ' ')
            t    = effective_target.lower()
            if t in name or t in pdf:
                return 10
            t_words = [w for w in t.split() if len(w) > 2]
            return sum(1 for w in t_words if w in name or w in pdf)

        scored = [(row, _score(row)) for row in master_data]
        best   = max(s for _, s in scored) if scored else 0
        rows   = [row for row, s in scored if s == best] if best > 0 else master_data
    else:
        rows = master_data

    data = []
    for row in rows:
        sname = row.get('Scheme Name', row.get('PDF', 'Unknown'))
        for field in wanted_fields:
            val = str(row.get(field, 'Not available')).strip()
            if val and val.lower() != 'not available' and len(val) > 5:
                data.append({'scheme': sname, 'field': field, 'value': val})
    schemes_found = list(dict.fromkeys(d['scheme'] for d in data))
    return {
        'success': bool(data),
        'data': data,
        'schemes_found': schemes_found,
        'error': None if data else 'No matching data found in Excel KB',
    }


def scheme_matching_tool(query: str, top_k: int = 3) -> dict:
    """High-level tool that runs the full DocAgent→RetrievalAgent pipeline."""
    # Always classify intent first to get target_scheme (works for both paths)
    try:
        intent, entities, target_scheme, field = DocAgent.classify_intent(query)
    except Exception:
        intent, entities, target_scheme, field = 'factual_lookup', '', 'all', ''

    if not all_pdf_paths:
        if not master_data:
            return {'matches': [], 'match_count': 0, 'schemes_found': [],
                    'success': False, 'error': 'No scheme data loaded.'}
        # Pass target_scheme so excel_retrieve filters to the right scheme
        matches       = RetrievalAgent.excel_retrieve(query, top_k, target_scheme=target_scheme)
        schemes_found = list(dict.fromkeys(m['scheme'] for m in matches))
        return {'matches': matches, 'match_count': len(matches),
                'schemes_found': schemes_found, 'success': bool(matches),
                'error': None if matches else 'No matching info'}

    if not query.strip():
        return {'matches': [], 'match_count': 0, 'schemes_found': [],
                'success': False, 'error': 'Empty query'}
    try:
        plan    = DocAgent.think_and_plan(query, intent, entities, target_scheme, field)
        scope   = DocAgent.resolve_scope(target_scheme)
        matches = RetrievalAgent.act_retrieve(plan, scope, query)
        matches.sort(key=lambda x: len(x.get('text', '')), reverse=True)
        top_matches   = matches[:top_k]
        schemes_found = list(dict.fromkeys(m['scheme'] for m in top_matches))
        return {'matches': top_matches, 'match_count': len(top_matches),
                'schemes_found': schemes_found, 'success': bool(top_matches),
                'error': None if top_matches else 'No matching scheme'}
    except Exception as e:
        return {'matches': [], 'match_count': 0, 'schemes_found': [],
                'success': False, 'error': str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT STATE  (Cell 9b)
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class AgentState:
    raw_input:    Any  = None
    input_type:   str  = 'text'
    lang:         str  = 'en'
    observations: list = dataclasses.field(default_factory=list)
    final_answer: str  = ''
    steps:        list = dataclasses.field(default_factory=list)
    done:         bool = False

    def add_observation(self, tool_name: str, result: dict) -> None:
        self.observations.append({'tool': tool_name, 'result': result})

    def add_step(self, decision: dict) -> None:
        self.steps.append(decision)


# ══════════════════════════════════════════════════════════════════════════════
# LLM PLANNER  (Cell 9b) — decides next tool or final answer
# ══════════════════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM = """\
You are an intelligent agent controller for a government scheme assistant.
You decide which tool to call next (or give a final answer) based on:
  1. The user's original input
  2. What tools are available
  3. What has already been observed

Available tools:
{tools}

DECISION RULES:
- If input is audio/voice and no speech result yet -> call speech_tool
- If speech confidence < 0.5 -> call speech_tool again (retry) or use low-confidence text
- If text is non-English and not yet translated -> call translation_tool
- If you have an English question and no scheme matches yet -> call scheme_matching_tool
- If scheme_matching_tool returned success=False -> give final_answer explaining no scheme found
- If you have scheme matches -> give final_answer (the synthesize function handles generation)
- If text is already English -> skip translation_tool entirely
- If no PDFs loaded but master_data available -> use excel_query_tool instead of scheme_matching_tool

Respond ONLY with valid JSON, no markdown:
{{
  "thought": "your reasoning here (1-2 sentences)",
  "action": "call_tool" OR "final_answer",
  "tool_name": "speech_tool|translation_tool|scheme_matching_tool|excel_query_tool",
  "tool_args": {{ ... }},
  "answer": "..."
}}"""

_EXCEL_AGENT_PLANNER = """\
You are an intelligent agent for a government scheme assistant.
The user's question must be answered from a structured Excel knowledge base.

Available tool:
  - excel_query_tool(fields, schemes, query)
    Retrieves field-value pairs from the master Excel KB.
    fields: list from [Eligibility, Benefits, Documents Required,
            Application Process, Deadline, Implementing Authority,
            No. of Scholarships, Eligible Courses]
    schemes: optional filter by scheme name ([] for all)
    query: the user question

DECISION RULES:
- First call: pick only the fields relevant to the question.
- If result has success=True and sufficient data -> give final_answer
- If result missing key fields -> call excel_query_tool again with additional fields
- If success=False -> give final_answer explaining no data found
- Never call excel_query_tool more than 2 times

Respond ONLY with valid JSON, no markdown:
{
  "thought": "your 1-2 sentence reasoning",
  "action": "call_tool" OR "final_answer",
  "tool_name": "excel_query_tool",
  "tool_args": {"fields": [...], "schemes": [...], "query": "..."},
  "answer": "..."
}"""


def llm_decide_next_action(user_input_desc: str, observations: list,
                            lang: str, max_steps_left: int) -> dict:
    from llm import groq_client
    tools_str = _format_tools_for_prompt()
    system    = _PLANNER_SYSTEM.format(tools=tools_str)
    obs_text  = 'None yet.' if not observations else ''
    for i, obs in enumerate(observations, 1):
        obs_text += (
            f'\nStep {i} — called {obs["tool"]}:\n'
            f'  result: {json.dumps(obs["result"], ensure_ascii=False)[:300]}'
        )
    user_msg = (
        f'User input: {user_input_desc}\nLanguage hint: {lang}\n'
        f'Steps remaining: {max_steps_left}\n\nObservations so far:\n{obs_text}\n\nWhat should I do next?'
    )
    _wait_if_needed(user_msg, 150)
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{'role': 'system', 'content': system},
                      {'role': 'user',   'content': user_msg}],
            temperature=0.0, max_tokens=200,
        )
        _log_tokens(user_msg, 200)
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json|```$', '', raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        for obs in reversed(observations):
            if obs['tool'] == 'translation_tool' and obs['result'].get('success'):
                return {
                    'thought': 'fallback',
                    'action': 'call_tool',
                    'tool_name': 'scheme_matching_tool',
                    'tool_args': {'query': obs['result']['english_text'], 'top_k': 3},
                }
        return {'thought': 'fallback final', 'action': 'final_answer',
                'answer': 'Agent encountered an error. Please try again.'}


def execute_tool(tool_name: str, tool_args: dict, state: AgentState) -> dict:
    """Dispatch a tool call and return its result dict."""
    if tool_name == 'speech_tool':
        if not isinstance(state.raw_input, bytes):
            return {'success': False, 'error': 'No audio bytes in state.'}
        return {'success': True, 'text': str(state.raw_input), 'confidence': 1.0}

    elif tool_name == 'translation_tool':
        text = tool_args.get('text')
        if not text:
            for obs in reversed(state.observations):
                if obs['tool'] == 'speech_tool' and obs['result'].get('success'):
                    text = obs['result']['text']
                    break
        if not text:
            text = str(state.raw_input)
        src_lang = tool_args.get('src_lang', state.lang)
        # Import translate at call-time to avoid circular dependency
        from app import translate_to_english  # noqa: lazy import
        result = translate_to_english(text, src_lang=src_lang)
        return {'success': True, 'english_text': result, 'src_lang': src_lang}

    elif tool_name == 'scheme_matching_tool':
        query = tool_args.get('query')
        if not query:
            for obs in reversed(state.observations):
                if obs['tool'] == 'translation_tool' and obs['result'].get('success'):
                    query = obs['result']['english_text']
                    break
            if not query:
                query = str(state.raw_input)
        return scheme_matching_tool(query, top_k=tool_args.get('top_k', 3))

    elif tool_name == 'excel_query_tool':
        return excel_query_tool(
            fields=tool_args.get('fields', []),
            schemes=tool_args.get('schemes', []),
            query=tool_args.get('query', str(state.raw_input)),
        )
    else:
        return {'success': False, 'error': f'Unknown tool: {tool_name}'}


# ══════════════════════════════════════════════════════════════════════════════
# AGENT LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def run_llm_agent_loop(state: AgentState, max_steps: int = 6) -> AgentState:
    """
    Main ReAct loop: THINK → ACT → OBSERVE → repeat until final answer.
    Corresponds to Cell 9b run_llm_agent_loop().
    """
    if state.input_type == 'voice':
        n = len(state.raw_input) if isinstance(state.raw_input, bytes) else 0
        input_desc = f'Voice/audio input ({n} bytes), language={state.lang}'
    else:
        input_desc = f'Text input: "{str(state.raw_input)[:100]}", language={state.lang}'

    for step in range(1, max_steps + 1):
        decision = llm_decide_next_action(input_desc, state.observations,
                                          state.lang, max_steps - step + 1)
        state.add_step(decision)

        if decision.get('action') == 'final_answer':
            raw_answer  = decision.get('answer', '')
            english_q   = str(state.raw_input)
            matches     = []
            for obs in state.observations:
                if obs['tool'] == 'scheme_matching_tool' and obs['result'].get('success'):
                    matches = obs['result']['matches']
                if obs['tool'] in ('translation_tool', 'excel_query_tool') and obs['result'].get('success'):
                    english_q = obs['result'].get('english_text', english_q)

            if raw_answer and len(raw_answer) > 20:
                state.final_answer = raw_answer
            elif matches:
                intent, entities, _, _ = DocAgent.classify_intent(english_q)
                state.final_answer = ResponseAgent.synthesize(english_q, intent, entities, matches)
            elif any(obs['tool'] == 'excel_query_tool' for obs in state.observations):
                all_data = []
                for obs in state.observations:
                    if obs['tool'] == 'excel_query_tool':
                        all_data.extend(obs['result'].get('data', []))
                if all_data:
                    intent, entities, _, _ = DocAgent.classify_intent(english_q)
                    obs_list = [{'scheme': d['scheme'], 'field': d['field'],
                                 'text': d['value'], 'source': 'excel_agent', 'pdf': ''}
                                for d in all_data]
                    state.final_answer = ResponseAgent.synthesize(english_q, intent, entities, obs_list)
                else:
                    state.final_answer = raw_answer or 'No scheme information found.'
            else:
                state.final_answer = raw_answer or 'No scheme information found.'

            state.done = True
            break

        tool_name = decision.get('tool_name', '')
        tool_args = decision.get('tool_args', {})
        if not tool_name or tool_name not in TOOL_REGISTRY:
            state.final_answer = 'Agent chose an invalid tool. Please try again.'
            state.done = True
            break

        result = execute_tool(tool_name, tool_args, state)
        state.add_observation(tool_name, result)
    else:
        if not state.final_answer:
            state.final_answer = 'The agent could not complete within the step limit. Please try again.'
        state.done = True

    return state


def run_llm_agentic_workflow(user_input: str,
                              input_type: str = 'text',
                              lang: str = 'en') -> str:
    """Entry point for text/voice queries routed through the full ReAct loop."""
    state = AgentState(raw_input=user_input, input_type=input_type, lang=lang)
    state = run_llm_agent_loop(state, max_steps=6)
    return state.final_answer


def run_excel_agent_loop(question: str, max_steps: int = 4) -> str:
    """
    Dedicated agent loop for Excel-KB-only queries (no PDFs loaded).
    Corresponds to Cell 11b run_excel_agent_loop().
    """
    from llm import groq_client

    # Extract target_scheme immediately so all retrieval is scoped correctly
    try:
        _, entities_pre, target_scheme_pre, _ = DocAgent.classify_intent(question)
    except Exception:
        entities_pre, target_scheme_pre = '', 'all'

    @dataclasses.dataclass
    class ExcelAgentState:
        question:     str  = ''
        observations: list = dataclasses.field(default_factory=list)
        steps:        list = dataclasses.field(default_factory=list)
        final_answer: str  = ''
        done:         bool = False

    state = ExcelAgentState(question=question)

    for step in range(1, max_steps + 1):
        obs_text = 'None yet.'
        if state.observations:
            obs_text = ''
            for i, obs in enumerate(state.observations, 1):
                r       = obs['result']
                snippet = (f"success={r['success']}  schemes={r.get('schemes_found', [])}  "
                           f"data_rows={len(r.get('data', []))}  ")
                if r.get('data'):
                    snippet += '  sample: ' + json.dumps(r['data'][:2], ensure_ascii=False)[:200]
                obs_text += f'\nStep {i} called excel_query_tool:\n  {snippet}'

        user_msg = (
            f'User question: {question}\n\nObservations so far:\n{obs_text}\n\n'
            f'Steps remaining: {max_steps - step + 1}\nWhat should I do next?'
        )
        _wait_if_needed(user_msg, 200)
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL_FAST,
                messages=[{'role': 'system', 'content': _EXCEL_AGENT_PLANNER},
                          {'role': 'user',   'content': user_msg}],
                temperature=0.0, max_tokens=250,
            )
            _log_tokens(user_msg, 200)
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r'^```json|```$', '', raw, flags=re.MULTILINE).strip()
            decision = json.loads(raw)
        except Exception:
            decision = {'thought': 'fallback', 'action': 'call_tool',
                        'tool_name': 'excel_query_tool',
                        'tool_args': {'fields': [], 'schemes': [], 'query': question}}

        state.steps.append(decision)

        if decision.get('action') == 'final_answer':
            raw_ans = decision.get('answer', '')
            if raw_ans and len(raw_ans) > 20:
                state.final_answer = raw_ans
            else:
                all_data = []
                for obs in state.observations:
                    all_data.extend(obs['result'].get('data', []))
                if all_data:
                    intent, entities, _, _ = DocAgent.classify_intent(question)
                    obs_list = [{'scheme': d['scheme'], 'field': d['field'],
                                 'text': d['value'], 'source': 'excel_agent', 'pdf': ''}
                                for d in all_data]
                    # Pass available scheme names so LLM knows scope
                    scheme_names = list(dict.fromkeys(
                        str(r.get('Scheme Name', r.get('PDF', ''))) for r in master_data
                    ))
                    scoped_q = (question +
                                f'\n\n[ONLY ANSWER ABOUT SCHEMES IN DATABASE: {", ".join(scheme_names)}]'
                                '\n[IF SCHEME NOT IN LIST ABOVE, SAY: Not in our uploaded documents]')
                    state.final_answer = ResponseAgent.synthesize(scoped_q, intent, entities, obs_list)
                else:
                    # No data found — give a clear refusal, not hallucination
                    scheme_names = list(dict.fromkeys(
                        str(r.get('Scheme Name', r.get('PDF', ''))) for r in master_data
                    ))
                    state.final_answer = (
                        'This information is not available in the uploaded scheme documents.\n\n'
                        f'Schemes currently in our database: {", ".join(scheme_names) if scheme_names else "None loaded yet."}'
                    )
            state.done = True
            break

        tool_args = decision.get('tool_args', {})
        if 'query' not in tool_args:
            tool_args['query'] = question
        result = excel_query_tool(
            fields=tool_args.get('fields', []),
            schemes=tool_args.get('schemes', []),
            query=tool_args.get('query', question),
            target_scheme=target_scheme_pre,  # always scope to detected scheme
        )
        state.observations.append({'tool': 'excel_query_tool', 'result': result})
    else:
        all_data = []
        for obs in state.observations:
            all_data.extend(obs['result'].get('data', []))
        if all_data:
            intent, entities, _, _ = DocAgent.classify_intent(question)
            obs_list = [{'scheme': d['scheme'], 'field': d['field'],
                         'text': d['value'], 'source': 'excel_agent', 'pdf': ''}
                        for d in all_data]
            scheme_names = list(dict.fromkeys(
                str(r.get('Scheme Name', r.get('PDF', ''))) for r in master_data
            ))
            scoped_q = (question +
                        f'\n\n[ONLY ANSWER ABOUT SCHEMES IN DATABASE: {", ".join(scheme_names)}]'
                        '\n[IF SCHEME NOT IN LIST ABOVE, SAY: Not in our uploaded documents]')
            state.final_answer = ResponseAgent.synthesize(scoped_q, intent, entities, obs_list)
        else:
            scheme_names = list(dict.fromkeys(
                str(r.get('Scheme Name', r.get('PDF', ''))) for r in master_data
            ))
            state.final_answer = (
                'This information is not available in the uploaded scheme documents.\n\n'
                f'Schemes currently in our database: {", ".join(scheme_names) if scheme_names else "None loaded yet."}'
            )
        state.done = True

    return state.final_answer


# ══════════════════════════════════════════════════════════════════════════════
# SMART CHAT — top-level entry point (Cell 11b)
# ══════════════════════════════════════════════════════════════════════════════

def smart_chat(question: str) -> str:
    """
    Routes question to the appropriate agent loop:
      - PDFs loaded  → full ReAct loop (DocAgent → RetrievalAgent → ResponseAgent)
      - Excel-only   → Excel agent loop
      - Nothing      → helpful error
    """
    from llm import groq_client
    from logger import log

    if not groq_client:
        return 'Error: Groq API key not configured.'

    log('smart_chat', f'Q: {question[:80]}')

    if all_pdf_paths:
        log('smart_chat', f'DocAgent→RetrievalAgent→ResponseAgent ({len(all_pdf_paths)} PDFs indexed)')
        return run_llm_agentic_workflow(question, input_type='text', lang='en')
    elif master_data:
        log('smart_chat', f'ExcelAgent ({len(master_data)} schemes in KB)')
        return run_excel_agent_loop(question)
    else:
        return 'No scheme data loaded. Upload scheme PDFs or load the master Excel first.'
