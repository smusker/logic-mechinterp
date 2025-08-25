# gpt_oss_experiment_utilities.py
"""
Utilities for parsing numbers, JSON, and robustly removing/altering answer mentions
from reasoning chains. Imported by gpt_oss_experiment.py.
"""

import re
import json
import random
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI

# ---------- Shared regex & small helpers ----------

NUM_RE = re.compile(r"(-?\d{1,3}(?:,\d{3})+|-?\d+(?:\.\d+)?)")

def first_int(text: str) -> Optional[int]:
    if not text:
        return None
    matches = NUM_RE.findall(text)
    if not matches:
        return None
    token = matches[-1].replace(",", "")
    try:
        return int(float(token))
    except Exception:
        return None

def json_from_text_maybe(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def replace_last_occurrence(text: str, old: str, new: str) -> str:
    idx = text.rfind(old)
    if idx == -1:
        return text
    return text[:idx] + new + text[idx + len(old):]

# ---------- Robust multi-mention answer stripping ----------

# Matches lines like: "Answer: 1579", "Therefore final result is 1272", etc.
ANSWER_CLAUSE_PATTERN = re.compile(
    r"""(?isx)
    (^|\n)                                   # start of text or new line
    [^\S\r\n]*                               # optional indentation
    (?:so\s+)?(?:therefore[,:\s]+)?(?:final\s+)?(?:answer|result|sum|total|output|final)\s*
    (?:is|=|:)\s*                            # the 'is' / '=' / ':' connector
    (.+?)\s*(?=\n|$)                         # capture the rest of the line (answer text)
    """
)

def _looks_like_step_line(line: str) -> bool:
    l = line.strip().lower()
    if any(tok in l for tok in ["+", "carry", "=", "units", "tens", "hundreds", "thousands", "→", "->"]):
        return True
    if len(re.findall(r"-?\d+", l)) >= 2:
        return True
    return False

def _strip_answer_clauses_everywhere(s: str) -> str:
    if not s:
        return s
    text = s
    # Remove all explicit "answer/result is ..." lines
    text = ANSWER_CLAUSE_PATTERN.sub(lambda m: (m.group(1) or "") , text)
    return text.strip()

def _strip_standalone_answer_numbers(s: str, answer_int: Optional[int]) -> str:
    """
    Remove standalone trailing or isolated numbers equal to answer_int.
    We remove *all* occurrences of that number if it appears as a standalone token
    (not part of '1234,' etc.) and not in lines that look like step computations.
    """
    if not s:
        return s
    if answer_int is None:
        return s

    # token regex that matches the answer number as a whole token
    num_token = re.escape(f"{answer_int:,}")
    num_token_nocomma = re.escape(str(answer_int))
    token_pat = re.compile(rf"(?<!\d)({num_token}|{num_token_nocomma})(?!\d)")

    lines = s.splitlines()
    out_lines: List[str] = []
    for ln in lines:
        if _looks_like_step_line(ln):
            # keep step lines untouched
            out_lines.append(ln)
            continue
        # remove standalone tokens equal to answer
        ln2 = token_pat.sub("", ln)
        # also drop empty residues of lines that only had that token
        out_lines.append(ln2.strip())
    return "\n".join([l for l in out_lines if l.strip() != ""]).strip()

def extract_answer_substrings_multi(client: OpenAI, reasoning_trace: str) -> List[str]:
    """
    Ask the model to COPY ALL substrings that state the final answer (verbatim).
    Returns a list (possibly empty) of exact substrings to remove.
    """
    prompt = f"""
You will receive a reasoning trace (plain text). Your task is ONLY to COPY the exact substrings
that state the final answer (e.g., lines like "Answer: 1579", "So final result is 1272", or a lone "1272"
if it appears as the declared final answer). There may be multiple mentions; RETURN THEM ALL.

Do NOT paraphrase or rewrite. Return strict JSON:
{{ "answer_texts": ["<substring1>", "<substring2>", ...] }}

TRACE:
\"\"\"{reasoning_trace}\"\"\"
"""
    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Output strict JSON only. Copy verbatim substrings; do not paraphrase."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = json_from_text_maybe(raw)
    if not data:
        return []
    arr = data.get("answer_texts", [])
    if not isinstance(arr, list):
        return []
    out: List[str] = []
    for x in arr:
        if isinstance(x, str) and x.strip():
            out.append(x)
    return out

def strip_answers_model_based_multi(
    client: OpenAI,
    reasoning_trace: str,
    allow_fallback: bool = True,
    known_answer: Optional[int] = None,
) -> str:
    """
    Remove ALL explicit answer mentions from the reasoning:
      1) Ask model for ALL exact answer substrings and drop every occurrence.
      2) Fallback: remove explicit 'answer/result is ...' lines anywhere (not just last).
      3) If known_answer is provided, also remove standalone tokens equal to that number in non-step lines.
    """
    if not reasoning_trace:
        return reasoning_trace

    text = reasoning_trace
    # 1) model-based multi-capture removal
    try:
        targets = extract_answer_substrings_multi(client, reasoning_trace)
    except Exception:
        targets = []

    if targets:
        for t in targets:
            # remove ALL occurrences (global)
            text = text.replace(t, "")
        text = "\n".join([ln for ln in text.splitlines() if ln.strip() != ""]).strip()

    # 2) heuristic fallback: drop explicit clauses
    if allow_fallback:
        before = text
        text = _strip_answer_clauses_everywhere(text)
        # 3) remove standalone tokens equal to known answer
        text = _strip_standalone_answer_numbers(text, known_answer)
        if not text:
            text = before  # avoid returning empty if we overstripped

    return text.strip() if text.strip() else reasoning_trace

# ---------- "Answer mentions" alteration helpers ----------

def choose_altered_answer_nearby(original: int) -> int:
    """
    Choose a close but different integer (±1..±9), avoiding negatives.
    """
    delta = random.choice([d for d in range(-9, 10) if d != 0])
    altered = original + delta
    if altered < 0:
        altered = original + abs(delta)
    return altered

def _answer_token_patterns(ans_int: int) -> List[re.Pattern]:
    """
    Build patterns that match the answer as a whole number token, with or without commas.
    """
    tokens = [str(ans_int), f"{ans_int:,}"]
    return [re.compile(rf"(?<!\d){re.escape(tok)}(?!\d)") for tok in tokens]

def replace_all_answer_numbers(s: str, original_answer: int, altered_answer: int) -> str:
    """
    Replace ALL whole-token occurrences of original_answer with altered_answer,
    leaving all other parts of the reasoning untouched (no substep correction).
    """
    out = s
    pats = _answer_token_patterns(original_answer)
    for pat in pats:
        out = pat.sub(str(altered_answer), out)
    return out
