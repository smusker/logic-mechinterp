# gpt_oss_experiment_utilities.py
"""
Utilities for parsing numbers, JSON, and robustly removing/altering answer mentions
from reasoning chains. Imported by gpt_oss_experiment.py.

Purging is MODEL-FIRST:
  1) Ask a model to list ALL answer substrings verbatim; remove every occurrence.
  2) Lightweight cleanup removes dangling artifacts (e.g., '= .', 'So answer is .').
  3) Minimal fallback (optional) to drop explicit answer-clauses and standalone answer tokens,
     only when the model misses obvious mentions.
"""

import re
import json
import random
from typing import Dict, Any, Optional, List
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

# ---------- MODEL-FIRST multi-mention answer stripping ----------

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

def _cleanup_dangling_artifacts(text: str) -> str:
    """
    Remove lines that become meaningless after answer removal, WITHOUT relying on
    model-dependent phrasing. Heuristics are structural and minimal:

      • Drop lines with no digits that contain a connector ('=' or ':') but have
        nothing substantive after (e.g., '= .', '=').
      • Drop lines that mention 'answer/result/sum/total/output/final' but contain no digits.
      • Collapse multiple blank lines.

    This keeps the chain readable while staying independent of any OSS model phrasing.
    """
    if not text:
        return text

    connector_pat = re.compile(r"[=:]")
    keyword_pat = re.compile(r"\b(answer|result|sum|total|output|final)\b", re.I)

    out_lines: List[str] = []
    for raw in text.splitlines():
        ln = raw.rstrip()

        has_digit = bool(re.search(r"\d", ln))
        has_connector = bool(connector_pat.search(ln))
        has_keyword = bool(keyword_pat.search(ln))

        # Normalize trivial punctuation remnants like '= .' or '= .' at end
        trimmed = ln.strip()
        # Drop lines that end with '=' or ':' or variants with only punctuation trailing
        if not has_digit:
            if has_keyword:
                # A keyword line with no digits is likely a dangling "So answer is ."
                continue
            if has_connector:
                # If there's a connector but no digits anywhere, it's likely '= .' or similar
                # Check if the part after the last connector has only punctuation/space
                last_pos = max(trimmed.rfind('='), trimmed.rfind(':'))
                after = trimmed[last_pos + 1:].strip() if last_pos != -1 else ""
                if after == "" or re.fullmatch(r"[^\w\d]+", after or ""):
                    continue
            # If no digits and the line is only punctuation/quotes/emphasis → drop
            if re.fullmatch(r"[^\w\d]+", trimmed or ""):
                continue

        out_lines.append(ln)

    # Collapse multiple blank lines
    compact: List[str] = []
    prev_blank = False
    for ln in out_lines:
        is_blank = (ln.strip() == "")
        if is_blank and prev_blank:
            continue
        compact.append(ln)
        prev_blank = is_blank

    return "\n".join(compact).strip()

# Minimal legacy helpers for fallback only
ANSWER_CLAUSE_PATTERN = re.compile(
    r"""(?isx)
    (^|\n)[^\S\r\n]*
    (?:so\s+)?(?:therefore[,:\s]+)?(?:final\s+)?(?:answer|result|sum|total|output|final)\s*
    (?:is|=|:)\s*
    (.+?)\s*(?=\n|$)
    """
)
def _strip_answer_clauses_everywhere(s: str) -> str:
    if not s:
        return s
    return ANSWER_CLAUSE_PATTERN.sub(lambda m: (m.group(1) or ""), s).strip()

def _looks_like_step_line(line: str) -> bool:
    l = line.strip().lower()
    if any(tok in l for tok in ["+", "carry", "=", "units", "tens", "hundreds", "thousands", "→", "->"]):
        return True
    if len(re.findall(r"-?\d+", l)) >= 2:
        return True
    return False

def _strip_standalone_answer_numbers(s: str, answer_int: Optional[int]) -> str:
    """
    Remove standalone tokens equal to answer_int in non-step lines.
    """
    if not s or answer_int is None:
        return s

    num_token = re.escape(f"{answer_int:,}")
    num_token_nocomma = re.escape(str(answer_int))
    token_pat = re.compile(rf"(?<!\d)({num_token}|{num_token_nocomma})(?!\d)")

    lines = s.splitlines()
    out_lines: List[str] = []
    for ln in lines:
        if _looks_like_step_line(ln):
            out_lines.append(ln)
            continue
        ln2 = token_pat.sub("", ln)
        if ln2.strip():
            out_lines.append(ln2.strip())
    return "\n".join(out_lines).strip()

def strip_answers_model_based_multi(
    client: OpenAI,
    reasoning_trace: str,
    allow_fallback: bool = True,
    known_answer: Optional[int] = None,
) -> str:
    """
    Remove ALL explicit answer mentions from the reasoning:

      MODEL-FIRST:
        1) Ask model for ALL exact answer substrings and drop every occurrence (global).
        2) Clean up dangling artifacts structurally (no model-phrase dependence).

      MINIMAL FALLBACK (optional):
        3) Remove explicit 'answer/result is ...' clauses anywhere.
        4) If known_answer is provided, remove standalone tokens equal to that number
           in non-step lines.

    Returns cleaned reasoning text; if we overstrip to empty, returns original.
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
            text = text.replace(t, "")
        # Remove any fully blank lines left by removal
        text = "\n".join([ln for ln in text.splitlines() if ln.strip() != ""])

    # 2) structural cleanup (handles '= .' & similar)
    text = _cleanup_dangling_artifacts(text)

    # 3-4) optional minimal fallback
    if allow_fallback:
        before = text
        text = _strip_answer_clauses_everywhere(text)
        text = _strip_standalone_answer_numbers(text, known_answer)
        text = _cleanup_dangling_artifacts(text)
        if not text:
            text = before  # avoid emptying the trace entirely

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
