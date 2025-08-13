import argparse
import os
import sys
import time
import re
import csv
from typing import List, Tuple, Optional, Dict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString
from dotenv import load_dotenv

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


BASE_URL = "https://www.just-the-word.com/"
MIN_FREQUENCY = 201  # strictly larger than 200
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-5"
COLLOCATIONS_LIMIT_PER_WORD = 8

# Debug logging toggle
DEBUG = os.getenv("DEBUG_COLL", "0").lower() in ("1", "true", "yes", "on")

def debug_log(message: str) -> None:
    if DEBUG:
        try:
            print(f"[DEBUG] {message}", file=sys.stderr)
        except Exception:
            pass


def fetch_collocations_page(word: str) -> str:
    """
    Fetches the Just-The-Word combinations page HTML for the given word.
    Uses a session to first hit the homepage to obtain a fresh cookie, then requests the combinations page.
    """
    base_url = urljoin(BASE_URL, "main.pl")
    params = {"word": word, "mode": "combinations"}

    session = requests.Session()

    # 1) Bootstrap: visit homepage to obtain cookie/session like a browser
    bootstrap_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:141.0) Gecko/20100101 Firefox/141.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        # Let requests negotiate encoding; don't force br/zstd which may require extra libs
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    last_exc: Optional[Exception] = None
    for attempt in range(5):
        try:
            # Bootstrap homepage
            session.get(BASE_URL, headers=bootstrap_headers, timeout=30)

            # 2) Now request the combinations page with Referer and same UA
            headers = {
                "User-Agent": bootstrap_headers["User-Agent"],
                "Accept": bootstrap_headers["Accept"],
                "Accept-Language": bootstrap_headers["Accept-Language"],
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": BASE_URL,
            }

            response = session.get(base_url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            html = response.text
            # Detect downtime banner
            if "the server is down temporarily" in html.lower() and "error code 304" in html.lower():
                raise RuntimeError("Just-The-Word is down temporarily (ERROR CODE 304)")
            return html
        except Exception as e:
            last_exc = e
            time.sleep(min(2 ** attempt + (attempt * 0.1), 5.0))
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown error fetching collocations page")


def parse_collocations(html: str) -> List[Tuple[str, int, Optional[str]]]:
    """
    Parses the HTML content to extract collocations, their frequencies, and the
    absolute URL of their examples page when available.

    Expects links under spans with class 'collocstring', with text like:
    'gain commitment (16)'. Returns list of tuples: ("gain commitment", 16, examples_url)
    where examples_url may be None if not found.
    """
    soup = BeautifulSoup(html, "lxml")

    collocations: List[Tuple[str, int, Optional[str]]] = []

    for span in soup.select("span.collocstring"):
        link = span.find("a")
        if not link or not link.text:
            continue
        text = link.get_text(strip=True)
        href = link.get("href")
        examples_url = urljoin(BASE_URL, href) if href else None
        # Expect pattern like: phrase (123)
        if text.endswith(")") and "(" in text:
            try:
                phrase_part, freq_part = text.rsplit("(", 1)
                phrase = phrase_part.strip()
                freq_str = freq_part.rstrip(")").strip()
                frequency = int(freq_str)
                if phrase:
                    collocations.append((phrase, frequency, examples_url))
            except (ValueError, IndexError):
                continue
    return collocations


def write_collocations_to_file(word: str, collocations: List[Tuple[str, int, Optional[str]]], output_dir: str) -> str:
    """
    Writes the collocations to a .txt file named '<word>_collocations.txt' in output_dir.
    Lines are in the format: 'phrase\tfrequency'. Sorted by descending frequency, then phrase.
    Returns the output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{word}_collocations.txt"
    output_path = os.path.join(output_dir, filename)

    collocations_sorted = sorted(collocations, key=lambda x: (-x[1], x[0].lower()))

    with open(output_path, "w", encoding="utf-8") as f:
        for phrase, frequency, _ in collocations_sorted:
            f.write(f"{phrase}\t{frequency}\n")

    return output_path


# spaCy model cache
_SPACY_NLP = None


def _get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except Exception as e:
            raise RuntimeError("spaCy model 'en_core_web_sm' is required but not installed. Install via: python -m spacy download en_core_web_sm") from e
    return _SPACY_NLP


def html_line_to_cloze_text(html_line: str, target_word: str) -> str:
    """
    Convert an example HTML line to text suitable for Anki cloze:
    - Remove <font> tags while keeping their inner text
    - Wrap one occurrence of target lemma or its inflected form with {{c1::...}}
    - If an article (a/an/the) directly precedes that token, include it inside the cloze
    - Return plain text (no other HTML)
    """
    soup = BeautifulSoup(html_line, "lxml")
    # Strip font tags, keep text
    for font_tag in soup.find_all("font"):
        font_tag.replace_with(NavigableString(font_tag.get_text()))

    text = soup.get_text()
    text = re.sub(r"\s+", " ", text).strip()

    nlp = _get_spacy()
    doc = nlp(text)

    target_lower = target_word.lower()

    # Find a token whose lemma matches the target (case-insensitive)
    idx = -1
    for i, tok in enumerate(doc):
        if tok.lemma_.lower() == target_lower or tok.text.lower() == target_lower:
            idx = i
            break

    # If not found, do not fallback; return unchanged
    if idx == -1:
        return text

    # Build the wrapped text using token offsets
    # Include preceding article if adjacent
    start = doc[idx].idx
    end = start + len(doc[idx].text)

    # Check for adjacent article token immediately before
    if idx - 1 >= 0:
        prev = doc[idx - 1]
        if prev.text.lower() in ("a", "an", "the") and prev.idx + len(prev.text) == start:
            start = prev.idx

    wrapped = text[:start] + "{{c1::" + text[start:end] + "}}" + text[end:]

    # If we didn't include an article in slicing (due to punctuation/spacing), run regex inclusion
    wrapped = re.sub(r"\b([Aa]n?|[Tt]he)\s+\{\{c1::([^}]+)\}\}", lambda m: f"{{{{c1::{m.group(1)} {m.group(2)}}}}}", wrapped)

    return wrapped


def collocation_label(phrase: str, target_word: str) -> str:
    """
    Format the collocation label like 'Commitment to': capitalize the target word only.
    """
    def repl(m: re.Match) -> str:
        w = m.group(0)
        return w[0].upper() + w[1:]

    return re.sub(rf"\b{re.escape(target_word)}\b", repl, phrase, flags=re.IGNORECASE)



def build_batch_filter_prompt(word: str, phrases: List[str]) -> List[Dict[str, str]]:
    system = "You are an expert ESL lexicographer. Return ONLY JSON as instructed."
    items = "\n".join(f"- {p}" for p in phrases)
    user = f"""
Task: For each collocation below (base lemma: "{word}"), decide two flags:
- british: whether the collocation is characteristically British usage.
- formal_only: whether the collocation is used only in formal or academic writing and is uncommon in everyday conversational English.

Input collocations (in order):
{items}

Output format (strict):
- Return ONLY a JSON array.
- The array MUST have the same length and order as the inputs.
- Each element MUST be an object with EXACTLY three keys: "phrase" (string, copied exactly as input), "british" (boolean), and "formal_only" (boolean).
- Do NOT include any other keys (no reasons, no variety) and no extra text before/after the JSON.

Meaning of british:
- british=true: the collocation is characteristically British (e.g., "take a decision").
- british=false: general/international usage (e.g., "make a decision", "reach a decision").

Meaning of formal_only:
- formal_only=true: sounds institutional, bureaucratic, legalistic, or academic and would rarely appear in everyday conversation. Often found in academic papers, official reports, legal documents, or formal memos.
- formal_only=false: acceptable in informal or general spoken English, even if also used in writing.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def filter_collocations_with_gpt(word: str, collocations: List[Tuple[str, int, Optional[str]]]) -> List[Tuple[str, int, Optional[str]]]:
    if OpenAI is None or not collocations or not OPENAI_API_KEY:
        debug_log(f"filter_collocations_with_gpt: skipping. OpenAI_present={OpenAI is not None}, collocations={len(collocations)}, has_api_key={bool(OPENAI_API_KEY)}")
        return collocations
    client = OpenAI(api_key=OPENAI_API_KEY)

    phrases = [phrase for phrase, _, _ in collocations]
    messages = build_batch_filter_prompt(word, phrases)
    debug_log(f"filter_collocations_with_gpt: model={OPENAI_MODEL}, num_phrases={len(phrases)}, sys_len={len(messages[0]['content']) if messages else 0}, user_len={len(messages[-1]['content']) if messages else 0}")

    def try_once() -> Optional[List[Dict[str, object]]]:
        text = attempt_responses(client, messages, reasoning_effort="medium")
        debug_log(f"filter_collocations_with_gpt: try_once text_len={len(text) if text else 0}")
        if not text:
            return None
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception as e:
            debug_log(f"filter_collocations_with_gpt: JSON parse error on first attempt: {e}")
            return None
        return None

    data = try_once()
    if data is None:
        strict = messages + [{"role": "user", "content": "Return ONLY a valid JSON array with objects: {phrase:string,british:boolean,formal_only:boolean}"}]
        text2 = attempt_responses(client, strict, reasoning_effort="medium")
        debug_log(f"filter_collocations_with_gpt: retry text_len={len(text2) if text2 else 0}")
        try:
            import json
            data = json.loads(text2) if text2 else None
        except Exception as e:
            debug_log(f"filter_collocations_with_gpt: JSON parse error on retry: {e}")
            data = None

    if not isinstance(data, list):
        debug_log("filter_collocations_with_gpt: no valid decisions, returning original collocations")
        return collocations

    decisions_brit: Dict[str, bool] = {}
    decisions_formal_only: Dict[str, bool] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        phr = str(item.get("phrase", ""))
        brit = bool(item.get("british", False))
        formal_only = bool(item.get("formal_only", False))
        key = phr.lower()
        decisions_brit[key] = brit
        decisions_formal_only[key] = formal_only

    # lightweight visibility of decisions
    print({
        k: {
            "british": decisions_brit.get(k, False),
            "formal_only": decisions_formal_only.get(k, False),
        }
        for k in set(decisions_brit) | set(decisions_formal_only)
    })

    filtered: List[Tuple[str, int, Optional[str]]] = []
    brit_true = 0
    formal_true = 0
    for phrase, freq, url in collocations:
        key = phrase.lower()
        brit = decisions_brit.get(key)
        formal_only = decisions_formal_only.get(key)
        if brit is True:
            brit_true += 1
            continue
        if formal_only is True:
            formal_true += 1
            continue
        filtered.append((phrase, freq, url))

    debug_log(f"filter_collocations_with_gpt: decisions={len(decisions_brit)}, brit_true={brit_true}, formal_only_true={formal_true}, kept={len(filtered)} of {len(collocations)}")
    return filtered


def _collocation_in_order(sentence: str, phrase: str, target_word: str) -> bool:
    """Return True if sentence contains all collocation words, unchanged, in order (not necessarily contiguous).
    Allows 'a' token to match 'a' or 'an'. Also enforces the target LEMMA appears exactly once overall.
    Uses spaCy lemmatization so that base-form tokens in the phrase (e.g., 'be') can match inflected forms in the sentence (e.g., 'is', 'being').
    """
    # Normalize text
    text = BeautifulSoup(sentence, "lxml").get_text()
    text = re.sub(r"\s+", " ", text).strip()

    nlp = _get_spacy()
    doc = nlp(text)

    # Only consider alphabetic tokens for matching sequence
    sent_tokens = [tok for tok in doc if tok.is_alpha]
    sent_lemmas = [tok.lemma_.lower() for tok in sent_tokens]
    sent_texts = [tok.text.lower() for tok in sent_tokens]

    tgt = target_word.lower()
    # Target lemma occurrence must be exactly once
    if sent_lemmas.count(tgt) != 1:
        return False

    # Build phrase lemmas using spaCy as well
    phrase_doc = nlp(phrase.lower())
    phrase_tokens = [tok for tok in phrase_doc if tok.is_alpha]
    phrase_lemmas = [tok.lemma_.lower() for tok in phrase_tokens]
    phrase_texts = [tok.text.lower() for tok in phrase_tokens]

    i = 0  # index in sentence tokens
    for p_text, p_lemma in zip(phrase_texts, phrase_lemmas):
        found = False
        if p_text in ("a", "an"):
            # Accept either 'a' or 'an' textually
            while i < len(sent_tokens):
                if sent_texts[i] in ("a", "an"):
                    found = True
                    i += 1
                    break
                i += 1
        else:
            # Lemma-based match for inflected forms
            while i < len(sent_tokens):
                if sent_lemmas[i] == p_lemma:
                    found = True
                    i += 1
                    break
                i += 1
        if not found:
            return False
    return True


# Helper to call the Responses API consistently
def attempt_responses(client: OpenAI, messages: List[Dict[str, str]], reasoning_effort: str = "minimal", gpt_model: str = OPENAI_MODEL) -> str:
    try:
        sys_prompt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        user_prompt = messages[-1]["content"] if messages else ""
        composed = (sys_prompt + "\n\n" + user_prompt).strip()
        debug_log(f"attempt_responses: model={gpt_model}, effort={reasoning_effort}")
        debug_log(f"attempt_responses: sys_len={len(sys_prompt)}, user_len={len(user_prompt)}, composed_len={len(composed)}")
        resp = client.responses.create(model=gpt_model, input=composed, reasoning={"effort": reasoning_effort})
        debug_log(f"attempt_responses: raw_response={resp}")
        text = getattr(resp, "output_text", None) or ""
        if not text and hasattr(resp, "output"):
            try:
                parts = []
                for item in resp.output:  # type: ignore[attr-defined]
                    if hasattr(item, "content"):
                        for c in item.content:  # type: ignore[attr-defined]
                            if hasattr(c, "text") and hasattr(c.text, "value"):
                                parts.append(c.text.value)
                text = "".join(parts)
            except Exception:
                text = ""
        debug_log(f"attempt_responses: text_len={len(text) if text else 0}")
        return text.strip()
    except Exception as e:
        debug_log(f"attempt_responses: error={e}")
        return ""



def compute_desired_counts(word: str, phrases: List[str]) -> Dict[str, int]:
    n = len(phrases)
    if n <= 2:
        base = 5
    elif n <= 5:
        base = 3
    else:
        base = 2
    counts: Dict[str, int] = {}
    for p in phrases:
        desired = base
        # For two-word content collocations we need dual sets (lemma+partner), so double
        if is_two_word_content_collocation(p):
            desired = base * 2
        # If this collocation is exactly the target lemma, double and cap at 5
        if p.lower() == word.lower():
            desired = min(5, base * 2)
        counts[p] = desired
    return counts

def gather_all_rows(word: str, gpt_examples: Dict[str, List[str]], defs: Dict[str, str]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for phrase, examples in gpt_examples.items():
        label = collocation_label(phrase, word)
        definition = defs.get(phrase, "")
        back = f"{label} — {definition}".strip().rstrip(" —")

        # Preposition collocations: full chunk (contiguous) else lemma fallback
        if is_prep_collocation(phrase):
            for html_line in examples:
                sent = plain_text_from_html_line(html_line)
                wrapped = wrap_full_chunk_prep(sent, phrase, word)
                if not wrapped:
                    wrapped = html_line_to_cloze_text(html_line, word)
                rows.append((wrapped, back))
            continue

        # Two-word content collocations: split 6 examples → 3 lemma, 3 partner
        if is_two_word_content_collocation(phrase):
            # Lemma: first half, Partner: second half (even split)
            n = len(examples)
            if n == 0:
                continue
            half = n // 2
            lemma_examples = examples[:half]
            partner_examples = examples[half:]

            for html_line in lemma_examples:
                rows.append((html_line_to_cloze_text(html_line, word), back))

            tokens = tokenize_phrase(phrase)
            t1, t2 = tokens
            partner = t1 if t2 == word.lower() else (t2 if t1 == word.lower() else None)
            if partner:
                for html_line in partner_examples:
                    if not html_line:
                        continue
                    sent = plain_text_from_html_line(html_line)
                    row = re.sub(rf"\b{re.escape(partner)}\b", lambda m: f"{{{{c1::{m.group(0)}}}}}", sent, flags=re.IGNORECASE, count=1)
                    rows.append((row, back))
            continue

        # Other types: lemma-only for all provided examples
        for html_line in examples:
            rows.append((html_line_to_cloze_text(html_line, word), back))

    return rows


def write_all_cards_csv(word: str, gpt_examples: Dict[str, List[str]], defs: Dict[str, str], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"anki_cloze_{word}.csv")
    rows = gather_all_rows(word, gpt_examples, defs)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for front, back in rows:
            writer.writerow([front, back])
    return out_path


# Collocation utility sets and helpers
PREPOSITIONS = {
    "to","for","with","on","in","at","of","from","about","over","under","into","onto",
    "through","against","between","among","by","without","within","upon","off","out","up",
    "down","across","after","before","around","beyond","during","like","near","past","since",
    "toward","towards","via"
}
STOPWORDS_FUNCTION = {
    "a","an","the","my","your","his","her","its","our","their","this","that","these",
    "those","any","some","no","and","or","but","not","as","than","so","very","too",
    "more","most","much","many","few","several","do","does","did","be","am","is","are",
    "was","were","been","being","have","has","had","will","would","can","could","should",
    "may","might","must","with","to","of","in","on","at","for","by","from","about","into",
    "over","under","after","before","between","among","without","within","upon","off","out","up",
    "down","across","around","beyond","during","since","like","near","past","through","against",
    "toward","towards","via","also"
}

def tokenize_phrase(phrase: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", phrase.lower())


def is_prep_collocation(phrase: str) -> bool:
    tokens = tokenize_phrase(phrase)
    return any(tok in PREPOSITIONS for tok in tokens) and len(tokens) >= 2


def is_two_word_content_collocation(phrase: str) -> bool:
    tokens = tokenize_phrase(phrase)
    if len(tokens) != 2:
        return False
    return all(tok not in STOPWORDS_FUNCTION and tok not in PREPOSITIONS for tok in tokens)


def plain_text_from_html_line(html_line: str) -> str:
    soup = BeautifulSoup(html_line, "lxml")
    text = soup.get_text()
    return re.sub(r"\s+", " ", text).strip()


def wrap_full_chunk_prep(sentence: str, phrase: str, target_word: str) -> Optional[str]:
    tokens = tokenize_phrase(phrase)
    prep_idxs = [i for i, t in enumerate(tokens) if t in PREPOSITIONS]
    if not prep_idxs:
        return None
    prep = tokens[prep_idxs[0]]
    target = None
    tw = target_word.lower()
    for t in tokens:
        if t == tw:
            target = t
            break
    if target is None:
        for t in tokens:
            if t not in PREPOSITIONS and t not in STOPWORDS_FUNCTION:
                target = t
                break
    if target is None:
        return None

    pat1 = re.compile(rf"\b{re.escape(prep)}\s+{re.escape(target)}\b", re.IGNORECASE)
    pat2 = re.compile(rf"\b{re.escape(target)}\s+{re.escape(prep)}\b", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        return f"{{{{c1::{m.group(0)}}}}}"

    out, n = pat1.subn(repl, sentence, count=1)
    if n == 0:
        out, n = pat2.subn(repl, sentence, count=1)
    return out if n > 0 else None


def normalize_collocation_phrase(word: str, phrase: str) -> Tuple[str, bool]:
    """
    Normalize JTW marker prefixes like '.word' (singular-only) by removing the marker
    Applies only when the marker is attached to the target word token.
    """
    w = re.escape(word)
    cleaned = re.sub(rf"(?<!\w)\.{w}(?!\w)", lambda m: m.group(0)[1:], phrase, flags=re.IGNORECASE)
    return cleaned


def build_examples_defs_prompt(word: str, phrase_to_count: Dict[str, int], extra_requirements: str = "") -> List[Dict[str, str]]:
    system = "You are an expert ESL materials writer creating natural, high-frequency example sentences for English collocations. Return ONLY valid JSON."
    extra = ("\n" + extra_requirements.strip()) if extra_requirements.strip() else ""
    phrases = list(phrase_to_count.keys())
    counts_map_str = str(phrase_to_count)
    user = f"""
Collocations: {str(phrases)}
Base lemma (target word): "{word}"

For each collocation, first classify whether it should be denied for use:
- denied=true when the collocation is characteristically British usage (see below) OR is used only in formal/academic/bureaucratic writing and is uncommon in everyday conversational English.
- reason must be one of: "british", "formal" when denied=true; otherwise "" (empty string).

Then, if denied=false, generate EXACTLY the number of sentences specified in this mapping:
{counts_map_str}

Return ONLY a JSON object where each key is a collocation string from the input list, and its value is an object with EXACTLY these keys:
- denied: boolean
- reason: string ("british" | "formal" | "")
- definition: string (5–15 words, plain-English core meaning, no examples or extra context)
- sentences: array of strings (exactly the number requested for that collocation when denied=false; [] when denied=true)

Meanings/guidance for denial:
- British (reason="british"): characteristically British (e.g., "take a decision").
- Formal-only (reason="formal"): sounds institutional, bureaucratic, legalistic, or academic and would rarely appear in everyday conversation. Often in academic papers, official reports, legal documents, or formal memos.

Core rules:
1. The collocation must appear exactly as given, in natural grammatical form (may inflect for tense/person/number, and add minimal function words like articles, prepositions, auxiliaries).
2. Never replace the collocation with synonyms or rephrases (e.g., do not replace “make a decision” with “decide”).
3. Grammar > naturalness > variety > exact wording.

Sentence requirements (apply only when denied=false):
- At least 1 question, 1 statement, and 1 imperative (when counts allow).
- At least 1 first-person and 1 third-person subject (when counts allow).
- Use at least 2 different real-life domains (e.g., work, home, travel, shopping, relationships, hobbies, study).
- No more than one sentence starting with the same first 3 words.
- Include a mix of tenses (present, past, future where possible).
- Avoid overly abstract or generic contexts; prefer specific, concrete everyday scenarios.
- Sentence length: mostly 8–15 words (slightly shorter/longer if needed for naturalness).
- Mix simple, compound, and complex sentence structures.
- Vary auxiliary verbs and determiners to avoid repetition.

Quality control before returning:
- Silently fix article/determiner use, subject–verb agreement, prepositions, and countable noun rules.
- If the collocation’s raw form is unidiomatic, minimally adjust to make it natural (e.g., add “a” in “make decision”).
- Check that no more than 30% of sentences share the same subject or domain.{extra}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_examples_and_definitions(word: str, collocations: List[Tuple[str, int, Optional[str]]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if OpenAI is None or not collocations or not OPENAI_API_KEY:
        debug_log(f"generate_examples_and_definitions: skipping. OpenAI_present={OpenAI is not None}, collocations={len(collocations)}, has_api_key={bool(OPENAI_API_KEY)}")
        return {}, {}
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare mappings and desired counts
    orig_phrases: List[str] = [p for p, _, _ in collocations]
    desired_counts = compute_desired_counts(word, orig_phrases)
    debug_log(f"generate_examples_and_definitions: num_phrases={len(orig_phrases)}, desired_total={sum(desired_counts.values())}")

    # Normalize phrases (e.g., strip '.lemma' markers)
    orig_to_norm: Dict[str, str] = {p: normalize_collocation_phrase(word, p) for p in orig_phrases}
    # Reverse map (normalized -> original); if collisions, prefer first
    norm_to_orig: Dict[str, str] = {}
    for p in orig_phrases:
        np = orig_to_norm[p]
        if np not in norm_to_orig:
            norm_to_orig[np] = p

    # Build a per-collocation desired count map on normalized phrases
    norm_to_count: Dict[str, int] = {orig_to_norm[p]: desired_counts.get(p, 3) for p in orig_phrases}

    # Single batched call 
    messages = build_examples_defs_prompt(word, norm_to_count)
    debug_log(f"generate_examples_and_definitions: model={OPENAI_MODEL}, sys_len={len(messages[0]['content']) if messages else 0}, user_len={len(messages[-1]['content']) if messages else 0}")
    text = attempt_responses(client, messages, reasoning_effort="minimal")
    debug_log(f"generate_examples_and_definitions: response text_len={len(text) if text else 0}")

    examples: Dict[str, List[str]] = {}
    defs: Dict[str, str] = {}

    if not text:
        debug_log("generate_examples_and_definitions: empty text, returning no examples/defs")
        return examples, defs

    try:
        import json
        data = json.loads(text)
        if not isinstance(data, dict):
            debug_log("generate_examples_and_definitions: JSON was not an object")
            return examples, defs
        produced_before = 0
        produced_after = 0
        ok_collocs = 0
        denied_total = 0
        denied_british = 0
        denied_formal = 0
        for norm_phrase, payload in data.items():
            if not isinstance(payload, dict):
                continue
            sents = payload.get("sentences")
            definition = payload.get("definition")
            denied_flag = bool(payload.get("denied", False))
            reason_val = str(payload.get("reason", "")).strip().lower()
            if not isinstance(norm_phrase, str):
                continue
            orig_phrase = norm_to_orig.get(norm_phrase, None)
            if not orig_phrase:
                # Try exact match fallback (case-insensitive)
                for k, v in norm_to_orig.items():
                    if k.lower() == norm_phrase.lower():
                        orig_phrase = v
                        break
            if not orig_phrase:
                continue

            # If denied by the model, skip producing outputs
            if denied_flag or reason_val in ("british", "formal"):
                denied_total += 1
                if reason_val == "british":
                    denied_british += 1
                elif reason_val == "formal":
                    denied_formal += 1
                # Do not include examples/defs for denied collocations
                continue

            desired = desired_counts.get(orig_phrase, 3)
            normalized_phrase = norm_phrase
            out_sents = [s for s in sents if isinstance(s, str) and s.strip()] if isinstance(sents, list) else []
            produced_before += len(out_sents)
            # Validate order and lemma once
            out_sents = [s for s in out_sents if _collocation_in_order(s, normalized_phrase, word)]
            produced_after += len(out_sents)
            if out_sents:
                ok_collocs += 1
                examples[orig_phrase] = out_sents[:desired]
            if isinstance(definition, str) and definition.strip():
                defs[orig_phrase] = definition.strip()
        debug_log(
            f"generate_examples_and_definitions: collocs_with_examples={ok_collocs}, sents_before_validation={produced_before}, sents_after_validation={produced_after}, defs={len(defs)}, denied_total={denied_total}, denied_british={denied_british}, denied_formal={denied_formal}"
        )
    except Exception as e:
        debug_log(f"generate_examples_and_definitions: JSON parse/processing error: {e}")
        return examples, defs

    return examples, defs


def strip_dot_markers_from_collocations(word: str, collocations: List[Tuple[str, int, Optional[str]]]) -> List[Tuple[str, int, Optional[str]]]:
    """
    Remove JTW dot-marker attached to the target word token (e.g., ".word")
    from collocation phrases, and de-duplicate by normalized phrase while
    keeping the highest frequency (and preserving an examples URL if present).
    """
    normalized_to_best: Dict[str, Tuple[int, Optional[str]]] = {}
    for phrase, freq, url in collocations:
        normalized_phrase = normalize_collocation_phrase(word, phrase)
        best = normalized_to_best.get(normalized_phrase)
        if best is None or freq > best[0]:
            normalized_to_best[normalized_phrase] = (freq, url)

    # Convert back to list of tuples
    out: List[Tuple[str, int, Optional[str]]] = []
    for phrase, (freq, url) in normalized_to_best.items():
        out.append((phrase, freq, url))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape collocations from Just-The-Word and examples.")
    parser.add_argument("word", nargs="?", help="The English word to look up.")
    parser.add_argument("--word-file", dest="word_file", help="Path to a text file with one word per line.")
    parser.add_argument(
        "--max-workers",
        dest="max_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Maximum parallel processes when using --word-file (default: half of CPU cores).",
    )

    args = parser.parse_args()
    debug_log(f"main: DEBUG_COLL={DEBUG}, OPENAI_KEY_SET={bool(OPENAI_API_KEY)}, MODEL={OPENAI_MODEL}, args={{'word': args.word, 'word_file': args.word_file, 'max_workers': args.max_workers}}")

    # Batch mode takes precedence if provided
    if args.word_file:
        words = _read_words_file(args.word_file)
        if not words:
            print(f"No words found in file: {args.word_file}", file=sys.stderr)
            sys.exit(1)
        _process_words_in_parallel(words, args.max_workers, args.word_file)
        return

    # Single word mode (original behavior)
    word = (args.word or "").strip()
    if not word:
        print("Error: provide a WORD or --word-file.", file=sys.stderr)
        sys.exit(1)

    exit_code = 0
    try:
        # Run and then write per-word artifacts
        collocations, rows = run_pipeline_collect(word)
        colloc_out_dir = "./outputs"
        out_txt = write_collocations_to_file(word, collocations, colloc_out_dir)
        print(f"[{word}] Wrote {len(collocations)} collocations (freq>200) to: {out_txt}")
        out_csv = write_rows_csv(word, rows, colloc_out_dir)
        print(f"[{word}] Wrote All Anki Cloze Cards CSV to: {out_csv}")
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        exit_code = 2
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        exit_code = 3
    sys.exit(exit_code)


# --------------------
# Batch helpers
# --------------------

def _read_words_file(path: str) -> List[str]:
    """Read a text file of words, one per line, ignoring blanks and comments (#...)."""
    words: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # strip inline comments and whitespace
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                words.append(line)
    except FileNotFoundError:
        print(f"Word file not found: {path}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to read word file {path}: {e}", file=sys.stderr)
    # Preserve order but drop duplicates
    seen = set()
    deduped: List[str] = []
    for w in words:
        wl = w.lower()
        if wl in seen:
            continue
        seen.add(wl)
        deduped.append(w)
    return deduped


def _derive_batch_paths(word_file_path: str) -> Tuple[str, str]:
    base = os.path.splitext(os.path.basename(word_file_path))[0] or "batch"
    out_dir = "./outputs"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"anki_cloze_{base}.csv")
    txt_path = os.path.join(out_dir, f"{base}_collocations.txt")
    return csv_path, txt_path


def _runner_collect(w: str) -> Tuple[str, bool, str, List[Tuple[str, int, Optional[str]]], List[Tuple[str, str]]]:
    try:
        collocations, rows = run_pipeline_collect(w)
        return (w, True, "ok", collocations, rows)
    except Exception as e:  # pragma: no cover (child process)
        return (w, False, str(e), [], [])


def _process_words_in_parallel(words: List[str], max_workers: int, word_file_path: str) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"Processing {len(words)} words with up to {max_workers} parallel processes...")
    failures: List[Tuple[str, str]] = []
    per_word_results: Dict[str, Tuple[List[Tuple[str, int, Optional[str]]], List[Tuple[str, str]]]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_word = {executor.submit(_runner_collect, w): w for w in words}
        for future in as_completed(future_to_word):
            w = future_to_word[future]
            try:
                _w, ok, msg, collocs, rows = future.result()
                if not ok:
                    failures.append((w, msg))
                    print(f"[FAIL] {w}: {msg}", file=sys.stderr)
                else:
                    per_word_results[w] = (collocs, rows)
                    print(f"[DONE] {w}")
            except Exception as e:  # pragma: no cover
                failures.append((w, str(e)))
                print(f"[FAIL] {w}: {e}", file=sys.stderr)

    # Write aggregated outputs
    csv_path, txt_path = _derive_batch_paths(word_file_path)
    _write_batch_csv(words, per_word_results, csv_path)
    _write_batch_collocations_txt(words, per_word_results, txt_path)

    print(f"Wrote batch CSV: {csv_path}")
    print(f"Wrote batch collocations TXT: {txt_path}")

    if failures:
        print(f"Completed with {len(failures)} failures:")
        for w, msg in failures:
            print(f" - {w}: {msg}")
    else:
        print("Completed successfully for all words.")


def _write_batch_csv(words: List[str], results: Dict[str, Tuple[List[Tuple[str, int, Optional[str]]], List[Tuple[str, str]]]], out_path: str) -> None:
    rows_flat: List[Tuple[str, str]] = []
    for w in words:
        pair = results.get(w)
        if not pair:
            continue
        _, rows = pair
        rows_flat.extend(rows)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for front, back in rows_flat:
            writer.writerow([front, back])


def _write_batch_collocations_txt(words: List[str], results: Dict[str, Tuple[List[Tuple[str, int, Optional[str]]], List[Tuple[str, str]]]], out_path: str) -> None:
    def sort_key(item: Tuple[str, int, Optional[str]]):
        return (-item[1], item[0].lower())

    with open(out_path, "w", encoding="utf-8") as f:
        first = True
        for w in words:
            pair = results.get(w)
            if not pair:
                continue
            collocs, _ = pair
            collocs_sorted = sorted(collocs, key=sort_key)
            if not first:
                f.write("\n")  # blank line between sections
            first = False
            f.write(f"{w}\n")  # header line
            for phrase, frequency, _ in collocs_sorted:
                f.write(f"{phrase}\t{frequency}\n")


# --------------------
# Single-word processing (refactored and shared)
# --------------------

def run_pipeline_collect(word: str) -> Tuple[List[Tuple[str, int, Optional[str]]], List[Tuple[str, str]]]:
    """Run the full pipeline for a single word and return collocations and CSV rows."""
    debug_log(f"run_pipeline_collect: word={word}")
    html = fetch_collocations_page(word)
    collocations = parse_collocations(html)
    debug_log(f"run_pipeline_collect: parsed_collocations={len(collocations)}")

    # Strip dot markers like ".word" and de-duplicate by normalized phrase
    collocations = strip_dot_markers_from_collocations(word, collocations)

    # Filter collocations strictly greater than 200 frequency, then keep top N by frequency
    collocations_all = collocations[:]
    collocations = [c for c in collocations_all if c[1] >= MIN_FREQUENCY]
    debug_log(f"run_pipeline_collect: above_threshold={len(collocations)} of total={len(collocations_all)}")

    # If we got <= 1 above threshold, take top 2 overall and add target lemma
    if len(collocations) <= 1:
        collocations = sorted(collocations_all, key=lambda x: -x[1])[:2]

    # Now cap to top N by frequency
    collocations = sorted(collocations, key=lambda x: -x[1])[:COLLOCATIONS_LIMIT_PER_WORD]
    debug_log(f"run_pipeline_collect: capped_count={len(collocations)} (limit={COLLOCATIONS_LIMIT_PER_WORD})")

    # Ensure target lemma present after capping: if missing, append (allow 9th)
    phrases_lower = {p.lower() for p, _, _ in collocations}
    article_forms = {f"a {word.lower()}", f"an {word.lower()}", f"the {word.lower()}"}
    has_article_lemma = any(form in phrases_lower for form in article_forms)
    had_lemma = word.lower() in phrases_lower or has_article_lemma
    if word.lower() not in phrases_lower and not has_article_lemma:
        collocations.append((word, 999, None))
    debug_log(f"run_pipeline_collect: lemma_present_before={had_lemma}, after={True}")

    if not collocations:
        print("No collocations with frequency > 200 found.")

    # Filter collocations and generate content
    # filtered_collocations = filter_collocations_with_gpt(word, collocations)
    # debug_log(f"run_pipeline_collect: filtered_collocations={len(filtered_collocations)}")
    gpt_examples, gpt_definitions = generate_examples_and_definitions(word, collocations)
    debug_log(f"run_pipeline_collect: examples_collocs={len(gpt_examples)}, definitions={len(gpt_definitions)}")

    # Build rows
    rows = gather_all_rows(word, gpt_examples, gpt_definitions)
    debug_log(f"run_pipeline_collect: total_rows={len(rows)}")
    return collocations, rows


def write_rows_csv(word: str, rows: List[Tuple[str, str]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"anki_cloze_{word}.csv")
    debug_log(f"write_rows_csv: writing rows={len(rows)} to path={out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for front, back in rows:
            writer.writerow([front, back])
    return out_path


# Backwards-compat shim kept for imports/tests elsewhere
# (process_single_word is now implemented via run_pipeline_collect and writers)

def process_single_word(word: str) -> None:
    collocations, rows = run_pipeline_collect(word)
    colloc_out_dir = "./outputs"
    output_path = write_collocations_to_file(word, collocations, colloc_out_dir)
    print(f"[{word}] Wrote {len(collocations)} collocations (freq>200) to: {output_path}")
    all_cards_csv_path = write_rows_csv(word, rows, colloc_out_dir)
    print(f"[{word}] Wrote All Anki Cloze Cards CSV to: {all_cards_csv_path}")


if __name__ == "__main__":
    main() 