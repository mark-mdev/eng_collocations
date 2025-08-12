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
COLLOCATIONS_LIMIT_PER_WORD = 5


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
Task: For each collocation below (base lemma: "{word}"), decide if it is characteristically British usage.
Input collocations (in order):
{items}

Output format (strict):
- Return ONLY a JSON array.
- The array MUST have the same length and order as the inputs.
- Each element MUST be an object with EXACTLY two keys: "phrase" (string, copied exactly as input) and "british" (boolean).
- Do NOT include any other keys (no reasons, no variety) and no extra text before/after the JSON.

Meaning of british:
- british=true: the collocation is characteristically British (e.g., "take a decision").
- british=false: general/international usage (e.g., "make a decision", "reach a decision").
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def filter_collocations_with_gpt(word: str, collocations: List[Tuple[str, int, Optional[str]]]) -> List[Tuple[str, int, Optional[str]]]:
    if OpenAI is None or not collocations or not OPENAI_API_KEY:
        return collocations
    client = OpenAI(api_key=OPENAI_API_KEY)

    phrases = [phrase for phrase, _, _ in collocations]
    messages = build_batch_filter_prompt(word, phrases)

    def try_once() -> Optional[List[Dict[str, object]]]:
        text = attempt_responses(client, messages, reasoning_effort="medium")
        if not text:
            return None
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except Exception:
            return None
        return None

    data = try_once()
    if data is None:
        strict = messages + [{"role": "user", "content": "Return ONLY a valid JSON array with objects: {phrase:string,british:boolean}"}]
        text2 = attempt_responses(client, strict, reasoning_effort="medium")
        try:
            import json
            data = json.loads(text2) if text2 else None
        except Exception:
            data = None

    if not isinstance(data, list):
        return collocations

    decisions: Dict[str, bool] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        phr = str(item.get("phrase", ""))
        brit = bool(item.get("british", False))
        decisions[phr.lower()] = brit

    print({k: {"british": v} for k, v in decisions.items()})

    filtered: List[Tuple[str, int, Optional[str]]] = []
    for phrase, freq, url in collocations:
        brit = decisions.get(phrase.lower())
        if brit is True:
            continue
        filtered.append((phrase, freq, url))

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
        resp = client.responses.create(model=gpt_model, input=composed, reasoning={"effort": reasoning_effort})
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
        return text.strip()
    except Exception:
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
    "toward","towards","via"
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

For each collocation, return EXACTLY the number of sentences specified in this mapping:
{counts_map_str}

Return ONLY a JSON object where each key is a collocation string from the input list,
and its value is an object with keys:
- definition: string (5-8 words, simple, no comma, plain-English core meaning)
- sentences: array of strings (exactly the number requested for that collocation)

Requirements for definition:
- Must express only the essential meaning
- No commas or semicolons
- Prefer the most common meaning for general audiences
- Do not include examples, explanations, or context clues

Requirements for sentences:
- Start at the beginning of a sentence; no quotes, bullets, or numbering.
- Natural, high-frequency, everyday contexts; CEFR B1–B2 difficulty.
- 8-15 words each. Avoid jargon, proper names, brand names, and niche topics.
- Use ALL collocation words exactly as written and in the same order.
- You MAY insert neutral words between them (e.g., adjectives, adverbs, determiners).
- You MAY inflect the words' forms or tenses
- Do not reorder or replace collocation words.
- Use the target word "{word}" exactly once per sentence.
- Vary vocabulary and structure across sentences; avoid repetition.
- Prefer concrete, practical, real-life situations over abstract or vague statements.
- Do not add any extra HTML{extra}
- Output ONLY valid JSON with the specified keys. No extra text.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_examples_and_definitions(word: str, collocations: List[Tuple[str, int, Optional[str]]]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    if OpenAI is None or not collocations or not OPENAI_API_KEY:
        return {}, {}
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare mappings and desired counts
    orig_phrases: List[str] = [p for p, _, _ in collocations]
    desired_counts = compute_desired_counts(word, orig_phrases)

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
    text = attempt_responses(client, messages, reasoning_effort="low")

    examples: Dict[str, List[str]] = {}
    defs: Dict[str, str] = {}

    if not text:
        return examples, defs

    try:
        import json
        data = json.loads(text)
        if not isinstance(data, dict):
            return examples, defs
        for norm_phrase, payload in data.items():
            if not isinstance(payload, dict):
                continue
            sents = payload.get("sentences")
            definition = payload.get("definition")
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
            desired = desired_counts.get(orig_phrase, 3)
            normalized_phrase = norm_phrase
            out_sents = [s for s in sents if isinstance(s, str) and s.strip()] if isinstance(sents, list) else []
            # Validate order and lemma once
            out_sents = [s for s in out_sents if _collocation_in_order(s, normalized_phrase, word)]
            if out_sents:
                examples[orig_phrase] = out_sents[:desired]
            if isinstance(definition, str) and definition.strip():
                defs[orig_phrase] = definition.strip()
    except Exception:
        return examples, defs

    return examples, defs


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape collocations from Just-The-Word and examples.")
    parser.add_argument("word", help="The English word to look up.")

    args = parser.parse_args()
    word = args.word.strip()
    if not word:
        print("Error: word must be non-empty", file=sys.stderr)
        sys.exit(1)

    try:
        html = fetch_collocations_page(word)
        collocations = parse_collocations(html)

        # Filter collocations strictly greater than 200 frequency, then keep top 8 by frequency
        collocations_all = collocations[:]
        collocations = [c for c in collocations_all if c[1] >= MIN_FREQUENCY]

        # If we got <= 1 above threshold, take top 2 overall and add target lemma
        if len(collocations) <= 1:
            collocations = sorted(collocations_all, key=lambda x: -x[1])[:2]

        # Now cap to top 8 by frequency
        collocations = sorted(collocations, key=lambda x: -x[1])[:COLLOCATIONS_LIMIT_PER_WORD]

        # Ensure target lemma present after capping: if missing, append (allow 9th)
        phrases_lower = {p.lower() for p, _, _ in collocations}
        article_forms = {f"a {word.lower()}", f"an {word.lower()}", f"the {word.lower()}"}
        has_article_lemma = any(form in phrases_lower for form in article_forms)
        if word.lower() not in phrases_lower and not has_article_lemma:
            collocations.append((word, 999, None))

        if not collocations:
            print("No collocations with frequency > 200 found.")

        colloc_out_dir = "./outputs"
        # Keep the plain text list of collocations for reference
        output_path = write_collocations_to_file(word, collocations, colloc_out_dir)
        print(f"Wrote {len(collocations)} collocations (freq>200) to: {output_path}")


        # Filter collocations with GPT first
        filtered_collocations = filter_collocations_with_gpt(word, collocations)
        print(f"Filtered {len(collocations)} collocations to {len(filtered_collocations)} using GPT.")

        # Generate examples and definitions in one call per collocation
        gpt_examples, gpt_definitions = generate_examples_and_definitions(word, filtered_collocations)

        # Write GPT-only CSVs in two-column (Front, Back) format
        all_cards_csv_path = write_all_cards_csv(word, gpt_examples, gpt_definitions, colloc_out_dir)
        print(f"Wrote All Anki Cloze Cards CSV to: {all_cards_csv_path}")

    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        sys.exit(2)
    except requests.RequestException as e:
        print(f"Network error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main() 