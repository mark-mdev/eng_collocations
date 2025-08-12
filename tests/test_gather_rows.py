import csv
from typing import Dict, List

from scrape_collocations import (
    gather_all_rows,
    write_all_cards_csv,
)


def test_gather_all_rows_two_word_content_and_prep():
    word = "decision"
    # Simulate outputs from the model with both types
    gpt_examples: Dict[str, List[str]] = {
        "make decision": [
            "We need to make the decision today.",
            "They made a difficult decision together.",
            "I will make a quick decision soon.",
            "Managers often make important decisions under pressure.",
            "She made several key decisions last year.",
            "The team must make a final decision by Friday.",
        ],
        "decision to": [
            "Her decision to move surprised everyone.",
            "The board's decision to merge was controversial.",
        ],
        "take a decision": [
            "They will take a decision after lunch.",
        ],
    }
    defs = {
        "make decision": "choose one option",
        "decision to": "choice leading to action",
        "take a decision": "formulate a choice",
    }

    rows = gather_all_rows(word, gpt_examples, defs)

    # For two-word content collocation, half should be lemma cloze and half partner cloze
    lemma_rows = [r for r in rows if "{{c1::decision}}" in r[0]]
    partner_rows = [r for r in rows if "{{c1::make}}" in r[0]]
    assert len(lemma_rows) >= 1
    assert len(partner_rows) >= 1

    # Preposition chunk should wrap the contiguous chunk once
    has_prep_chunk = any("{{c1::decision to}}" in r[0] for r in rows)
    assert has_prep_chunk is True



def test_write_all_cards_csv(tmp_path):
    word = "decision"
    gpt_examples = {"make decision": ["We made the decision yesterday."]}
    defs = {"make decision": "choose one option"}

    out_path = write_all_cards_csv(word, gpt_examples, defs, str(tmp_path))

    with open(out_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0][1].startswith("make Decision")  # label capitalization follows target-only rule 