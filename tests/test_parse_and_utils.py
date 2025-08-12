import os
from urllib.parse import urljoin

import pytest

from scrape_collocations import (
    BASE_URL,
    parse_collocations,
    write_collocations_to_file,
    plain_text_from_html_line,
    tokenize_phrase,
    is_two_word_content_collocation,
    normalize_collocation_phrase,
    _collocation_in_order,
)


def test_parse_collocations_simple():
    html = (
        '<span class="collocstring"><a href="/path1">gain commitment (16)</a></span>'
        '<span class="collocstring"><a href="/path2">reach a decision (205)</a></span>'
        '<span class="collocstring"><a href="/bad">malformed</a></span>'
    )
    out = parse_collocations(html)
    assert ("gain commitment", 16, urljoin(BASE_URL, "/path1")) in out
    assert ("reach a decision", 205, urljoin(BASE_URL, "/path2")) in out
    # malformed entry without "(n)" is ignored
    assert all("malformed" not in p for p, _, _ in out)


def test_write_collocations_to_file_sorted(tmp_path):
    collocs = [
        ("b item", 10, None),
        ("a item", 10, None),
        ("z item", 11, None),
    ]
    out_path = write_collocations_to_file("word", collocs, str(tmp_path))
    with open(out_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    # Sorted by freq desc, then phrase ascending (case-insensitive)
    assert lines == [
        "z item\t11",
        "a item\t10",
        "b item\t10",
    ]


def test_plain_text_from_html_line_cleans_spaces():
    html = "One  <b>  two</b>   three\n <i>four</i>"
    out = plain_text_from_html_line(html)
    assert out == "One two three four"


def test_tokenize_and_two_word_content_collocation():
    assert tokenize_phrase("Make a decision") == ["make", "a", "decision"]
    assert is_two_word_content_collocation("make decision") is True
    assert is_two_word_content_collocation("a decision") is False
    assert is_two_word_content_collocation("make a decision") is False


def test_normalize_collocation_phrase_strips_dot_marker():
    cleaned = normalize_collocation_phrase("decision", ".decision to")
    assert cleaned == "decision to"
    # does not alter non-exact tokens like ".decisions"
    unchanged = normalize_collocation_phrase("decision", ".decisions are hard")
    assert unchanged == ".decisions are hard"


def test_collocation_in_order_handles_be_inflections_and_target_once():
    phrase = "decision be"
    target = "decision"
    # Should match because 'be' lemma is realized as 'being' and target lemma occurs once
    s1 = "Our joint decision is being reviewed by the committee."
    assert _collocation_in_order(s1, phrase, target) is True

    # Should not match when target lemma appears twice
    s2 = "The decision is a decision that will be remembered."
    assert _collocation_in_order(s2, phrase, target) is False 