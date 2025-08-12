import os
import pytest

# Ensure spaCy model exists before tests
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    pytest.skip("spaCy model en_core_web_sm not installed", allow_module_level=True)

from scrape_collocations import html_line_to_cloze_text


@pytest.mark.parametrize(
    "sentence,target,expected_contains",
    [
        ("She decided yesterday.", "decide", "{{c1::decided}}"),
        ("He is deciding now.", "decide", "{{c1::deciding}}"),
        ("They will decide later.", "decide", "{{c1::decide}}"),
    ],
)
def test_inflection_wrapping(sentence, target, expected_contains):
    out = html_line_to_cloze_text(sentence, target)
    assert expected_contains in out
    assert out.count("{{c1::") == 1


@pytest.mark.parametrize(
    "sentence,target,expected",
    [
        ("We must make a decision.", "decision", "make {{c1::a decision}}."),
        ("They announced the decision.", "decision", "announced {{c1::the decision}}."),
        ("She made an effort.", "effort", "made {{c1::an effort}}."),
    ],
)
def test_article_inclusion(sentence, target, expected):
    out = html_line_to_cloze_text(sentence, target)
    # normalize spacing for robust matching
    assert expected in out


def test_no_fallback_when_not_found():
    sentence = "He chose quickly."
    target = "decide"
    out = html_line_to_cloze_text(sentence, target)
    # unchanged (no cloze) because lemma not present
    assert "{{c1::" not in out


@pytest.mark.parametrize(
    "sentence,target",
    [
        ("I will decide and then decide again.", "decide"),
        ("Decisions, decisions, decisions.", "decision"),
    ],
)
def test_wrap_only_first_occurrence(sentence, target):
    out = html_line_to_cloze_text(sentence, target)
    assert out.count("{{c1::") == 1


def test_preexisting_font_tags_are_removed():
    html = 'Please <font color="red">make</font> a <font color="red">decision</font> today.'
    out = html_line_to_cloze_text(html, "decision")
    assert "<font" not in out
    assert "{{c1::" in out 