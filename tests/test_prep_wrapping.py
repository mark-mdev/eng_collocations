import pytest

from scrape_collocations import wrap_full_chunk_prep


def test_wrap_prep_target_order():
    sent = "They showed commitment to quality in every step."
    # phrase contains prep and target; target word is 'commitment'
    out = wrap_full_chunk_prep(sent, "commitment to", "commitment")
    assert out is not None
    assert "{{c1::commitment to}}" in out


def test_wrap_target_prep_order():
    sent = "Their success was due to commitment and hard work."
    out = wrap_full_chunk_prep(sent, "to commitment", "commitment")
    assert out is not None
    assert "{{c1::to commitment}}" in out


def test_wrap_prep_returns_none_when_no_match():
    sent = "People admired her dedication and passion."
    out = wrap_full_chunk_prep(sent, "commitment to", "commitment")
    assert out is None 