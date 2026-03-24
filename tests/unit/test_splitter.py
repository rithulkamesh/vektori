"""Unit tests for sentence splitter."""

from vektori.ingestion.splitter import _merge_short_sentences, split_sentences


def test_basic_split():
    text = "Hello world. This is a test. It should split correctly."
    result = split_sentences(text)
    assert len(result) >= 1


def test_empty_string():
    assert split_sentences("") == []


def test_whitespace_only():
    assert split_sentences("   ") == []


def test_merge_short_fragment():
    sentences = ["The user called.", "And asked about their balance."]
    merged = _merge_short_sentences(sentences)
    assert len(merged) == 1
    assert "And asked" in merged[0]


def test_merge_conjunction_start():
    sentences = ["She preferred WhatsApp.", "But also accepted SMS."]
    merged = _merge_short_sentences(sentences)
    assert len(merged) == 1


def test_no_merge_long_sentences():
    sentences = [
        "The borrower clearly stated their preference for WhatsApp communication.",
        "The agent confirmed and updated the contact preferences accordingly.",
    ]
    merged = _merge_short_sentences(sentences)
    assert len(merged) == 2


def test_single_sentence():
    text = "I prefer WhatsApp over email for all communication."
    result = split_sentences(text)
    assert len(result) == 1
    assert result[0] == text
