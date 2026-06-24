"""Tests for evaluation metrics."""
import numpy as np
from src.evaluation import (
    compute_wer,
    parse_sentiment_json,
    parse_keywords_json,
    compute_keyword_f1,
    compute_sentiment_accuracy,
)


def test_wer_identical_is_zero():
    assert compute_wer("hello world", "hello world") == 0.0


def test_wer_substitution_is_one():
    assert compute_wer("hello", "hallo") == 1.0


def test_wer_deletion_is_correct():
    assert compute_wer("hello world", "hello") == 0.5


def test_wer_insertion_is_correct():
    assert compute_wer("hello", "hello world") == 1.0


def test_wer_empty_reference():
    assert compute_wer("", "something") == 1.0


def test_parse_sentiment_json_standard():
    result = parse_sentiment_json('{"sentiment": "positive", "confidence": 0.92}')
    assert result["sentiment"] == "positive"
    assert result["confidence"] == 0.92


def test_parse_sentiment_json_with_markdown():
    result = parse_sentiment_json('```json\n{"sentiment": "neutral", "confidence": 0.75}\n```')
    assert result["sentiment"] == "neutral"


def test_parse_keywords_json_standard():
    result = parse_keywords_json('["AI", "climate change", "innovation"]')
    assert result == ["AI", "climate change", "innovation"]


def test_compute_keyword_f1_perfect():
    ref = ["AI", "climate", "energy"]
    pred = ["AI", "climate", "energy"]
    f1, p, r = compute_keyword_f1(ref, pred)
    assert f1 == 1.0
    assert p == 1.0


def test_compute_keyword_f1_partial():
    ref = ["AI", "climate", "energy", "future"]
    pred = ["AI", "climate", "technology"]
    f1, p, r = compute_keyword_f1(ref, pred)
    assert 0.4 < f1 < 0.8


def test_compute_sentiment_accuracy_perfect():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "neutral"]
    acc, f1 = compute_sentiment_accuracy(y_true, y_pred)
    assert acc == 1.0
    assert f1 == 1.0
