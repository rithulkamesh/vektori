"""Unit tests for LoCoMo dataset cooking."""

import json

from benchmarks.locomo.cook_locomo import cook_locomo_dataset


def _sample_locomo_payload() -> list[dict]:
    return [
        {
            "sample_id": "abc123",
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024/01/01 09:00",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "d1", "text": "I bought a red bike."},
                    {"speaker": "Bob", "dia_id": "d2", "text": "Nice, where from?"},
                ],
                "session_2_date_time": "2024/01/02 11:00",
                "session_2": [
                    {"speaker": "Alice", "dia_id": "d3", "text": "From Trek store."},
                    {"speaker": "Bob", "dia_id": "d4", "text": "Great brand."},
                ],
            },
            "qa": [
                {
                    "question": "What color bike did Alice buy?",
                    "answer": "Red",
                    "category": "single-hop",
                    "evidence": ["d1"],
                },
                {
                    "question": "Where did Alice buy the bike?",
                    "answer": "Trek store",
                    "category": "single-hop",
                    "evidence": ["d3"],
                },
            ],
        }
    ]


def test_cook_locomo_all_sessions(tmp_path):
    input_path = tmp_path / "locomo.json"
    output_path = tmp_path / "locomo_cooked.json"
    input_path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")

    stats = cook_locomo_dataset(
        input_path=input_path,
        output_path=output_path,
        history_policy="all_sessions",
    )

    assert stats["samples"] == 1
    assert stats["qa_total"] == 2
    assert stats["qa_cooked"] == 2
    assert stats["qa_skipped"] == 0

    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(rows) == 2

    first = rows[0]
    assert first["question_id"] == "locomo_abc123_q1"
    assert first["question_type"] == "single-hop"
    assert first["answer_session_ids"] == ["locomo_abc123_s1"]
    assert first["haystack_session_ids"] == ["locomo_abc123_s1", "locomo_abc123_s2"]
    assert first["haystack_sessions"][0][0]["role"] == "user"
    assert first["haystack_sessions"][0][1]["role"] == "assistant"


def test_cook_locomo_through_latest_evidence(tmp_path):
    input_path = tmp_path / "locomo.json"
    output_path = tmp_path / "locomo_cooked.json"
    input_path.write_text(json.dumps(_sample_locomo_payload()), encoding="utf-8")

    cook_locomo_dataset(
        input_path=input_path,
        output_path=output_path,
        history_policy="through_latest_evidence",
    )

    rows = json.loads(output_path.read_text(encoding="utf-8"))
    assert rows[0]["haystack_session_ids"] == ["locomo_abc123_s1"]
    assert rows[0]["question_date"] == "2024/01/01 09:00"

    assert rows[1]["haystack_session_ids"] == ["locomo_abc123_s1", "locomo_abc123_s2"]
    assert rows[1]["question_date"] == "2024/01/02 11:00"
