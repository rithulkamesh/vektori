"""Cook raw LoCoMo data into a LongMemEval-compatible benchmark format."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")


def _session_numbers(conversation: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    for key, value in conversation.items():
        match = _SESSION_KEY_RE.match(key)
        if match and isinstance(value, list):
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def _normalize_role(turn: dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    speaker = str(turn.get("speaker") or turn.get("name") or "").strip()
    if speaker and speaker == speaker_a:
        return "user"
    if speaker and speaker == speaker_b:
        return "assistant"

    role = str(turn.get("role") or "").strip().lower()
    if role in {"user", "assistant", "system"}:
        return role
    return "user"


def _turn_text(turn: dict[str, Any], include_image_fields: bool) -> str:
    content = str(turn.get("content") or turn.get("text") or "").strip()
    if not include_image_fields:
        return content

    extras: list[str] = []
    img_url = str(turn.get("img_url") or "").strip()
    blip_caption = str(turn.get("blip_caption") or "").strip()
    img_search_query = str(turn.get("img_search_query") or "").strip()
    if img_url:
        extras.append(f"[image_url] {img_url}")
    if blip_caption:
        extras.append(f"[image_caption] {blip_caption}")
    if img_search_query:
        extras.append(f"[image_query] {img_search_query}")

    if not extras:
        return content
    suffix = "\n".join(extras)
    return f"{content}\n{suffix}" if content else suffix


def _build_conversation_index(
    sample_id: str,
    conversation: dict[str, Any],
    include_image_fields: bool,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    speaker_a = str(conversation.get("speaker_a") or "").strip()
    speaker_b = str(conversation.get("speaker_b") or "").strip()

    sessions: list[dict[str, Any]] = []
    dia_to_session_id: dict[str, str] = {}

    for session_num in _session_numbers(conversation):
        session_key = f"session_{session_num}"
        date_key = f"session_{session_num}_date_time"
        turns = conversation.get(session_key) or []
        if not isinstance(turns, list):
            continue

        session_id = f"locomo_{sample_id}_s{session_num}"
        date_value = str(conversation.get(date_key) or "").strip()

        messages: list[dict[str, str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            content = _turn_text(turn, include_image_fields=include_image_fields)
            if not content:
                continue

            messages.append(
                {
                    "role": _normalize_role(turn, speaker_a=speaker_a, speaker_b=speaker_b),
                    "content": content,
                }
            )

            dia_id = str(turn.get("dia_id") or "").strip()
            if dia_id:
                dia_to_session_id[dia_id] = session_id

        if messages:
            sessions.append(
                {
                    "session_num": session_num,
                    "session_id": session_id,
                    "session_date": date_value,
                    "messages": messages,
                }
            )

    return sessions, dia_to_session_id


def _question_type(qa_item: dict[str, Any]) -> str:
    value = (
        qa_item.get("category")
        or qa_item.get("category_label")
        or qa_item.get("type")
        or "unknown"
    )
    return str(value).strip() or "unknown"


def _evidence_session_ids(evidence: Any, dia_to_session_id: dict[str, str]) -> list[str]:
    if not isinstance(evidence, list):
        return []

    seen: set[str] = set()
    session_ids: list[str] = []
    for item in evidence:
        dia_id = str(item).strip()
        if not dia_id:
            continue
        session_id = dia_to_session_id.get(dia_id)
        if session_id and session_id not in seen:
            seen.add(session_id)
            session_ids.append(session_id)
    return session_ids


def _truncate_sessions_for_evidence(
    sessions: list[dict[str, Any]],
    answer_session_ids: list[str],
) -> list[dict[str, Any]]:
    if not answer_session_ids:
        return sessions

    max_evidence_session_num: int | None = None
    evidence_set = set(answer_session_ids)
    for s in sessions:
        if s["session_id"] in evidence_set:
            session_num = int(s["session_num"])
            if max_evidence_session_num is None or session_num > max_evidence_session_num:
                max_evidence_session_num = session_num

    if max_evidence_session_num is None:
        return sessions
    return [s for s in sessions if int(s["session_num"]) <= max_evidence_session_num]


def cook_locomo_dataset(
    input_path: str | Path,
    output_path: str | Path,
    history_policy: str = "all_sessions",
    include_image_fields: bool = False,
) -> dict[str, int]:
    """Convert LoCoMo raw JSON into LongMemEval-style benchmark records.

    Args:
        input_path: Path to raw ``locomo10.json``.
        output_path: Output path for cooked dataset JSON.
        history_policy: ``all_sessions`` or ``through_latest_evidence``.
        include_image_fields: Include image URL/caption/query in message text.

    Returns:
        Summary stats with keys ``samples``, ``qa_total``, ``qa_cooked``, ``qa_skipped``.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    with in_path.open(encoding="utf-8") as f:
        samples = json.load(f)

    if not isinstance(samples, list):
        raise ValueError("LoCoMo input must be a JSON list of conversation samples")

    if history_policy not in {"all_sessions", "through_latest_evidence"}:
        raise ValueError(
            "history_policy must be one of: all_sessions, through_latest_evidence"
        )

    cooked: list[dict[str, Any]] = []
    qa_skipped = 0
    qa_total = 0

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        sample_id = str(sample.get("sample_id") or "").strip()
        conversation = sample.get("conversation")
        qa_items = sample.get("qa")
        if not sample_id or not isinstance(conversation, dict) or not isinstance(qa_items, list):
            continue

        sessions, dia_to_session_id = _build_conversation_index(
            sample_id=sample_id,
            conversation=conversation,
            include_image_fields=include_image_fields,
        )
        if not sessions:
            continue

        for qa_idx, qa_item in enumerate(qa_items, start=1):
            qa_total += 1
            if not isinstance(qa_item, dict):
                qa_skipped += 1
                continue

            question = str(qa_item.get("question") or "").strip()
            answer = str(qa_item.get("answer") or "").strip()
            if not question or not answer:
                qa_skipped += 1
                continue

            answer_session_ids = _evidence_session_ids(
                evidence=qa_item.get("evidence"),
                dia_to_session_id=dia_to_session_id,
            )

            selected_sessions = sessions
            if history_policy == "through_latest_evidence":
                selected_sessions = _truncate_sessions_for_evidence(
                    sessions=sessions,
                    answer_session_ids=answer_session_ids,
                )
            if not selected_sessions:
                qa_skipped += 1
                continue

            question_date = str(selected_sessions[-1].get("session_date") or "")
            question_id = f"locomo_{sample_id}_q{qa_idx}"

            cooked.append(
                {
                    "question_id": question_id,
                    "question_type": _question_type(qa_item),
                    "question": question,
                    "question_date": question_date,
                    "answer": answer,
                    "answer_session_ids": answer_session_ids,
                    "haystack_dates": [s["session_date"] for s in selected_sessions],
                    "haystack_session_ids": [s["session_id"] for s in selected_sessions],
                    "haystack_sessions": [s["messages"] for s in selected_sessions],
                    "metadata": {
                        "source": "locomo",
                        "sample_id": sample_id,
                        "qa_index": qa_idx,
                        "evidence_dialog_ids": qa_item.get("evidence") or [],
                    },
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cooked, f, indent=2, ensure_ascii=False)

    return {
        "samples": len(samples),
        "qa_total": qa_total,
        "qa_cooked": len(cooked),
        "qa_skipped": qa_skipped,
    }


def _download_locomo_dataset(source_url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source_url) as response:
        payload = response.read()
    with output_path.open("wb") as f:
        f.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cook LoCoMo raw dataset into LongMemEval-compatible format"
    )
    parser.add_argument("--input", default="data/locomo10.json")
    parser.add_argument("--output", default="data/locomo10_cooked.json")
    parser.add_argument(
        "--history-policy",
        choices=["all_sessions", "through_latest_evidence"],
        default="all_sessions",
        help=(
            "all_sessions: include full conversation context per QA; "
            "through_latest_evidence: include sessions only up to latest evidence session"
        ),
    )
    parser.add_argument(
        "--include-image-fields",
        action="store_true",
        help="Append image URL/caption/query fields into turn content when present",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download LoCoMo raw JSON if --input file does not exist",
    )
    parser.add_argument(
        "--source-url",
        default=DEFAULT_LOCOMO_URL,
        help="Source URL used when --download-if-missing is enabled",
    )

    args = parser.parse_args()
    in_path = Path(args.input)

    if not in_path.exists():
        if not args.download_if_missing:
            raise FileNotFoundError(
                f"Input dataset not found: {in_path}. Use --download-if-missing to fetch it."
            )
        print(f"Downloading LoCoMo dataset from {args.source_url} -> {in_path}")
        _download_locomo_dataset(source_url=args.source_url, output_path=in_path)

    stats = cook_locomo_dataset(
        input_path=in_path,
        output_path=args.output,
        history_policy=args.history_policy,
        include_image_fields=args.include_image_fields,
    )

    print(f"Cooked LoCoMo dataset written to {args.output}")
    print(
        "Summary: "
        f"samples={stats['samples']}, "
        f"qa_total={stats['qa_total']}, "
        f"qa_cooked={stats['qa_cooked']}, "
        f"qa_skipped={stats['qa_skipped']}"
    )


if __name__ == "__main__":
    main()
