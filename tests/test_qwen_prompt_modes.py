"""Checks for the controlled Qwen prompt-placement ablation."""
from src.direct_qwen import (
    SYSTEM_PROMPTS,
    SYSTEM_TURN_PROMPTS,
    SYSTEM_USER_INSTRUCTION,
    build_conversation,
)


def test_user_mode_keeps_task_prompt_with_audio() -> None:
    conversation = build_conversation("intent", "user")
    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert conversation[0]["content"][0]["type"] == "audio"
    assert conversation[0]["content"][1]["text"] == SYSTEM_PROMPTS["intent"]


def test_default_mode_uses_system_turn() -> None:
    conversation = build_conversation("intent")
    assert conversation[0] == {
        "role": "system",
        "content": SYSTEM_TURN_PROMPTS["intent"],
    }


def test_system_mode_moves_only_task_prompt_to_system() -> None:
    conversation = build_conversation("intent", "system")
    assert conversation[0] == {
        "role": "system",
        "content": SYSTEM_TURN_PROMPTS["intent"],
    }
    assert conversation[1]["role"] == "user"
    assert conversation[1]["content"][0]["type"] == "audio"
    assert conversation[1]["content"][1]["text"] == SYSTEM_USER_INSTRUCTION
