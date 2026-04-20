"""Pydantic round-trip tests for the grammar-constrained answer envelope."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from finqa_bot.types import AnswerEnvelope, Step


def test_envelope_roundtrip_json() -> None:
    env = AnswerEnvelope(
        program=[
            Step(op="subtract", args=[120.0, 100.0], source="table_1"),
            Step(op="divide", args=["#0", 100.0], source="table_1"),
            Step(op="multiply", args=["#1", "const_100"], source="const"),
        ],
        answer_value=20.0,
        answer_form="percent",
        scale="units",
        grounded_numbers=[120.0, 100.0],
        confidence=0.9,
        rationale="Revenue growth from 2023 to 2024.",
    )
    dumped = env.model_dump()
    as_json = json.dumps(dumped, default=str)
    back = AnswerEnvelope.model_validate_json(as_json)
    assert back == env


def test_envelope_rejects_extra_fields() -> None:
    payload = {
        "program": [],
        "answer_value": 0.0,
        "answer_form": "decimal",
        "scale": "units",
        "grounded_numbers": [],
        "confidence": 0.5,
        "rationale": "",
        "some_extra_field": True,
    }
    with pytest.raises(ValidationError):
        AnswerEnvelope.model_validate(payload)


def test_envelope_rejects_invalid_op() -> None:
    payload = {
        "program": [{"op": "multiply_then_panic", "args": [1, 2], "source": "x"}],
        "answer_value": 0.0,
        "answer_form": "decimal",
        "scale": "units",
        "grounded_numbers": [],
        "confidence": 0.5,
        "rationale": "",
    }
    with pytest.raises(ValidationError):
        AnswerEnvelope.model_validate(payload)


def test_step_normalizes_int_arg_to_float() -> None:
    step = Step.model_validate({"op": "add", "args": [1, 2.5, "#0"], "source": ""})
    assert step.args == [1.0, 2.5, "#0"]


def test_envelope_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        AnswerEnvelope.model_validate({
            "program": [],
            "answer_value": 0.0,
            "answer_form": "decimal",
            "scale": "units",
            "grounded_numbers": [],
            "confidence": 1.5,
            "rationale": "",
        })
