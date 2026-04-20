"""Tests for the Gradio launcher diagnostics."""

from __future__ import annotations

import pytest

from finqa_bot.ui import gradio_app


class _FakeDemo:
    def __init__(self, launch_result=None, launch_error: Exception | None = None) -> None:
        self.launch_result = launch_result
        self.launch_error = launch_error
        self.launch_kwargs: dict[str, object] | None = None

    def queue(self) -> _FakeDemo:
        return self

    def launch(self, **kwargs: object) -> object:
        self.launch_kwargs = kwargs
        if self.launch_error is not None:
            raise self.launch_error
        return self.launch_result


class _FakeSettings:
    finqa_api_port = 8001


def test_launch_enables_share_in_colab(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _FakeDemo(launch_result=(object(), "http://127.0.0.1:7860", "https://demo.gradio.live"))
    emitted: list[str] = []

    monkeypatch.setattr(gradio_app, "_build_ui", lambda api_base: demo)
    monkeypatch.setattr(gradio_app, "_is_colab", lambda: True)
    monkeypatch.setattr(gradio_app, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(gradio_app, "_emit_launch_diagnostic", emitted.append)

    gradio_app.launch(share=False, server_port=7860)

    assert demo.launch_kwargs is not None
    assert demo.launch_kwargs["share"] is True
    assert any("share=True" in line for line in emitted)
    assert any("share_url=https://demo.gradio.live" in line for line in emitted)


def test_launch_logs_failure_and_reraises(monkeypatch: pytest.MonkeyPatch) -> None:
    demo = _FakeDemo(launch_error=RuntimeError("share failed"))
    emitted: list[str] = []

    monkeypatch.setattr(gradio_app, "_build_ui", lambda api_base: demo)
    monkeypatch.setattr(gradio_app, "_is_colab", lambda: False)
    monkeypatch.setattr(gradio_app, "get_settings", lambda: _FakeSettings())
    monkeypatch.setattr(gradio_app, "_emit_launch_diagnostic", emitted.append)

    with pytest.raises(RuntimeError, match="share failed"):
        gradio_app.launch(share=True, server_port=7860)

    assert any("Gradio launch failed" in line for line in emitted)
