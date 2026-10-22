import pytest

from adaptive_scheduler.cli import main
from adaptive_scheduler.config import ConfigurationError, LLMProvider, Settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "APSCHED_ANTHROPIC_API_KEY", "APSCHED_LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    # Never read a developer's real .env during tests.
    monkeypatch.setitem(Settings.model_config, "env_file", None)


def test_auto_without_key_falls_back_to_mock() -> None:
    assert Settings().resolve_provider() is LLMProvider.MOCK


def test_auto_with_key_uses_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert Settings().resolve_provider() is LLMProvider.ANTHROPIC


def test_blank_key_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert Settings().resolve_provider() is LLMProvider.MOCK


def test_explicit_mock_ignores_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("APSCHED_LLM_PROVIDER", "mock")
    assert Settings().resolve_provider() is LLMProvider.MOCK


def test_explicit_anthropic_without_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APSCHED_LLM_PROVIDER", "anthropic")
    with pytest.raises(ConfigurationError):
        Settings().resolve_provider()


def test_key_is_never_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out
    assert "sk-super-secret" not in out


def test_cli_provider_override(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "mock", "info"]) == 0
    assert "resolved provider  : mock" in capsys.readouterr().out


def test_cli_reports_config_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--provider", "anthropic", "info"]) == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
