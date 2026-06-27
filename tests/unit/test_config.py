import pytest

from draftforge.config import Settings


def test_load_reads_all_fields_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("MODEL_FAST", "model-fast-override")
    monkeypatch.setenv("MODEL_SMART", "model-smart-override")
    monkeypatch.setenv("APP_PASSWORD", "hunter2")

    settings = Settings.load()

    assert settings.api_key == "sk-ant-test"
    assert settings.model_fast == "model-fast-override"
    assert settings.model_smart == "model-smart-override"
    assert settings.app_password == "hunter2"


def test_load_applies_defaults_when_optional_env_missing(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("MODEL_FAST", raising=False)
    monkeypatch.delenv("MODEL_SMART", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)

    settings = Settings.load()

    assert settings.api_key == "sk-ant-test"
    assert settings.model_fast == "claude-haiku-4-5-20251001"
    assert settings.model_smart == "claude-opus-4-8"
    assert settings.app_password == "change-me"


def test_load_raises_keyerror_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(KeyError):
        Settings.load()
