import pytest

from draftforge import config
from draftforge.config import (
    LaunchConfigError,
    Settings,
    load_launch_config,
    validate_app_password,
)

# Clearly fake fixture values — never real credentials.
FIXTURE_PASSWORD = "fixture-login-not-a-real-secret"
FIXTURE_KEY = "fixture-key-not-real"


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Keep any real .env (repo root or an ancestor dir) out of these tests."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)


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


def test_load_applies_defaults_but_never_a_default_app_password(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("MODEL_FAST", raising=False)
    monkeypatch.delenv("MODEL_SMART", raising=False)
    monkeypatch.delenv("APP_PASSWORD", raising=False)

    settings = Settings.load()

    assert settings.api_key == "sk-ant-test"
    assert settings.model_fast == "claude-haiku-4-5-20251001"
    assert settings.model_smart == "claude-opus-4-8"
    # The documented example ("change-me") is no longer a fallback: unset stays
    # unset, and the web app refuses to launch on it.
    assert settings.app_password is None
    with pytest.raises(LaunchConfigError):
        validate_app_password(settings.app_password)


def test_load_raises_keyerror_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(KeyError):
        Settings.load()


def test_settings_repr_hides_key_and_password(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FIXTURE_KEY)
    monkeypatch.setenv("APP_PASSWORD", FIXTURE_PASSWORD)

    text = repr(Settings.load())

    assert FIXTURE_KEY not in text
    assert FIXTURE_PASSWORD not in text


# --- APP_PASSWORD validation -------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, "", "   ", "\t\n", "change-me", " change-me ", "CHANGE-ME"],
    ids=["unset", "empty", "spaces", "whitespace", "example", "example-padded", "example-upper"],
)
def test_validate_app_password_rejects_missing_blank_and_example(value):
    with pytest.raises(LaunchConfigError, match="APP_PASSWORD") as excinfo:
        validate_app_password(value)
    if value and value.strip():
        assert value.strip() not in str(excinfo.value)  # never echo the value


@pytest.mark.parametrize("value", [FIXTURE_PASSWORD, f"  {FIXTURE_PASSWORD}  ", "hunter2"])
def test_validate_app_password_returns_supplied_value_unchanged(value):
    assert validate_app_password(value) == value


# --- launch configuration ----------------------------------------------------------


def _env(**overrides):
    env = {"APP_PASSWORD": FIXTURE_PASSWORD, "ANTHROPIC_API_KEY": FIXTURE_KEY}
    env.update(overrides)
    return {k: v for k, v in env.items() if v is not None}


def test_launch_config_defaults_to_loopback_without_share():
    cfg = load_launch_config(environ=_env())

    assert cfg.server_name == "127.0.0.1"
    assert cfg.share is False
    assert cfg.external is False
    assert cfg.app_password == FIXTURE_PASSWORD


def test_launch_config_ignores_gradio_env_fallbacks():
    cfg = load_launch_config(
        environ=_env(GRADIO_SERVER_NAME="0.0.0.0", GRADIO_SHARE="True")
    )

    assert cfg.server_name == "127.0.0.1"
    assert cfg.share is False


def test_launch_config_external_bind_needs_explicit_opt_in():
    from_env = load_launch_config(environ=_env(DRAFTFORGE_HOST="0.0.0.0"))
    explicit = load_launch_config(server_name="0.0.0.0", environ=_env())
    shared = load_launch_config(share=True, environ=_env())

    assert (from_env.server_name, from_env.external) == ("0.0.0.0", True)
    assert (explicit.server_name, explicit.external) == ("0.0.0.0", True)
    assert (shared.server_name, shared.share, shared.external) == ("127.0.0.1", True, True)


def test_explicit_server_name_wins_over_env():
    cfg = load_launch_config(server_name="127.0.0.1", environ=_env(DRAFTFORGE_HOST="0.0.0.0"))

    assert cfg.server_name == "127.0.0.1"


@pytest.mark.parametrize("server_name", ["0.0.0.0", "::"])
@pytest.mark.parametrize("password", [None, "", "  ", "change-me"])
def test_external_opt_in_still_requires_a_valid_password(server_name, password):
    with pytest.raises(LaunchConfigError, match="APP_PASSWORD"):
        load_launch_config(server_name=server_name, environ=_env(APP_PASSWORD=password))
    with pytest.raises(LaunchConfigError, match="APP_PASSWORD"):
        load_launch_config(share=True, environ=_env(APP_PASSWORD=password))


@pytest.mark.parametrize("host", ["", "   ", "0.0.0.0 ", "bad host"])
def test_blank_or_malformed_host_is_refused_not_defaulted(host):
    with pytest.raises(LaunchConfigError):
        load_launch_config(environ=_env(DRAFTFORGE_HOST=host))
    with pytest.raises(LaunchConfigError):
        load_launch_config(server_name=host, environ=_env())


@pytest.mark.parametrize("share", ["true", 1, "yes"])
def test_share_must_be_a_real_bool(share):
    with pytest.raises(LaunchConfigError, match="share"):
        load_launch_config(share=share, environ=_env())


def test_launch_config_requires_api_key_without_echoing_secrets():
    with pytest.raises(LaunchConfigError, match="ANTHROPIC_API_KEY") as excinfo:
        load_launch_config(environ=_env(ANTHROPIC_API_KEY=None))

    assert FIXTURE_PASSWORD not in str(excinfo.value)


def test_launch_config_repr_hides_password():
    cfg = load_launch_config(environ=_env())

    assert FIXTURE_PASSWORD not in repr(cfg)


def test_launch_config_default_environ_loads_dotenv_then_reads_os_environ(monkeypatch):
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: calls.append(1) or False)
    monkeypatch.setenv("APP_PASSWORD", FIXTURE_PASSWORD)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FIXTURE_KEY)
    monkeypatch.delenv("DRAFTFORGE_HOST", raising=False)

    cfg = load_launch_config()

    assert calls == [1]
    assert cfg.app_password == FIXTURE_PASSWORD
    assert cfg.server_name == "127.0.0.1"
