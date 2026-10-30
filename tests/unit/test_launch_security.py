"""Launch-path security: login required, loopback by default (P2 auth/bind).

Both supported entry paths — the package ``draftforge.app.launch()`` / ``main()``
and the repo-root ``app.py`` (Hugging Face Spaces' app file) — are driven with a
FAKE ``build_ui`` whose Blocks records ``launch()`` kwargs, so no gradio import,
no store, and no real listener or share link is ever created. Every refusal is
asserted to happen before the UI/store is built and before any launch call.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from draftforge import app, config
from draftforge.config import LaunchConfigError

ROOT_APP = Path(__file__).resolve().parents[2] / "app.py"

# Clearly fake fixture values — never real credentials.
FIXTURE_PASSWORD = "fixture-login-not-a-real-secret"
FIXTURE_KEY = "fixture-key-not-real"

BAD_PASSWORDS = [None, "", "   ", "change-me", " Change-Me "]
BAD_IDS = ["unset", "empty", "whitespace", "example", "example-variant"]


class FakeBlocks:
    def __init__(self, recorder):
        self._recorder = recorder

    def launch(self, **kwargs):
        self._recorder["launches"].append(kwargs)
        return "launched"


@pytest.fixture
def ui(monkeypatch, tmp_path):
    """Fake build_ui + hermetic env/cwd; returns the call recorder."""
    recorder = {"builds": [], "launches": []}

    def fake_build_ui(**kwargs):
        recorder["builds"].append(kwargs)
        return FakeBlocks(recorder)

    monkeypatch.setattr(app, "build_ui", fake_build_ui)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    for name in ("APP_PASSWORD", "ANTHROPIC_API_KEY", "DRAFTFORGE_HOST",
                 "GRADIO_SERVER_NAME", "GRADIO_SHARE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    recorder["cwd"] = tmp_path
    return recorder


def _set_env(monkeypatch, password=FIXTURE_PASSWORD, key=FIXTURE_KEY, **extra):
    if password is not None:
        monkeypatch.setenv("APP_PASSWORD", password)
    if key is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def _assert_nothing_built(ui):
    assert ui["builds"] == []
    assert ui["launches"] == []
    assert not (ui["cwd"] / "data").exists()  # no store directory either


# --- package launch() ----------------------------------------------------------------


@pytest.mark.parametrize("password", BAD_PASSWORDS, ids=BAD_IDS)
def test_launch_refuses_bad_password_before_ui_store_or_listener(ui, monkeypatch, password):
    _set_env(monkeypatch, password=password)

    with pytest.raises(LaunchConfigError) as excinfo:
        app.launch()

    _assert_nothing_built(ui)
    if password and password.strip():
        assert password.strip() not in str(excinfo.value)


def test_launch_passes_supplied_password_unchanged_on_loopback(ui, monkeypatch):
    _set_env(monkeypatch)

    app.launch()

    assert len(ui["builds"]) == 1
    (kwargs,) = ui["launches"]
    assert kwargs["auth"] == ("draftforge", FIXTURE_PASSWORD)
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False


def test_launch_ignores_gradio_env_bind_and_share(ui, monkeypatch):
    _set_env(monkeypatch, GRADIO_SERVER_NAME="0.0.0.0", GRADIO_SHARE="True")

    app.launch()

    (kwargs,) = ui["launches"]
    assert (kwargs["server_name"], kwargs["share"]) == ("127.0.0.1", False)


@pytest.mark.parametrize("override", [{"auth": None}, {"auth": ("x", "y")},
                                      {"auth_dependency": lambda request: "anyone"}])
def test_launch_kwargs_cannot_replace_auth(ui, monkeypatch, override):
    _set_env(monkeypatch)

    with pytest.raises(LaunchConfigError, match="auth"):
        app.launch(**override)

    _assert_nothing_built(ui)


def test_launch_kwargs_cannot_skip_password_validation(ui, monkeypatch):
    _set_env(monkeypatch, password="change-me")

    with pytest.raises(LaunchConfigError):
        app.launch(server_name="0.0.0.0", share=True, server_port=7861)

    _assert_nothing_built(ui)


def test_launch_external_opt_in_keeps_login_and_reports_exposure(ui, monkeypatch, capsys):
    _set_env(monkeypatch)

    app.launch(server_name="0.0.0.0", server_port=7861)

    (kwargs,) = ui["launches"]
    assert kwargs == {
        "auth": ("draftforge", FIXTURE_PASSWORD),
        "server_name": "0.0.0.0",
        "share": False,
        "server_port": 7861,
    }
    err = capsys.readouterr().err
    assert "0.0.0.0" in err
    assert FIXTURE_PASSWORD not in err


def test_launch_env_opt_in_binds_externally(ui, monkeypatch):
    _set_env(monkeypatch, DRAFTFORGE_HOST="0.0.0.0")

    app.launch()

    (kwargs,) = ui["launches"]
    assert kwargs["server_name"] == "0.0.0.0"
    assert kwargs["auth"] == ("draftforge", FIXTURE_PASSWORD)


def test_launch_share_opt_in_still_requires_password(ui, monkeypatch):
    _set_env(monkeypatch, password=None)
    with pytest.raises(LaunchConfigError):
        app.launch(share=True)
    _assert_nothing_built(ui)

    _set_env(monkeypatch)
    app.launch(share=True)
    (kwargs,) = ui["launches"]
    assert (kwargs["server_name"], kwargs["share"]) == ("127.0.0.1", True)


def test_launch_requires_api_key_before_building(ui, monkeypatch):
    _set_env(monkeypatch, key=None)

    with pytest.raises(LaunchConfigError, match="ANTHROPIC_API_KEY"):
        app.launch()

    _assert_nothing_built(ui)


# --- package main() (python -m draftforge.app) ------------------------------------


def test_main_refuses_with_exit_2_and_no_secret_echo(ui, monkeypatch, capsys):
    _set_env(monkeypatch, password="change-me")

    assert app.main([]) == 2

    _assert_nothing_built(ui)
    err = capsys.readouterr().err
    assert "APP_PASSWORD" in err
    assert "change-me" not in err


def test_main_flags_map_to_validated_launch(ui, monkeypatch):
    _set_env(monkeypatch)

    assert app.main(["--host", "0.0.0.0", "--share"]) == 0

    (kwargs,) = ui["launches"]
    assert (kwargs["server_name"], kwargs["share"]) == ("0.0.0.0", True)
    assert kwargs["auth"] == ("draftforge", FIXTURE_PASSWORD)


# --- repo-root app.py (Hugging Face Spaces app file) -----------------------------


@pytest.fixture
def root_argv(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))  # app.py prepends src/

    def set_argv(*args):
        monkeypatch.setattr(sys, "argv", [str(ROOT_APP), *args])

    return set_argv


def test_root_app_import_builds_and_launches_nothing(ui, monkeypatch, root_argv):
    root_argv()
    _set_env(monkeypatch)

    namespace = runpy.run_path(str(ROOT_APP), run_name="draftforge_root_app_probe")

    _assert_nothing_built(ui)
    # No module-level Blocks for a discovery launcher to serve without login.
    assert "demo" not in namespace


@pytest.mark.parametrize("password", BAD_PASSWORDS, ids=BAD_IDS)
def test_root_app_main_refuses_bad_password(ui, monkeypatch, root_argv, password):
    root_argv()
    _set_env(monkeypatch, password=password)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT_APP), run_name="__main__")

    assert excinfo.value.code == 2
    _assert_nothing_built(ui)


def test_root_app_main_launches_loopback_with_supplied_password(ui, monkeypatch, root_argv):
    root_argv()
    _set_env(monkeypatch)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT_APP), run_name="__main__")

    assert excinfo.value.code == 0
    (kwargs,) = ui["launches"]
    assert kwargs == {
        "auth": ("draftforge", FIXTURE_PASSWORD),
        "server_name": "127.0.0.1",
        "share": False,
    }


def test_root_app_share_flag_still_requires_password(ui, monkeypatch, root_argv):
    root_argv("--share")
    _set_env(monkeypatch, password="   ")

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(ROOT_APP), run_name="__main__")

    assert excinfo.value.code == 2
    _assert_nothing_built(ui)


def test_root_app_share_and_host_flags_reach_launch(ui, monkeypatch, root_argv):
    root_argv("--share", "--host", "0.0.0.0")
    _set_env(monkeypatch)

    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT_APP), run_name="__main__")

    (kwargs,) = ui["launches"]
    assert (kwargs["server_name"], kwargs["share"]) == ("0.0.0.0", True)
    assert kwargs["auth"] == ("draftforge", FIXTURE_PASSWORD)
