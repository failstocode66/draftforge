"""Entrypoint for Hugging Face Spaces (and local `python app.py`).

HF Spaces runs this file (``app_file: app.py`` in the README frontmatter). It puts
``src/`` on the path, exposes a module-level ``demo`` for Spaces to discover, and —
when run as the main script (which is how Spaces launches it) — starts the server
gated by ``APP_PASSWORD`` (username ``draftforge``).

The real handlers/UI live in :mod:`draftforge.app`; this is a thin launcher kept at
the repo root only because that is where Spaces looks for the app file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
os.makedirs("data", exist_ok=True)  # the SQLite store's parent dir

from draftforge.app import _auth, build_ui  # noqa: E402

# Module-level Blocks so Hugging Face Spaces can discover the app.
demo = build_ui()

if __name__ == "__main__":
    import argparse

    from draftforge.config import Settings

    parser = argparse.ArgumentParser(description="Launch DraftForge (auth-gated).")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public gradio.live link (still password-gated) "
        "so a remote tester can reach a locally-run instance.",
    )
    args, _ = parser.parse_known_args()

    settings = Settings.load()  # requires ANTHROPIC_API_KEY
    demo.launch(
        server_name="0.0.0.0", share=args.share, auth=_auth(settings.app_password)
    )
