"""Entrypoint for Hugging Face Spaces (and local `python app.py`).

HF Spaces runs this file (``app_file: app.py`` in the README frontmatter) as the
main script. It puts ``src/`` on the path and hands off to
:func:`draftforge.app.main` — the same validated launch path as
``python -m draftforge.app``. That validates ``APP_PASSWORD`` (required; no
default, example values refused) and the bind options BEFORE the UI, the store,
or any listener is created, then serves the login-gated app (username
``draftforge``) on ``127.0.0.1`` unless the operator explicitly opts in with
``--host`` / ``DRAFTFORGE_HOST`` or ``--share``.

Importing this module builds nothing: there is deliberately no module-level
``demo``, because a discovered-and-served Blocks object would bypass the
validated launch (and its login). The real handlers/UI live in
:mod:`draftforge.app`; this thin launcher stays at the repo root only because
that is where Spaces looks for the app file.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from draftforge.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
