"""Allow ``python -m gofish config.ini`` invocation."""

from .cli import main

raise SystemExit(main())
