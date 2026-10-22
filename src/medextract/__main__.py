"""Allow ``python -m medextract ...`` as an alternative to the ``medextract`` command."""

import sys

from medextract.cli import main

sys.exit(main())
