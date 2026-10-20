"""The Senses - one route per medium.

The four members are Eye, Ear, Lip and Tongue: Eye perceives vision in, Ear
perceives audio in, Lip speaks out, Tongue carries language across the other
three. `router.route()` is the only entry point the
rest of one-cli uses.
"""

from onecli.senses import capabilities, ear, eye, lip, logbook, router, tongue  # noqa: F401
from onecli.senses.router import Routed, route, route_history  # noqa: F401
