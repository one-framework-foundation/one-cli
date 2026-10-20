"""one-cli: a terminal agent loop over the raw model wire.

Drives any OpenAI- or Anthropic-shaped endpoint through its own tool loop, with
no vendor CLI in the path.
"""

__version__ = "0.1.1"

from onecli import netio as _netio

_netio.install()
