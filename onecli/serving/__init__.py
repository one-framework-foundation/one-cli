"""Local serving: get the weights, measure how to serve them, run the lane.

Three questions this package answers for a host, in order:

1. What can this box run?          `one doctor`
2. Where do the weights come from? `one models`
3. What is the fastest way to      `one tune`, and then `one serve`, which
   serve them HERE?                launches with the measured winner.

The through-line is that the last question is answered by measurement on the
machine in front of you, never by a default someone wrote down. Speculative
decoding with a drafter can be much faster than serving without one, and no
amount of reading tells you whether it is on this box - the sweep does.
"""

from . import catalog, hostfacts, pull, recipes, serve, tune  # noqa: F401

__all__ = ["catalog", "hostfacts", "pull", "recipes", "serve", "tune"]
