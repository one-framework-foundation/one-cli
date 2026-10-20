"""HTTP opening that never carries credentials to a host we did not ask for.

urllib's default redirect handler copies every request header, including
`Authorization` and `x-api-key`, onto the redirected request. A gateway or CDN
that answers 302 to another origin would then receive the key. Same-origin
redirects keep their headers. A cross-origin one keeps only a short allowlist
of headers that cannot carry a credential; everything else the caller set is
dropped, including configured extras (`ONE_PROVIDER_HEADERS`) whose names we
cannot know in advance. A redirect from https to http is refused outright.
"""
from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request

# The only headers that survive a cross-origin redirect. `Range` keeps a
# resumed model download working through a CDN redirect.
SAFE_CROSS_ORIGIN_HEADERS = frozenset({
    "accept", "accept-encoding", "accept-language", "content-type", "range", "user-agent"})


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    return scheme, (parts.hostname or "").lower(), parts.port or _DEFAULT_PORTS.get(scheme)


class StripCredentialsOnRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return new
        old_origin, new_origin = _origin(req.full_url), _origin(new.full_url)
        if old_origin[0] == "https" and new_origin[0] != "https":
            raise urllib.error.HTTPError(
                newurl, code, f"refused redirect from https to {new_origin[0]}",
                headers, fp)
        if new_origin != old_origin:
            for name in list(new.headers):
                if name.lower() not in SAFE_CROSS_ORIGIN_HEADERS:
                    del new.headers[name]
        return new


def install() -> None:
    """Make every plain `urllib.request.urlopen` in this process use the handler.

    Process-wide on purpose: the providers, model downloads, probes and the
    vendored drop-in reader all call `urlopen` directly, and one opener covers
    each of them, including call sites added later. A call that passes its
    own `context=` or builds its own opener is not covered.
    """
    urllib.request.install_opener(
        urllib.request.build_opener(StripCredentialsOnRedirect()))
