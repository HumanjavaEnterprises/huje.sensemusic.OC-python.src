# Changelog

## 0.1.4 — 2026-07-17

### Security

- **SSRF hardening (redirect + DNS-rebinding TOCTOU).** `analyze()` URL fetching
  previously validated the hostname once and then downloaded with
  `urllib.request.urlretrieve()`, which performed its own second DNS resolution
  and silently followed 3xx redirects. An attacker-controlled server could
  302-redirect to an internal address (e.g. cloud metadata at
  `169.254.169.254`) or rebind DNS to loopback between check and fetch.
  The download now uses a custom pinned-connection fetcher:
  - The private/loopback/link-local/reserved/multicast/unspecified address
    check runs on **every** redirect hop before connecting.
  - The TCP connection is made to the exact IP that passed the check (DNS
    pinning) while the Host header and TLS SNI/certificate verification still
    use the original hostname — closing the rebinding TOCTOU.
  - Redirects are followed manually and capped at 5 hops.
  - Bodies are streamed with a hard size cap (Content-Length pre-check plus
    an enforced limit while streaming).
  - IPv4-mapped IPv6 addresses (`::ffff:127.0.0.1`) are unwrapped before the
    block-list check.
- DNS failures (`socket.gaierror`) are now re-raised as `ValueError` for
  consistent error typing.
- Fixed the resolver default port: 80 for `http`, 443 for `https` (previously
  always 443).

### Tests

- New security tests with a live local HTTP server: redirect-to-internal-IP
  blocked, redirect-to-`file://` blocked, redirect-loop cap, oversized body
  (streaming) rejection, oversized Content-Length rejection, happy-path fetch
  through a public-style redirect, and unresolvable-hostname error typing.

## 0.1.3

- Prior release (see git history).
