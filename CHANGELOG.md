# Changelog

## 0.3.1 — 2026-07-19

Reconciliation release. Two lines of work had diverged: the deep-perception
feature train (v0.2.0 loop/motif + key timeline, v0.3.0 stems/CLAP/madmom/
chords/loudness/caption + cut grid) and the SSRF security hardening that was
published to PyPI as 0.1.x. This release merges both so the repo and the
published package carry the full feature set **and** the security fix.

### Added

- All v0.2.0 / v0.3.0 deep-perception layers: loop/motif detection, per-section
  key timeline, narrative labels, stems (Demucs), CLAP embedding + tags,
  madmom rhythm/chords, loudness, Qwen2-Audio caption, and the cut grid.

### Security

- Preserved the full SSRF redirect + DNS-rebinding TOCTOU fix in `analyze()`
  URL fetching: per-hop address checks, DNS pinning to the vetted IP, manual
  redirect following capped at 5 hops, and a streamed body size cap. URL
  downloads go through `_fetch_url`, never `urllib.request.urlretrieve`.

## 0.1.6 — 2026-07-19

Version-drift reconciliation. PyPI 0.1.5 was published from a branch that
carried error-message info-disclosure hardening but *not* the SSRF
redirect/DNS-rebinding fix that landed locally as 0.1.4. This release merges
both lines of work (superseding both 0.1.4 and 0.1.5) so the published package
and the repo converge.

### Security

- Preserved the full SSRF redirect + DNS-rebinding TOCTOU fix from 0.1.4
  (per-hop address checks, DNS pinning, redirect cap, streamed size cap).
- Merged the info-disclosure hardening from 0.1.5: error messages no longer
  leak filesystem paths or full URIs — `_validate_file` reports only the
  basename, unsupported-scheme errors report only the scheme, and the path
  traversal error no longer echoes the attempted path.

### Dependencies

- Pinned minimum versions carried over from 0.1.5: `numpy>=1.24`,
  `openai-whisper>=20231117`.

### Tests

- Merged 0.1.5's tests on top of 0.1.4's redirect/rebinding suite:
  error-message path-leak assertions, `repr` safety, and frozen-dataclass
  immutability checks.

## 0.1.5 — PyPI only (superseded)

- Published to PyPI from a parallel branch. Added error-message
  info-disclosure hardening, `repr`/immutability tests, and dependency version
  pins, but did **not** include the SSRF redirect/DNS-rebinding fix. Folded
  into 0.1.6.

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
