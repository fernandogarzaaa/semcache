# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Email
**fernandogarzaaa@gmail.com** with a description, reproduction steps and the
affected version or commit. You can expect an acknowledgement within a few days.

## Threat model (short)

semcache sees every prompt, every answer and, in pass-through mode, the callers'
provider keys. See the README "Security" section for how the cache is isolated
per caller, how to require proxy tokens, and why the admin API is off by
default. Run it on a private network or behind TLS; it does not terminate TLS.

## Supported versions

Only the default branch receives security fixes.
