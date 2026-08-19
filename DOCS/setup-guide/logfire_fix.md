# Logfire not sending data — fix log

## Symptom
`logfire.configure()` ran fine and local console spans showed up, but nothing appeared in the Logfire UI.

## Causes (two, found in order)

1. **`LOGFIRE_TOKEN` was empty** in `.env`. Fix: generate a write token at
   https://logfire.pydantic.dev (project `rahulvansh66/medi-agent`) → Settings → Write tokens,
   and set `LOGFIRE_TOKEN` in `.env`.

2. **SSL handshake failing** (`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`)
   when reaching `logfire-us.pydantic.dev`. Cause: Avast's HTTPS-scanning feature had set three
   user-level env vars (persisted in `HKCU:\Environment`, so they survive reboots and app restarts)
   pointing every Python/Node/curl tool at Avast's own root cert:
   - `SSL_CERT_FILE`
   - `REQUESTS_CA_BUNDLE`
   - `NODE_EXTRA_CA_CERTS`

   Fix:
   - Disable Avast's HTTPS scanning (Settings → General → "Enable HTTPS scanning").
   - Remove the stale env vars from the registry (`Remove-ItemProperty -Path 'HKCU:\Environment' -Name 'SSL_CERT_FILE','REQUESTS_CA_BUNDLE'`).
   - **Reboot** — Windows caches the user env block at login in `explorer.exe`; new terminals/apps
     inherit that cached block, so reopening a window/IDE alone doesn't pick up registry changes.
     A reboot (or full log off/on) *after* the registry cleanup is required.

## Verifying it's fixed
Run any ingestion job or hit `/query`, then check https://logfire-us.pydantic.dev/rahulvansh66/medi-agent
for the corresponding spans within a few seconds.
