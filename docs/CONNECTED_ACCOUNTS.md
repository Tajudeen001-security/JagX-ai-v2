# Connected accounts — secure integration design
Never accept third-party passwords or expose provider tokens to clients.
## Connection lifecycle
- A signed-in user needs a stable identity from a verified session/JWT subject. An API key is a service credential, not a durable human identity.
- Start OAuth 2.0 Authorization Code + PKCE server-side; use state and a PKCE verifier; validate issuer, redirect URI, audience, expiry, and scopes.
- Exchange codes server-side. Store tokens only in a managed secret store or encrypted database using a server-side key; never commit, return, or log tokens.
- Ordinary metadata should contain provider, external subject, scopes, connection time, and health status only.
- Support list/status, reconnect, revoke/disconnect, least privilege, and clear consent.
- Add provider integrations only after official API and OAuth configuration is available.
## Current status
The backend currently uses API keys and has no verified end-user JWT identity or OAuth client registrations. Do not represent accounts as connected until identity, redirect URLs, client IDs, scopes, and secure token storage are configured. No paid API key is required for this framework, but providers may impose quotas.
