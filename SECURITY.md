# Security Policy

## Supported Versions

Security fixes are applied to the `main` branch.

## Reporting a Vulnerability

If you discover a security vulnerability, do not open a public issue.

- Email: GitHubAdministrators@stjude.org
- Include: affected component, reproduction steps, potential impact, and any
  known mitigations.

We will acknowledge receipt within 5 business days and coordinate triage and
remediation according to internal security response procedures.

## Secret Handling

- Never commit credentials, tokens, keys, PHI/PII, or internal infrastructure
  details.
- Use environment or repository secrets for CI/CD and runtime configuration.
- Rotate exposed credentials immediately if accidental disclosure occurs.
