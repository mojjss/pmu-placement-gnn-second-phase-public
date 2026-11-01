# Security policy

This is research software, not a certified operational power-system tool.
Security reports for the current main branch and latest release are reviewed on
a best-effort basis. There is no guaranteed response time or long-term support
commitment for older versions.

## Reporting a vulnerability

Do not publish exploit details, credentials, private network data, or malicious
files in a public issue.

If GitHub private vulnerability reporting is enabled, use
[Report a vulnerability](https://github.com/mojjss/pmu-placement-gnn-second-phase-public/security/advisories/new).
Include the affected version or commit, relevant command/file, expected impact,
and a minimal reproduction that contains no sensitive information.

If that private form is unavailable, open a public issue titled
**Request for a private security reporting channel**, with no vulnerability
details. Wait for a private reporting route before sharing the report.
Creating this policy does not itself enable GitHub's private reporting feature.

Ordinary numerical disagreements, reproducibility questions, and installation
bugs belong in normal issues unless they also create a security vulnerability.

## Handling research artifacts

- Use checkpoints and datasets only from sources you trust.
- Checkpoint SHA-256 verification detects changed bytes; it does not establish
  that an unknown checkpoint is safe or scientifically correct.
- The checkpoint loader uses `weights_only=True`; retain that restriction.
- Saved NPZ inspection avoids loading pickled objects. Do not enable pickle
  loading to work around an unfamiliar input.
- Keep new outputs in fresh directories and preserve saved evidence.
- Keep dependencies updated in a separate test environment and record version
  changes; do not relabel a new environment as the historical one.

Please allow time for investigation and coordinated disclosure before publishing
details of a confirmed security issue.
