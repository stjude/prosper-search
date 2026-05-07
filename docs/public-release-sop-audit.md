# Public Release SOP Audit

This checklist tracks each requirement of the St. Jude "Making a GitHub
Repository Public" SOP against the current state of this repository. It is the
authoritative status doc - update it whenever a row changes state.

Legend: PASS = requirement met; PENDING = needs human / GitHub-side action;
FAIL = remediation work required and not yet done.

## 1. Pre-Publication Readiness Checklist (SOP section 1)

- PASS - No hardcoded secrets in tracked files. The Azure OpenAI API key
  was removed from `prosper/llm/summarizer/run_summarizer_dev.sh` and
  `prosper/llm/summarizer/run_summarizer_user.sh`; both now require the
  caller to export `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and
  `AZURE_OPENAI_API_KEY`.
- PENDING - **Leaked key fully removed from history AND rotated.** A history
  rewrite using `git-filter-repo` will purge the literal from all reachable
  commits before push. Verify after rewrite with
  `git log --all -S"<leaked-key>"` returning empty. Per SOP, the key must
  also be rotated in Azure Portal regardless of rewrite, since exposure has
  occurred.
- PASS (current tree) / PENDING (history) - No internal-only absolute paths
  in tracked files. `scripts/download_data.py`, `scripts/get_slide_dimensions.py`,
  `scripts/test_ot_heatmap.py`, `scripts/test_ot_modules.py`, `data/README.md`,
  `README.md`, and the summarizer scripts now use environment variables or
  repo-relative paths. The `REDACTED_INTERNAL_PATH/...` prefix still
  appears in commits `f74c7a7`, `74d0607`, `8a83b35` and will be redacted
  by the same history rewrite.
- PASS - No proprietary or third-party code without licensing clearance.
  Direct dependencies in `setup.py` / `pyproject.toml` are all permissive
  (BSD, MIT, Apache 2.0, HPND), with one exception: `openslide-python` is
  LGPL 2.1. LGPL is compatible with Apache 2.0 when linked dynamically (we
  import it; we do not vendor or modify it). No GPL dependencies present;
  Apache 2.0 remains valid.
- PASS - `README.md` present with description, installation, usage, and
  contribution link.
- PASS - `CONTRIBUTING.md` present; placeholder GitHub URLs replaced with
  `https://github.com/stjude/prosper-search`.
- PASS - `CODE_OF_CONDUCT.md` present (Contributor Covenant v2.1) and
  linked from `README.md` Community and Security section.
- PASS - `SECURITY.md` present, defines responsible disclosure contact and
  secret-handling expectations, linked from `README.md`.
- PASS - `.gitignore` excludes build artifacts, local configs, and data
  files; now also excludes `examples/output/`, `examples/llm_foundry_credentials.sh`,
  and `TechDocs-SOP*.pdf`.
- PASS - CI/CD does not expose internal credentials. `.github/workflows/`
  uses only the default `GITHUB_TOKEN`; no hardcoded secrets.
- PASS - `examples/output/` no longer tracked. Removed from index. PENDING
  for history - the artifacts (incl. a 1.9 MB heatmap PNG) still live in
  commit `07c4825` and will be purged by `git-filter-repo --path
  examples/output --invert-paths`.
- PASS - No oversized binaries. `data/fig/overview.png` (4.7 MB) is the
  only larger tracked file and is a legitimate README asset; large model /
  data files use Git LFS via `.gitattributes`.

## 2. Branch Protection Configuration (SOP section 2)

- PENDING (GitHub UI) - Protect `main`:
  - Require pull request reviews before merging (1 approver; owner bypass
    permitted under SOP single-developer clause).
  - Dismiss stale approvals on new commits.
  - Require status checks to pass: `basic-check`, `CodeQL`, `gitleaks`,
    `Dependabot`.
  - Require branches to be up to date before merging.
  - Require linear history.
  - Disallow force pushes (after the one-time history rewrite push).
  - Disallow deletions.
  - Require CODEOWNERS review for matched paths.
  - Require conversation resolution before merging.

## 3. Code Scanning, Secret Scanning, Dependency Review (SOP section 3)

- PASS (workflow committed) / PENDING (GitHub-side enable) - CodeQL.
  `.github/workflows/codeql.yml` runs on push, PR, and weekly schedule with
  the `security-and-quality` suite for Python. Code scanning must also be
  enabled in Settings -> Code security and analysis.
- PASS (workflow committed) - gitleaks PR check at
  `.github/workflows/gitleaks.yml` for belt-and-braces secret scanning.
- PENDING (GitHub UI) - GitHub Secret Scanning + Push Protection toggled on
  in Settings -> Code security and analysis.
- PASS (config committed) / PENDING (GitHub-side enable) - Dependabot.
  `.github/dependabot.yml` covers `pip` and `github-actions`. Dependabot
  alerts and security updates also need to be enabled in repo settings.

## 4. License Selection (SOP section 4)

- PASS - `LICENSE` is Apache License 2.0, copyright 2026 St. Jude
  Children's Research Hospital. `NOTICE` file added per Apache convention.
  `setup.py`, `pyproject.toml`, and `README.md` updated to advertise
  Apache-2.0.
- PASS - License compatibility checked. No GPL dependencies; LGPL via
  `openslide-python` is compatible. Apache 2.0 is appropriate per SOP
  default recommendation.

## 5. Approval Process (SOP section 5)

- PENDING - Repository owner submits publication request to
  `GitHubAdministrators@STJUDE.ORG` with: repo URL, this completed
  checklist, CodeQL run summary, license (Apache-2.0), and business
  justification.
- PENDING - Security review of scanning results, branch protection, and
  secret scanning by GitHub Admin / Security.
- PENDING - Legal/Compliance review of dependency licenses.

## 6. Publication (SOP section 6)

- PENDING - GitHub Admin: Settings -> General -> Danger Zone -> Change
  repository visibility -> Make public.
- PENDING - Verify accessibility in incognito window.
- PENDING - Update internal catalogs / documentation.

## 7. Post-Publication Monitoring (SOP section 7)

- PENDING - Keep code scanning, secret scanning, and Dependabot enabled.
- PENDING - Maintain branch protection rules.
- PENDING - Quarterly compliance review.

## Outstanding Local Action Items (in order)

1. Run history rewrite (see [history-rewrite-runbook.md](history-rewrite-runbook.md))
   to purge the leaked Azure key, `REDACTED_INTERNAL_PATH` prefix, and
   `examples/output/` from all reachable history.
2. Repository owner: rotate the exposed Azure OpenAI key in Azure Portal.
3. Force-push the rewritten `pub-dev`, open PR to `main`, fast-forward
   `main` after review.
4. Repository admin: complete the GitHub-side PENDING items above.
5. Submit publication request per SOP section 5.
