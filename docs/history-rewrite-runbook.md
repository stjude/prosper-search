# History Rewrite Runbook (Secrets/Data Purge)

This runbook captures the steps used to purge leaked secrets and generated sample
outputs from repository history before public release.

## What Was Purged

- Leaked Azure API key literal from historical revisions of:
  - `prosper/llm/summarizer/run_summarizer_user.sh`
  - `prosper/llm/summarizer/run_summarizer_dev.sh`
- All historical content under `examples/output/`

## Rewrite Command Pattern

```bash
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --tree-filter '
  rm -rf examples/output
  for f in prosper/llm/summarizer/run_summarizer_user.sh prosper/llm/summarizer/run_summarizer_dev.sh; do
    if [ -f "$f" ]; then
      sed -i "s/<LEAKED_KEY>/REDACTED_REMOVED_SECRET/g" "$f"
    fi
  done
' --prune-empty --tag-name-filter cat -- --all
```

## Post-Rewrite Cleanup

```bash
git for-each-ref --format='%(refname)' refs/original/ | while read -r ref; do
  git update-ref -d "$ref"
done
git stash clear
git reflog expire --expire=now --all
git gc --prune=now
```

## Verification Commands

```bash
git log --all --oneline -S"<LEAKED_KEY>"
git log --all --oneline -- "examples/output"
```

Both commands must return no commits.

## Required Coordination After Rewrite

1. Rotate any exposed key immediately (even after purge).
2. Force-push rewritten refs:
   - `git push --force-with-lease origin main`
3. Tell collaborators to re-sync:
   - Fresh clone recommended, or hard reset to rewritten `origin/main`.
4. Re-run secret scanning and code scanning on the rewritten history.
