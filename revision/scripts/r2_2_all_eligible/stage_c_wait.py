#!/usr/bin/env python3
"""Block (silently) until every named lsfq pool has no backlog and no active task; print counts.
Reads only the pools' state.json (no LSF calls); the watcher drives the pools."""
import json
import os
import sys
import time


def counts(name):
    p = os.path.expanduser(f"~/.lsfq/pools/{name}/state.json")
    if not os.path.exists(p):
        return None
    try:
        st = json.load(open(p))
    except ValueError:
        return None
    c = {}
    for t in st["tasks"].values():
        c[t["state"]] = c.get(t["state"], 0) + 1
    return c


def main(names, poll=60, limit=None):
    t0 = time.time()
    while True:
        cs = {n: counts(n) for n in names}
        if all(c is not None and not c.get("backlog") and not c.get("active") for c in cs.values()):
            print(time.strftime("%H:%M:%S"), json.dumps(cs, sort_keys=True), flush=True)
            return 0
        if limit and time.time() - t0 > limit:
            print(time.strftime("%H:%M:%S"), "WAIT_LIMIT", json.dumps(cs, sort_keys=True), flush=True)
            return 1
        time.sleep(poll)


if __name__ == "__main__":
    lim = float(os.environ.get("WAIT_LIMIT_S", "0")) or None
    sys.exit(main(sys.argv[1:], limit=lim))
