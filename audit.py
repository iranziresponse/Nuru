"""An append-only trail of who did what to which document, when.

Deliberately logs metadata, not content: the event, the document's name
and inferred type, the action taken and its destination, and the
outcome — never the extracted field values themselves. Nuru's whole
privacy design is built around not keeping a second permanent copy of
financial data lying around; an audit log that captured full field
contents would work against that. What's recorded here is enough to
answer "did someone send this document somewhere, when, and did it
succeed" without doing that.

"Who" is currently the requester's IP address, not a named person —
Nuru has one shared access password today, not per-user accounts. A
real identity here needs that project first; this is what's honestly
available without it.
"""

import datetime
import json
import os

AUDIT_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nuru_cache", "audit.log")


def record(event, ip=None, **fields):
    """Append one entry. Never raises — a logging failure must not break
    the request that triggered it."""
    entry = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        "ip": ip,
        **fields,
    }
    try:
        os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
    return entry


def read_recent(limit=200):
    """Most-recent-first, capped at `limit` entries."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries.reverse()
    return entries
