"""
Shared infrastructure-error classifier + a strict, ID-aligned, atomically-completed JSONL
checkpoint for long API runs.

Two review findings motivate this module:

1. (2026-08-07 P0-2) The CoT runner's marker list contained "connection" but an actual SDK
   message was "Could not connect to the endpoint URL", which does NOT contain the substring
   "connection". 225 endpoint-connection failures were serialized into predicted/full_response and
   scored 0, silently deflating hotpot/2wiki. `is_infra_error` centralizes the classification so no
   future experiment re-implements a fragile list.

2. (2026-08-07 P0-4) Resume was by line COUNT, so a file with a wrong/duplicated/extra row resumed
   at the wrong offset and a partially-written file was indistinguishable from a finished one.
   `JsonlCheckpoint` validates every existing row against the source records index-by-index, writes
   only to `*.partial.jsonl` while running, and produces the final path solely via an atomic
   `os.replace` after a full re-read and re-validation. A final file therefore always means
   "complete, error-free, exactly aligned".
"""
import json
import os

# Substrings (matched case-insensitively) that indicate an INFRASTRUCTURE failure — the call must
# be paused/retried, never written as a scored-0 model response. Kept broad on purpose; a false
# "infra" classification only pauses the run, which is safe.
INFRA_MARKERS = (
    "credential", "expiredtoken", "expired token",
    "could not connect", "connect", "connection",          # endpoint connection failures
    "readtimeout", "read timeout", "connecttimeout", "timed out", "timeout",
    "throttl", "toomanyrequests", "rate exceeded", "rate limit",
    "serviceunavailable", "service unavailable",
    "internalserver", "internal server", "500", "502", "503", "504",
    "endpointconnectionerror", "endpoint url",
)

# Every result row must carry these keys, or it is not a usable record.
REQUIRED_KEYS = ("record_index", "id", "gold_answer", "predicted", "metrics", "full_response")


def is_infra_error(err) -> bool:
    """True if the exception/text is an infrastructure failure (pause), not a model/validator error."""
    s = str(err).lower()
    return any(m in s for m in INFRA_MARKERS)


def is_error_record(rec: dict) -> bool:
    """True if a stored result record is an ERROR row (needs a re-run, not a re-score)."""
    fr = rec.get("full_response", "")
    if isinstance(fr, str) and fr.startswith("ERROR"):
        return True
    if rec.get("error"):
        return True
    return False


def expected_id(rec, index):
    """The `id` a result row must carry for source record `rec` at position `index`.

    Mirrors the writer exactly. Datasets whose records have no `id` (or a legitimately empty one)
    fall back to the positional index, which is why `record_index` is also stored and checked —
    an empty ID can never be a unique alignment key on its own.
    """
    rid = rec.get("id") if isinstance(rec, dict) else None
    return rid if rid not in (None, "") else index


def validate_row(rec, index, source_rec):
    """Return None if `rec` is a valid, aligned, error-free result row; else a reason string."""
    if not isinstance(rec, dict):
        return "row is not a JSON object"
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        return f"missing keys {missing}"
    if rec["record_index"] != index:
        return f"record_index {rec['record_index']!r} != expected {index}"
    exp = expected_id(source_rec, index)
    if rec["id"] != exp:
        return f"id {rec['id']!r} != expected {exp!r}"
    if not isinstance(rec["metrics"], dict) or not rec["metrics"]:
        return "metrics missing or empty"
    if rec["gold_answer"] is None:
        return "gold_answer is null"
    if is_error_record(rec):
        return "ERROR record"
    return None


def scan_resume(path, records, verbose=True):
    """Inspect an existing JSONL and return (valid_rows, start_index) for a strict resume.

    `valid_rows` is the longest prefix in which row i is a well-formed, error-free record whose
    `record_index`/`id` match `records[i]`. `start_index` is the first index that must be (re)run.
    Everything at or after the first problem — wrong ID, duplicate landing at the wrong index,
    missing field, ERROR row, truncated/corrupt line, or an EXTRA row past the end of `records` —
    is discarded rather than trusted. Never returns start > len(records).
    """
    valid = []
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return [], 0

    def stop(i, reason):
        if verbose:
            print(f"    resume: discarding {path} from row {i} ({reason})")
        return valid, len(valid)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            return stop(i, "blank line")
        if i >= len(records):
            return stop(i, f"extra row past end of dataset ({len(records)} records)")
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return stop(i, "unparseable/truncated line")
        reason = validate_row(rec, i, records[i])
        if reason:
            return stop(i, reason)
        valid.append(rec)
    return valid, len(valid)


class JsonlCheckpoint:
    """Crash-safe, ID-aligned result writer.

    Rows are appended to `<final>.partial` and flushed per item (fsync every `fsync_every` rows).
    `finalize()` re-reads the partial from disk, re-validates every row against the source records,
    and only then `os.replace`s it onto the final path — so the final file cannot exist in a
    partial, misaligned, or ERROR-containing state.
    """

    def __init__(self, final_path, records, fsync_every=20):
        self.final = str(final_path)
        self.partial = self.final + ".partial"
        self.records = records
        self.fsync_every = fsync_every
        self._f = None
        self._since_sync = 0

    def resume(self):
        """Validate what exists, seed the partial with the good prefix, return the start index.

        Prefers an existing final file (a completed earlier run) and otherwise falls back to a
        leftover partial from an interrupted run.
        """
        source = self.final if os.path.exists(self.final) else self.partial
        valid, start = scan_resume(source, self.records)
        with open(self.partial, "w") as f:
            for r in valid:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return valid, start

    def __enter__(self):
        self._f = open(self.partial, "a")
        return self

    def __exit__(self, *exc):
        if self._f:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None
        return False

    def write(self, index, source_rec, gold, predicted, metrics, full_response):
        row = {"record_index": index, "id": expected_id(source_rec, index),
               "gold_answer": gold, "predicted": predicted,
               "metrics": metrics, "full_response": full_response}
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        self._since_sync += 1
        if self._since_sync >= self.fsync_every:
            os.fsync(self._f.fileno())
            self._since_sync = 0

    def finalize(self):
        """Re-read + re-validate the partial, then atomically publish it. Raises on any problem."""
        rows = []
        with open(self.partial) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    raise ValueError(f"{self.partial}: blank line at row {i}")
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{self.partial}: unparseable row {i}: {e}") from e
                if i >= len(self.records):
                    raise ValueError(f"{self.partial}: extra row {i}, dataset has {len(self.records)}")
                reason = validate_row(rec, i, self.records[i])
                if reason:
                    raise ValueError(f"{self.partial}: row {i} invalid: {reason}")
                rows.append(rec)
        if len(rows) != len(self.records):
            raise ValueError(f"{self.partial}: {len(rows)} rows != {len(self.records)} records")
        os.replace(self.partial, self.final)
        return rows
