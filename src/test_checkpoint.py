"""
Regression tests for infra_errors.JsonlCheckpoint / scan_resume — code review 2026-08-07 P0-4.

Covers the six required cases: wrong ID, duplicate ID, extra row, truncated tail, ERROR in the
middle, and a successful atomic rename. Plus the liblouis-selftest gating required by P0-3.

Run: python src/test_checkpoint.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from infra_errors import JsonlCheckpoint, scan_resume, is_infra_error

fails = []


def check(desc, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {desc}\n         got={got!r} want={want!r}")
    if not ok:
        fails.append(desc)


RECS = [{"id": "a"}, {"id": "b"}, {"id": "c"}]


def row(i, rid, resp="fine"):
    return {"record_index": i, "id": rid, "gold_answer": ["g"], "predicted": "p",
            "metrics": {"em": 1.0}, "full_response": resp}


def write(path, rows, raw_tail=None):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        if raw_tail is not None:
            f.write(raw_tail)


def tmp():
    d = tempfile.mkdtemp()
    return os.path.join(d, "out_details.jsonl")


print("=== scan_resume rejects misaligned / malformed rows ===")

p = tmp()
write(p, [row(0, "a"), row(1, "WRONG")])
valid, start = scan_resume(p, RECS, verbose=False)
check("wrong ID at index 1 -> resume from 1", (len(valid), start), (1, 1))

p = tmp()
write(p, [row(0, "a"), row(1, "a")])   # duplicate of row 0's id landing at index 1
valid, start = scan_resume(p, RECS, verbose=False)
check("duplicate ID -> resume from 1", (len(valid), start), (1, 1))

p = tmp()
write(p, [row(0, "a"), row(1, "b"), row(2, "c"), row(3, "d")])
valid, start = scan_resume(p, RECS, verbose=False)
check("extra row past dataset end -> start clamped to 3", (len(valid), start), (3, 3))

p = tmp()
write(p, [row(0, "a"), row(1, "b")], raw_tail='{"record_index": 2, "id": "c", "pred')
valid, start = scan_resume(p, RECS, verbose=False)
check("truncated final line dropped -> resume from 2", (len(valid), start), (2, 2))

p = tmp()
write(p, [row(0, "a"), row(1, "b", resp="ERROR: boom"), row(2, "c")])
valid, start = scan_resume(p, RECS, verbose=False)
check("ERROR in the middle -> resume from 1 (later rows discarded)", (len(valid), start), (1, 1))

p = tmp()
write(p, [{"record_index": 0, "id": "a", "predicted": "p"}])   # no metrics/gold/full_response
valid, start = scan_resume(p, RECS, verbose=False)
check("missing required keys -> resume from 0", (len(valid), start), (0, 0))

p = tmp()
check("missing file -> start 0", scan_resume(p, RECS, verbose=False), ([], 0))

print("\n=== record_index catches misalignment even when IDs are empty ===")
norecs = [{}, {}, {}]           # no `id` field: expected_id falls back to the index
p = tmp()
write(p, [row(0, 0), row(1, 5)])   # id 5 at index 1 is wrong
valid, start = scan_resume(p, norecs, verbose=False)
check("empty-ID dataset still detects bad row", (len(valid), start), (1, 1))

print("\n=== running run writes only *.partial; final appears atomically ===")
p = tmp()
ck = JsonlCheckpoint(p, RECS)
valid, start = ck.resume()
check("fresh resume -> 0", (len(valid), start), (0, 0))
with ck:
    ck.write(0, RECS[0], ["g"], "p", {"em": 1.0}, "fine")
    ck.write(1, RECS[1], ["g"], "p", {"em": 1.0}, "fine")
    check("final path absent mid-run", os.path.exists(p), False)
    check("partial path present mid-run", os.path.exists(p + ".partial"), True)

print("\n=== finalize() refuses an incomplete file ===")
try:
    ck.finalize()
    check("short file raises", "no raise", "ValueError")
except ValueError as e:
    check("short file raises ValueError", "2 rows != 3 records" in str(e), True)
check("final still absent after failed finalize", os.path.exists(p), False)

print("\n=== finalize() refuses a file containing an ERROR row ===")
p2 = tmp()
ck2 = JsonlCheckpoint(p2, RECS)
ck2.resume()
with ck2:
    ck2.write(0, RECS[0], ["g"], "p", {"em": 1.0}, "fine")
    ck2.write(1, RECS[1], ["g"], "p", {"em": 0.0}, "ERROR: boom")
    ck2.write(2, RECS[2], ["g"], "p", {"em": 1.0}, "fine")
try:
    ck2.finalize()
    check("ERROR row raises", "no raise", "ValueError")
except ValueError as e:
    check("ERROR row raises ValueError", "ERROR record" in str(e), True)
check("final absent when an ERROR row exists", os.path.exists(p2), False)

print("\n=== successful atomic rename ===")
with ck:                                  # reopen and finish the earlier run
    ck.write(2, RECS[2], ["g"], "p", {"em": 1.0}, "fine")
rows = ck.finalize()
check("finalize returns all rows", len(rows), 3)
check("final file now exists", os.path.exists(p), True)
check("partial removed by os.replace", os.path.exists(p + ".partial"), False)
check("row IDs aligned", [r["id"] for r in rows], ["a", "b", "c"])
check("record_index aligned", [r["record_index"] for r in rows], [0, 1, 2])

print("\n=== resume of a completed final file is a no-op SKIP ===")
ck3 = JsonlCheckpoint(p, RECS)
valid, start = ck3.resume()
check("complete final -> start == len(records)", (len(valid), start), (3, 3))
check("finalize re-publishes cleanly", len(ck3.finalize()), 3)

print("\n=== infra classifier (P0-2) ===")
check("endpoint connect is infra", is_infra_error("Could not connect to the endpoint URL"), True)
check("expired token is infra", is_infra_error("ExpiredTokenException"), True)
check("throttling is infra", is_infra_error("ThrottlingException: Rate exceeded"), True)
check("validation error is NOT infra", is_infra_error("ValidationException: bad input"), False)

print("\n=== P0-3: liblouis self-test gating in the evaluator's startup path ===")
import inspect
import evaluate_braille as EB
src = inspect.getsource(EB.main)
check("main() calls liblouis_selftest", "liblouis_selftest()" in src, True)
check("main() uses the shared liblouis gate", "configs_require_liblouis(config_list)" in src, True)
gate = EB.configs_require_liblouis
check("EN-EN only -> no liblouis needed", gate(["EN-EN"]), False)
check("G1-EN (english output) -> no liblouis needed", gate(["G1-EN"]), False)
check("EN-G1 -> liblouis required", gate(["EN-G1"]), True)
check("G1-G1 -> liblouis required", gate(["EN-EN", "G1-G1"]), True)
check("FULLBR-G1 prompt translation -> liblouis required", gate(["FULLBR-G1"]), True)
check("all configs -> liblouis required", gate(list(EB.CONFIGS.keys())), True)
# self-test must raise (fail closed) when the tables are unreachable.
old = os.environ.get("LOUIS_TABLEPATH")
os.environ["LOUIS_TABLEPATH"] = "/nonexistent-table-path-for-test"
try:
    try:
        import louis
    except ModuleNotFoundError:
        louis = None
        print("  [SKIP] native liblouis is not installed; real self-test runs in the full environment")
    if louis is not None:
        louis.liblouis.lou_free()      # drop cached tables so the bad path takes effect
        try:
            EB.liblouis_selftest()
            check("self-test fails closed on a bad table path", "no raise", "raise")
        except Exception:
            check("self-test fails closed on a bad table path", "raised", "raised")
finally:
    if old is None:
        os.environ.pop("LOUIS_TABLEPATH", None)
    else:
        os.environ["LOUIS_TABLEPATH"] = old
    if "louis" in locals() and louis is not None:
        try:
            louis.liblouis.lou_free()
        except Exception:
            pass

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("ALL CHECKPOINT TESTS PASSED")
