# -*- coding: utf-8 -*-
"""verify_freeze.py -- the two integrity layers of this release, which make DIFFERENT claims.

LAYER 1, the pre-registration freeze (7 files).
    Hashed before the first participant. The claim is the one that cannot be made any other way:
    the hypotheses, the exclusion rules, the stopping rule, the analyser and the served stimulus
    set could not have been chosen after seeing data. The freeze record's path column comes from
    the machine it was made on, and several files were renamed for readability here, so
    `sha256sum -c` cannot resolve it -- this layer matches by CONTENT, which is what carries the
    claim.

LAYER 2, release integrity (every shipped file).
    Hashed at publication, not before the study. It says only "this is the bundle as published,
    unchanged". It is NOT a pre-registration claim and must never be read as one. It exists
    because Layer 1 deliberately covers only 7 files, while a reader downloading a zip has no
    other way to tell whether the other 40-odd arrived intact.

    SHA256SUMS is written in the standard format, so nobody has to trust this script:
        sha256sum -c SHA256SUMS

WHAT NEITHER LAYER PROVES. The freeze is self-attested: PLAN_FREEZE.frozen_at is a local
timestamp, not a third-party one. Anyone reproducing this should treat the git tag and the
archive DOI as the external evidence of when the tree existed, and read PLAN_FREEZE.frozen_at as
the authors' own record.

RUN: python verify_freeze.py              check both layers
     python verify_freeze.py --write-sums (re)generate SHA256SUMS for the current tree
Exit status 0 only if every layer that can be checked passes.
"""
import hashlib
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(ROOT, "prereg", "PLAN_FREEZE.sha256")
SUMS = os.path.join(ROOT, "SHA256SUMS")
SKIP_DIRS = {".git", "__pycache__", "figures_out"}


def sha(path):
    h = hashlib.sha256()
    with io.open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def shipped():
    """Every file in the release, relative and forward-slashed, except SHA256SUMS itself."""
    out = []
    for dp, dn, fns in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for fn in fns:
            p = os.path.join(dp, fn)
            rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
            if rel != "SHA256SUMS":
                out.append(rel)
    return sorted(out)


def write_sums():
    rows = ["%s  %s\n" % (sha(os.path.join(ROOT, rel)), rel) for rel in shipped()]
    io.open(SUMS, "w", encoding="utf-8", newline="\n").writelines(rows)
    print("wrote SHA256SUMS: %d files" % len(rows))
    return 0


def layer1():
    want = {}
    for ln in io.open(REC, encoding="utf-8"):
        m = re.match(r"([0-9a-f]{64})\s+\*?(.+)", ln.strip())
        if m:
            want[m.group(1)] = os.path.basename(m.group(2).strip())

    have = {}
    for dp, dn, fns in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        for fn in fns:
            try:
                have.setdefault(sha(os.path.join(dp, fn)),
                                os.path.relpath(os.path.join(dp, fn), ROOT).replace(os.sep, "/"))
            except Exception:                                          # noqa: BLE001
                continue

    stamp = io.open(os.path.join(ROOT, "prereg", "PLAN_FREEZE.frozen_at"),
                    encoding="utf-8").read().strip().splitlines()[0]
    print("LAYER 1 -- PRE-REGISTRATION FREEZE")
    print("  frozen at %s, before the first participant" % stamp)
    print("  %d files; matched by content, because several were renamed for this release\n" % len(want))
    ok = 0
    for h, name in sorted(want.items(), key=lambda x: x[1]):
        here = have.get(h)
        print("    %-30s %s" % (name, ("OK   " + here) if here else "MISSING from this repository"))
        ok += bool(here)
    print("\n  %d/%d verify" % (ok, len(want)))
    print("  Claim: these could not have been chosen after seeing data.")
    return ok == len(want)


def layer2():
    print("\nLAYER 2 -- RELEASE INTEGRITY")
    if not os.path.isfile(SUMS):
        print("  SHA256SUMS is not present: this layer is not being claimed.")
        print("  Generate it at publication with:  python verify_freeze.py --write-sums")
        return None
    want = {}
    for ln in io.open(SUMS, encoding="utf-8"):
        m = re.match(r"([0-9a-f]{64})\s+\*?(.+)", ln.strip())
        if m:
            want[m.group(2).strip()] = m.group(1)
    here = set(shipped())
    missing = sorted(set(want) - here)
    extra = sorted(here - set(want))
    changed = [rel for rel in sorted(set(want) & here)
               if sha(os.path.join(ROOT, rel)) != want[rel]]
    print("  %d files recorded, %d present" % (len(want), len(here)))
    for label, rows in (("MISSING", missing), ("CHANGED", changed), ("NOT RECORDED", extra)):
        for rel in rows[:20]:
            print("    %-14s %s" % (label, rel))
        if len(rows) > 20:
            print("    %-14s ... and %d more" % (label, len(rows) - 20))
    good = len(want) - len(missing) - len(changed)
    print("\n  %d/%d verify%s" % (good, len(want), "" if not extra else
                                  "   (%d file(s) present but not recorded)" % len(extra)))
    print("  Claim: this is the bundle as published. NOT a pre-registration claim.")
    print("  Independent check, no need to trust this script:  sha256sum -c SHA256SUMS")
    return not (missing or changed or extra)


def main():
    if "--write-sums" in sys.argv:
        return write_sums()
    a = layer1()
    b = layer2()
    return 0 if (a and b is not False) else 1


if __name__ == "__main__":
    sys.exit(main())
