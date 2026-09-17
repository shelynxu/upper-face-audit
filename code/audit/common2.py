# -*- coding: utf-8 -*-
"""common2.py -- item list of the audit2 lane (pure stdlib; imported by every stage in every env).

Items = every wav of voicedose/ladder2/manifest.json (12 cells x 17 rungs = 204). Read-only.
items() -> list of dicts: id, cell, lang, label, rung, family, role, rate, wav (WSL path), verdict, realised, target, dose
"""
import io
import json
import os

LANE = "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/audit2"
L2 = "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/ladder2"
MANIFEST = L2 + "/manifest.json"
STRUCT = L2 + "/structure"
TIMEMAP = L2 + "/timemap"
HUMAN2 = "${PROJECT_ROOT}/pipeline/prosody_ladder/_explore/voicedose/human2"
RATE_RUNGS = ("rate090", "effort0", "effort")


def local(p):
    """Path usable by the current interpreter (WSL /mnt/d or Windows D:/)."""
    return p if os.path.exists(p) else p.replace("/mnt/d/", "D:/")


def manifest():
    return json.load(io.open(local(MANIFEST), encoding="utf-8"))


def items(cells=None):
    m = manifest()
    out = []
    for c in m["cells"]:
        if cells and c["cell"] not in cells:
            continue
        for it in c["items"]:
            out.append({"id": it["id"], "cell": c["cell"], "lang": c["lang"], "label": c["label"], "rung": it["rung"],
                        "family": it["family"], "role": it.get("role", ""), "rate": float(it.get("rate", 1.0) or 1.0),
                        "wav": it["path"], "verdict": it.get("verdict", {}).get("status", ""),
                        "realised": it.get("realised", {}), "target": it.get("target", {})})
    return out


def structure(cell):
    return json.load(io.open(local("%s/%s.json" % (STRUCT, cell)), encoding="utf-8"))


if __name__ == "__main__":
    from collections import Counter
    it = items()
    print(len(it), Counter(i["rung"] for i in it))
    print("missing:", [i["wav"] for i in it if not os.path.exists(local(i["wav"]))][:5])
