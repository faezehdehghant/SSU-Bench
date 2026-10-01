#!/usr/bin/env python3
"""Check that the SSU-Bench files and metadata are complete and consistent.

Run from the repository root:  python scripts/validate_dataset.py
Requires Pillow and NumPy. Exits with status 1 if any check fails.
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)


def validate_text():
    d = os.path.join(ROOT, "text_counterfactuals")
    meta = json.load(open(os.path.join(d, "prompts.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(d, "pairs.csv"), encoding="utf-8")))
    check(len(meta) == 57, f"text prompts.json has {len(meta)} entries, expected 57")
    check(len(rows) == len(meta), "text pairs.csv and prompts.json differ in length")
    ids = [int(e["pair_id"]) for e in meta]
    check(ids == sorted(set(ids)), "text pair ids are not unique and sorted")
    formats = {}
    for e, r in zip(meta, rows):
        pid = int(e["pair_id"])
        check(int(r["pair_id"]) == pid, f"text pair {pid}: csv/json order mismatch")
        path = os.path.join(d, "images", e["image_name"])
        check(os.path.isfile(path), f"text pair {pid}: image missing ({e['image_name']})")
        if os.path.isfile(path):
            try:
                with Image.open(path) as im:
                    im.load()
                    formats[im.format] = formats.get(im.format, 0) + 1
            except Exception as err:  # noqa: BLE001
                check(False, f"text pair {pid}: image cannot be opened ({err})")
        s, u, w = e["safe_prompt"], e["unsafe_prompt"], e["marked_word"]
        check(s != u, f"text pair {pid}: prompts identical")
        check(w and w in u.split() or w in u, f"text pair {pid}: critical word '{w}' not in unsafe prompt")
        check(r["safe_prompt"] == s and r["unsafe_prompt"] == u and r["critical_word"] == w,
              f"text pair {pid}: pairs.csv disagrees with prompts.json")
        ds, du = s.split(), u.split()
        n_diff = sum(1 for a, b in zip(ds, du) if a != b) + abs(len(ds) - len(du))
        if n_diff != 1:
            print(f"note: text pair {pid} differs in more than one word ({s!r} vs {u!r})")
    extra = sorted(set(os.listdir(os.path.join(d, "images"))) - {e["image_name"] for e in meta})
    check(not extra, f"text images without metadata: {extra}")
    print(f"text subset: {len(meta)} pairs, image formats {formats}")


def validate_image():
    d = os.path.join(ROOT, "image_counterfactuals")
    meta = json.load(open(os.path.join(d, "prompts.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(d, "pairs.csv"), encoding="utf-8")))
    check(len(meta) == 60, f"image prompts.json has {len(meta)} entries, expected 60")
    check(len(rows) == len(meta), "image pairs.csv and prompts.json differ in length")
    sizes = {}
    coverage = []
    for e, r in zip(meta, rows):
        n = int(e["image_number"])
        check(int(r["pair_id"]) == n, f"image pair {n}: csv/json order mismatch")
        check(r["prompt"] == e["prompt"] and r["critical_word"] == e["word"],
              f"image pair {n}: pairs.csv disagrees with prompts.json")
        check(e["word"] in e["prompt"], f"image pair {n}: critical word '{e['word']}' not in prompt")
        paths = {k: os.path.join(d, r[c]) for k, c in (("safe", "safe_image"), ("unsafe", "unsafe_image"), ("mask", "mask"))}
        ims = {}
        for k, p in paths.items():
            check(os.path.isfile(p), f"image pair {n}: {k} file missing ({os.path.relpath(p, d)})")
            if os.path.isfile(p):
                try:
                    im = Image.open(p)
                    im.load()
                    ims[k] = im
                except Exception as err:  # noqa: BLE001
                    check(False, f"image pair {n}: {k} cannot be opened ({err})")
        if len(ims) == 3:
            check(ims["safe"].size == ims["unsafe"].size == ims["mask"].size,
                  f"image pair {n}: sizes differ {[im.size for im in ims.values()]}")
            m = np.asarray(ims["mask"].convert("L"))
            vals = set(np.unique(m).tolist())
            check(vals <= {0, 255}, f"image pair {n}: mask is not binary 0/255 ({sorted(vals)[:6]})")
            check(255 in vals, f"image pair {n}: mask is empty")
            coverage.append(float((m == 255).mean()))
            sizes[ims["safe"].size] = sizes.get(ims["safe"].size, 0) + 1
    n_img = len([f for f in os.listdir(os.path.join(d, "images")) if f.endswith(".png")])
    n_mask = len([f for f in os.listdir(os.path.join(d, "masks")) if f.endswith(".png")])
    check(n_img == 120 and n_mask == 60, f"expected 120 images and 60 masks, found {n_img} and {n_mask}")
    if coverage:
        print(f"image subset: {len(meta)} pairs, {len(sizes)} distinct sizes (most common {max(sizes, key=sizes.get)}), "
              f"mask coverage {100*min(coverage):.1f}-{100*max(coverage):.1f}% (median {100*float(np.median(coverage)):.1f}%)")


def validate_eligibility():
    p = os.path.join(ROOT, "eligibility", "eligible_pairs.json")
    e = json.load(open(p, encoding="utf-8"))
    text_ids = {int(x["pair_id"]) for x in json.load(open(os.path.join(ROOT, "text_counterfactuals", "prompts.json"), encoding="utf-8"))}
    image_ids = {int(x["image_number"]) for x in json.load(open(os.path.join(ROOT, "image_counterfactuals", "prompts.json"), encoding="utf-8"))}
    for model, ids in e["text_counterfactuals"]["eligible"].items():
        check(set(ids) <= text_ids, f"eligibility: text ids for {model} not in dataset")
        bad = set(ids) & set(e["text_counterfactuals"]["tokenizer_check_failed"][model])
        check(not bad, f"eligibility: {model} text pairs both eligible and tokenizer-failed: {sorted(bad)}")
    for model, ids in e["image_counterfactuals"]["eligible"].items():
        check(set(ids) <= image_ids, f"eligibility: image ids for {model} not in dataset")
    counts = {m: len(v) for m, v in e["text_counterfactuals"]["eligible"].items()}
    counts_i = {m: len(v) for m, v in e["image_counterfactuals"]["eligible"].items()}
    print(f"eligible pairs: text {counts}, image {counts_i}")


if __name__ == "__main__":
    validate_text()
    validate_image()
    validate_eligibility()
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("all checks passed")
