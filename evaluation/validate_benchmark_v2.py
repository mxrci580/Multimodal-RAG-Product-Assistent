"""
Benchmark Validation Script: questions_v2_50.json
Audits catalog validity, constraint compliance, image availability,
evidence distribution, and ground-truth consistency.
"""

import csv
import json
import os
import re
from PIL import Image

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "amazon.csv")
BENCHMARK_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset", "questions_v2_50.json")
IMAGE_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "image_cache")


def parse_price(p_str):
    if not p_str:
        return 0.0
    val = re.sub(r"[^\d.]", "", str(p_str).replace(",", ""))
    return float(val) if val else 0.0


def validate_benchmark():
    print("=" * 80)
    print("PHASE 4A BENCHMARK INTEGRITY AUDIT: questions_v2_50.json")
    print("=" * 80)

    # 1. Load catalog
    assert os.path.exists(CATALOG_PATH), f"Catalog not found at {CATALOG_PATH}"
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = {r["product_id"]: r for r in csv.DictReader(f)}
    print(f"[✓] Catalog loaded: {len(catalog)} products")

    # 2. Load questions
    assert os.path.exists(BENCHMARK_PATH), f"Benchmark file not found at {BENCHMARK_PATH}"
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"[✓] Benchmark loaded: {len(questions)} queries")
    assert len(questions) == 50, f"Expected exactly 50 queries, found {len(questions)}"

    # 3. Audit Distributions
    ev_counts = {}
    diff_counts = {}
    intent_counts = {}

    for q in questions:
        ev_counts[q["evidence_type"]] = ev_counts.get(q["evidence_type"], 0) + 1
        diff_counts[q["difficulty"]] = diff_counts.get(q["difficulty"], 0) + 1
        intent_counts[q["intent_type"]] = intent_counts.get(q["intent_type"], 0) + 1

    print("\n--- Evidence Distribution Check ---")
    for ev in ["IMAGE", "TEXT+IMAGE", "TEXT"]:
        count = ev_counts.get(ev, 0)
        expected = {"IMAGE": 15, "TEXT+IMAGE": 25, "TEXT": 10}[ev]
        status = "✓ PASS" if count == expected else "✗ FAIL"
        print(f"  {ev:<12}: {count:>2} / {expected} [{status}]")
        assert count == expected, f"Evidence mismatch for {ev}"

    print("\n--- Difficulty Distribution Check ---")
    for diff in ["easy", "medium", "hard", "adversarial"]:
        count = diff_counts.get(diff, 0)
        expected = {"easy": 15, "medium": 15, "hard": 12, "adversarial": 8}[diff]
        status = "✓ PASS" if count == expected else "✗ FAIL"
        print(f"  {diff:<12}: {count:>2} / {expected} [{status}]")
        assert count == expected, f"Difficulty mismatch for {diff}"

    # 4. Catalog & Constraint Audit
    print("\n--- Catalog & Constraint Verification ---")
    referenced_pids = set()
    constraint_violations = []
    missing_pids = []
    image_issues = []

    for q in questions:
        qid = q["id"]
        constraints = q.get("constraints", {})
        price_max = constraints.get("price_max")

        for rp in q["relevant_products"]:
            pid = rp["product_id"]
            grade = rp["relevance"]
            referenced_pids.add(pid)

            if pid not in catalog:
                missing_pids.append((qid, pid))
                continue

            prod = catalog[pid]
            prod_price = parse_price(prod["discounted_price"])

            # Verify price constraint for Grade 3 (ideal) products
            if grade == 3 and price_max is not None:
                if prod_price > price_max:
                    constraint_violations.append((qid, pid, prod_price, price_max))

            # Verify image cache
            img_path = os.path.join(IMAGE_CACHE_DIR, f"{pid}.jpg")
            if not os.path.exists(img_path) or os.path.getsize(img_path) == 0:
                image_issues.append((pid, "missing on disk"))
            else:
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    image_issues.append((pid, f"corrupt image: {e}"))

    print(f"  Total Unique Product IDs Referenced: {len(referenced_pids)}")
    print(f"  Missing Catalog Products: {len(missing_pids)}")
    if missing_pids:
        for qid, pid in missing_pids:
            print(f"    [FAIL] {qid} references unknown PID {pid}")
    assert len(missing_pids) == 0, "Missing product IDs detected!"

    print(f"  Price Constraint Violations (Grade 3): {len(constraint_violations)}")
    if constraint_violations:
        for qid, pid, p_actual, p_max in constraint_violations:
            print(f"    [FAIL] {qid} ({pid}): Price ₹{p_actual} exceeds max ₹{p_max}")
    assert len(constraint_violations) == 0, "Price constraint violations detected!"

    print(f"  Image Verification (all 79 products): {len(image_issues)} issues")
    if image_issues:
        for pid, err in image_issues:
            print(f"    [FAIL] {pid}: {err}")
    assert len(image_issues) == 0, "Image verification issues detected!"

    # 5. Schema Integrity
    required_keys = {"id", "question", "evidence_type", "difficulty", "intent_type", "category", "constraints", "relevant_products", "required_evidence"}
    for q in questions:
        missing_keys = required_keys - set(q.keys())
        assert not missing_keys, f"Question {q['id']} missing keys: {missing_keys}"
        assert len(q["relevant_products"]) >= 1, f"Question {q['id']} has no relevant products"
        has_grade_3 = any(rp["relevance"] == 3 for rp in q["relevant_products"])
        assert has_grade_3, f"Question {q['id']} has no Grade 3 (ideal) product"

    print("\n" + "=" * 80)
    print("ALL AUDITS PASSED: Benchmark is 100% verified and ready to be frozen!")
    print("=" * 80)


if __name__ == "__main__":
    validate_benchmark()
