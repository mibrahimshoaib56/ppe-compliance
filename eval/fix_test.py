"""One-shot script to update test_compliance.py and format_worker_table in utils.py."""
import pathlib

# ── 1. Fix tests/test_compliance.py ──────────────────────────────────────────
test_path = pathlib.Path("tests/test_compliance.py")
text = test_path.read_text(encoding="utf-8")

old = (
    '    assert w["status"] == "Unknown"\n'
    '\n'
    '    assert summary["total_workers"] == 1\n'
    '    assert summary["compliant_count"] == 0\n'
    '    assert summary["violation_count"] == 0\n'
    '    assert summary["unknown_count"] == 1\n'
)
new = (
    '    # vest unknown with no hardhat missing -> "Vest not visible: manual check"\n'
    '    assert w["status"] == "Vest not visible: manual check"\n'
    '\n'
    '    assert summary["total_workers"] == 1\n'
    '    assert summary["compliant_count"] == 0\n'
    '    assert summary["violation_count"] == 0\n'
    '    assert summary["needs_review_count"] == 1\n'
    '    assert summary["unknown_count"] == 1  # unknown_count mirrors needs_review_count\n'
)

if old in text:
    text = text.replace(old, new, 1)
    test_path.write_text(text, encoding="utf-8")
    print("✅ tests/test_compliance.py updated.")
else:
    print("❌ Pattern not found in test_compliance.py — check manually.")
    # Print what line 145 looks like
    lines = text.splitlines()
    for i, l in enumerate(lines[140:155], start=141):
        print(f"  {i}: {repr(l)}")

# ── 2. Fix src/utils.py format_worker_table ──────────────────────────────────
utils_path = pathlib.Path("src/utils.py")
utext = utils_path.read_text(encoding="utf-8")

old_u = (
    '        records.append({\n'
    '            "Worker ID": f"Worker #{w[\'worker_id\']}\",\n'
    '            "Overall Status": w["status"],\n'
    '            "Hardhat": w["hardhat_status"].capitalize(),\n'
    '            "Safety Vest": w["vest_status"].capitalize(),\n'
    '            "Confidence": f"{int(w.get(\'confidence\', 1.0) * 100)}%",\n'
    '        })\n'
)
new_u = (
    '        vest_raw = w["vest_status"]\n'
    '        vest_display = (\n'
    '            "Not visible — check manually"\n'
    '            if vest_raw == "unknown"\n'
    '            else vest_raw.capitalize()\n'
    '        )\n'
    '        records.append({\n'
    '            "Worker ID": f"Worker #{w[\'worker_id\']}\",\n'
    '            "Overall Status": w["status"],\n'
    '            "Hardhat": w["hardhat_status"].capitalize(),\n'
    '            "Safety Vest": vest_display,\n'
    '            "Confidence": f"{int(w.get(\'confidence\', 1.0) * 100)}%",\n'
    '        })\n'
)

if old_u in utext:
    utext = utext.replace(old_u, new_u, 1)
    utils_path.write_text(utext, encoding="utf-8")
    print("✅ src/utils.py format_worker_table updated.")
else:
    print("❌ Pattern not found in utils.py — checking current state...")
    idx = utext.find("format_worker_table")
    snippet = utext[idx:idx+600]
    print(repr(snippet))
