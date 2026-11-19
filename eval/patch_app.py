"""Patch app.py to add Needs Review KPI card to Dashboard and Inspection Results."""
import pathlib, re

app_path = pathlib.Path("app.py")
text = app_path.read_text(encoding="utf-8")

# ── PATCH 1: Dashboard ───────────────────────────────────────────────────────
# Add needs_review_w variable after unknown_w line
old1a = '    unknown_w = summary.get("unknown_count", 0)\n'
new1a = (
    '    unknown_w = summary.get("unknown_count", 0)\n'
    '    review_w = summary.get("needs_review_count", unknown_w)\n'
)
if old1a in text:
    text = text.replace(old1a, new1a, 1)
    print("✅ Dashboard: added review_w variable")
else:
    print("❌ Could not add review_w")

# Change 5-column layout and add 6th card
old1b = (
    '    k1, k2, k3, k4, k5 = st.columns(5)\n'
    '    with k1:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Total Workers</div>"\n'
    '            f"<div class=\'metric-card-value\'>{total_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k2:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliant</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#22c55e;\'>{comp_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k3:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Violations</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#ef4444;\'>{viol_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k4:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliance Rate</div>"\n'
    '            f"<div class=\'metric-card-value\'>{rate}%</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k5:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Site Status</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:{status_color}; font-size:20px;\'>{status_badge}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
)
new1b = (
    '    k1, k2, k3, k4, k5, k6 = st.columns(6)\n'
    '    with k1:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Total Workers</div>"\n'
    '            f"<div class=\'metric-card-value\'>{total_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k2:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliant</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#22c55e;\'>{comp_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k3:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Violations</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#ef4444;\'>{viol_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k4:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Needs Review</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#f59e0b;\'>{review_w}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k5:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliance Rate</div>"\n'
    '            f"<div class=\'metric-card-value\'>{rate}%</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with k6:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Site Status</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:{status_color}; font-size:20px;\'>{status_badge}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
)
if old1b in text:
    text = text.replace(old1b, new1b, 1)
    print("✅ Dashboard KPI row: 5→6 columns with Needs Review card")
else:
    print("❌ Dashboard KPI columns pattern not found")

# ── PATCH 2: Dashboard bar chart — add "Needs Review" category ───────────────
old2 = (
    '        chart_data = pd.DataFrame({\n'
    '            "Status Category": ["Compliant Workers", "PPE Violations", "Inconclusive / Unknown"],\n'
    '            "Worker Count": [comp_w, viol_w, unknown_w],\n'
    '        }).set_index("Status Category")\n'
)
new2 = (
    '        chart_data = pd.DataFrame({\n'
    '            "Status Category": ["Compliant Workers", "PPE Violations", "Needs Review"],\n'
    '            "Worker Count": [comp_w, viol_w, review_w],\n'
    '        }).set_index("Status Category")\n'
)
if old2 in text:
    text = text.replace(old2, new2, 1)
    print("✅ Dashboard chart updated with Needs Review")
else:
    print("❌ Dashboard chart pattern not found")

# ── PATCH 3: Inspection Results KPI banner (5→6 columns) ────────────────────
old3 = (
    '    r1, r2, r3, r4, r5 = st.columns(5)\n'
    '    with r1:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Workers Detected</div>"\n'
    '            f"<div class=\'metric-card-value\'>{summary[\'total_workers\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r2:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliant</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#22c55e;\'>{summary[\'compliant_count\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r3:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Violations</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#ef4444;\'>{summary[\'violation_count\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r4:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliance Rate</div>"\n'
    '            f"<div class=\'metric-card-value\'>{summary[\'compliance_rate\']}%</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r5:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Site Status</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:{status_color}; font-size:20px;\'>{status_badge}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
)
new3 = (
    '    r1, r2, r3, r4, r5, r6 = st.columns(6)\n'
    '    with r1:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Workers Detected</div>"\n'
    '            f"<div class=\'metric-card-value\'>{summary[\'total_workers\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r2:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliant</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#22c55e;\'>{summary[\'compliant_count\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r3:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Violations</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#ef4444;\'>{summary[\'violation_count\']}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r4:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Needs Review</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:#f59e0b;\'>{summary.get(\'needs_review_count\', 0)}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r5:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Compliance Rate</div>"\n'
    '            f"<div class=\'metric-card-value\'>{summary[\'compliance_rate\']}%</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
    '    with r6:\n'
    '        st.markdown(\n'
    '            f"<div class=\'metric-card\'><div class=\'metric-card-title\'>Site Status</div>"\n'
    '            f"<div class=\'metric-card-value\' style=\'color:{status_color}; font-size:20px;\'>{status_badge}</div></div>",\n'
    '            unsafe_allow_html=True,\n'
    '        )\n'
)
if old3 in text:
    text = text.replace(old3, new3, 1)
    print("✅ Inspection Results KPI row: 5→6 columns with Needs Review card")
else:
    print("❌ Inspection Results KPI pattern not found")

app_path.write_text(text, encoding="utf-8")
print("\nDone — app.py written.")
