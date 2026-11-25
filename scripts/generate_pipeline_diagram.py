"""Generate a high-resolution pipeline architecture diagram for the PPE Compliance System.

Produces both PNG (300 DPI) and SVG formats in eval/results/ for embedding in
the AI2002 final report and presentation slides.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_pipeline_diagram(output_png: str, output_svg: str):
    fig = plt.figure(figsize=(18, 10.5), facecolor="#F8FAFC")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#F8FAFC")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    # Title Banner
    ax.text(
        9.0, 9.9,
        "HARDHAT & SAFETY VEST COMPLIANCE DETECTION PIPELINE",
        fontsize=18, fontweight="bold", ha="center", va="center", color="#0F172A", family="sans-serif"
    )
    ax.text(
        9.0, 9.55,
        "End-to-End System Architecture: Deep Learning Object Detection + Rule-Based Spatial Compliance & Reliability Engine",
        fontsize=11.5, fontstyle="italic", ha="center", va="center", color="#475569", family="sans-serif"
    )

    def draw_card(x, y, w, h, bg_color="#FFFFFF", border_color="#94A3B8", radius=0.15, lw=1.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad={radius}",
            facecolor=bg_color,
            edgecolor=border_color,
            linewidth=lw,
            zorder=2
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color="#64748B", style="->", lw=2.0, zorder=3):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                color=color,
                lw=lw,
                shrinkA=3,
                shrinkB=3,
                mutation_scale=16,
            ),
            zorder=zorder
        )

    # =========================================================================
    # ROW 1 (Top Flow): Input -> YOLOv8 Detection -> 2-Stage Filter -> Anatomical Zones
    # =========================================================================

    # 1. INPUT STAGE
    draw_card(0.6, 6.2, 3.2, 2.6, bg_color="#F0F9FF", border_color="#0284C7", lw=2.0)
    ax.text(2.2, 8.45, "1. INPUT SOURCE", fontsize=11, fontweight="bold", color="#0369A1", ha="center")
    ax.text(2.2, 7.95, "• Site Image (JPEG / PNG)\n• CCTV Video Stream / MP4\n• YouTube Surveillance Clip", fontsize=9.5, color="#334155", ha="center", va="center")
    ax.text(2.2, 6.75, "Preprocessing:\n• Proportional resize ≤ 416px / 640px\n• RGB normalization", fontsize=8.8, color="#0284C7", ha="center", va="center")

    # Arrow 1 -> 2
    draw_arrow(3.9, 7.5, 4.7, 7.5)

    # 2. YOLOv8 DETECTOR STAGE
    draw_card(4.8, 6.2, 3.7, 2.6, bg_color="#EEF2FF", border_color="#4F46E5", lw=2.0)
    ax.text(6.65, 8.45, "2. YOLOv8n INFERENCE", fontsize=11, fontweight="bold", color="#4338CA", ha="center")
    ax.text(6.65, 7.95, "Ultralytics YOLOv8 Nano\n(129 Layers, 3.01M Params)", fontsize=9.5, color="#1E1B4B", ha="center", va="center", fontweight="bold")
    ax.text(6.65, 7.05, "• Base Conf = 0.10 (prevents NMS drop)\n• 10 Trained Classes:\n  Person, Hardhat, NO-Hardhat,\n  Safety Vest, NO-Safety Vest, Context", fontsize=8.8, color="#3730A3", ha="center", va="center")

    # Arrow 2 -> 3
    draw_arrow(8.6, 7.5, 9.4, 7.5)

    # 3. TWO-STAGE PER-CLASS FILTER
    draw_card(9.5, 6.2, 3.7, 2.6, bg_color="#FAF5FF", border_color="#7C3AED", lw=2.0)
    ax.text(11.35, 8.45, "3. CLASS CONFIDENCE GATES", fontsize=11, fontweight="bold", color="#6D28D9", ha="center")
    ax.text(11.35, 7.85, "Independent Class Thresholds:", fontsize=9.2, color="#581C87", ha="center", fontweight="bold")
    ax.text(11.35, 6.95, "• CONF_PERSON = 0.30 (suppress clutter)\n• CONF_HARDHAT = 0.30 (positive gear certainty)\n• CONF_NO_HARDHAT = 0.20 (favor recall)\n• CONF_SAFETY_VEST = 0.20 (occluded recall)\n• CONF_NO_SAFETY_VEST = 0.20 (violation recall)", fontsize=8.2, color="#4C1D95", ha="center", va="center")

    # Arrow 3 -> 4
    draw_arrow(13.3, 7.5, 14.1, 7.5)

    # 4. ANATOMICAL ZONE DECOMPOSITION
    draw_card(14.2, 6.2, 3.2, 2.6, bg_color="#FDF4FF", border_color="#C026D3", lw=2.0)
    ax.text(15.8, 8.45, "4. ANATOMICAL ZONES", fontsize=11, fontweight="bold", color="#A21CAF", ha="center")
    ax.text(15.8, 7.85, "Worker Coordinate Split:\n(Per Detected Person Box)", fontsize=9.2, color="#701A75", ha="center")
    ax.text(15.8, 6.9, "• Head Zone (top 0% - 35%):\n  For Hardhat / NO-Hardhat\n\n• Torso Zone (widened 15% - 75%):\n  For Vest / NO-Safety Vest\n  (Accommodates tool bags & belts)", fontsize=8.3, color="#86198F", ha="center", va="center")

    # Arrow from Row 1 down to Row 2: from Zone Decomposition down to IoG Disambiguation
    draw_arrow(15.8, 6.0, 15.8, 5.1, color="#64748B")

    # =========================================================================
    # ROW 2 (Middle/Bottom Flow): IoG Matching -> Reliability Engine -> Decision Matrix -> Outputs
    # =========================================================================

    # 5. SPATIAL DISAMBIGUATION & IoG MATCHING
    draw_card(13.5, 2.3, 3.9, 2.6, bg_color="#ECFEFF", border_color="#0891B2", lw=2.0)
    ax.text(15.45, 4.55, "5. GEAR DISAMBIGUATION", fontsize=11, fontweight="bold", color="#0E7490", ha="center")
    ax.text(15.45, 3.95, "Intersection over Gear Area (IoG):", fontsize=9.2, color="#155E75", ha="center", fontweight="bold")
    ax.text(15.45, 3.35, r"$\mathrm{IoG} = \frac{\mathrm{Area}(\mathrm{Gear} \cap \mathrm{Zone})}{\mathrm{Area}(\mathrm{Gear})} \geq 0.30$", fontsize=10.5, color="#0891B2", ha="center")
    ax.text(15.45, 2.7, "• Greedy Bipartite Association:\n  Gear linked to worker with max overlap\n• Prevents sharing gear between workers", fontsize=8.2, color="#164E63", ha="center", va="center")

    # Arrow 5 -> 6 (Right to Left on Row 2)
    draw_arrow(13.4, 3.6, 12.6, 3.6)

    # 6. RELIABILITY & SAFEGUARD ENGINE
    draw_card(8.7, 2.3, 3.8, 2.6, bg_color="#FFFBEB", border_color="#D97706", lw=2.0)
    ax.text(10.6, 4.55, "6. RELIABILITY SAFEGUARDS", fontsize=11, fontweight="bold", color="#B45309", ha="center")
    ax.text(10.6, 4.0, "Multi-Criteria Quality Filters:", fontsize=9.2, color="#92400E", ha="center", fontweight="bold")
    ax.text(10.6, 3.15, "• Aspect Ratio (h / w ≥ 1.2):\n  Excludes overhead / squashed views\n• Scale (height ≥ 12% frame):\n  Excludes tiny background figures\n• Confidence (Person conf ≥ 0.40):\n  Filters noisy/uncertain detections\n• Conflicting Detections Check:\n  Flags simultaneous positive/negative", fontsize=8.0, color="#78350F", ha="center", va="center")

    # Arrow 6 -> 7
    draw_arrow(8.6, 3.6, 7.8, 3.6)

    # 7. DECISION MATRIX & COMPLIANCE LOGIC
    draw_card(4.4, 2.3, 3.3, 2.6, bg_color="#F0FDF4", border_color="#16A34A", lw=2.0)
    ax.text(6.05, 4.55, "7. COMPLIANCE ENGINE", fontsize=11, fontweight="bold", color="#15803D", ha="center")
    ax.text(6.05, 3.95, "Worker Status Resolution:", fontsize=9.2, color="#166534", ha="center", fontweight="bold")
    ax.text(6.05, 3.1, "• Compliant:\n  Hardhat (Present) + Vest (Present)\n• Violation:\n  Missing hardhat / vest / both\n  (Strict policy if reliable)\n• Needs Review:\n  Small / Unreliable / Conflicting /\n  Overhead / Unconfirmed vest", fontsize=8.2, color="#14532D", ha="center", va="center")

    # Arrow 7 -> 8
    draw_arrow(4.3, 3.6, 3.5, 3.6)

    # 8. TEMPORAL SMOOTHING & OUTPUT STAGE
    draw_card(0.5, 2.3, 2.9, 2.6, bg_color="#F8FAFC", border_color="#475569", lw=2.0)
    ax.text(1.95, 4.55, "8. AGGREGATION & UI", fontsize=11, fontweight="bold", color="#334155", ha="center")
    ax.text(1.95, 3.95, "Temporal Smoothing (Video):\n7-frame median on counts & rate", fontsize=8.5, color="#1E293B", ha="center")
    ax.text(1.95, 3.1, "Site Safety Metrics:\n• Compliance Rate = Compliant / All\n• Safe (≥85%) | Warning | Critical\n\nReporting Artifacts:\n• Streamlit Interactive Dashboard\n• Risk Moments Gallery (Top 3)\n• CSV Inspection Log Export", fontsize=8.0, color="#334155", ha="center", va="center")

    # Bottom Footer Note
    ax.text(
        9.0, 0.7,
        "System Policy: Unreliable, distant, or squashed overhead workers are routed to 'Needs Review' rather than penalized unjustly. "
        "Strict Hardhat policy requires confirmed reliability.",
        fontsize=9.2, ha="center", va="center", color="#64748B", style="italic"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved PNG diagram to: {output_png}")
    print(f"Saved SVG diagram to: {output_svg}")


if __name__ == "__main__":
    out_png = os.path.join("eval", "results", "pipeline_architecture_diagram.png")
    out_svg = os.path.join("eval", "results", "pipeline_architecture_diagram.svg")
    create_pipeline_diagram(out_png, out_svg)
