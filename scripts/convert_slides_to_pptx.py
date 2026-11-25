"""Convert SLIDES.md to SLIDES.pptx using python-pptx without altering text."""

import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def markdown_slides_to_pptx(md_path: str, pptx_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split slides by horizontal rule ---
    # Strip YAML frontmatter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2]

    raw_slides = [s.strip() for s in content.split("\n---") if s.strip()]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    for raw_slide in raw_slides:
        slide = prs.slides.add_slide(blank_layout)

        # Extract speaker notes
        notes = []
        body_lines = []
        for line in raw_slide.split("\n"):
            if line.strip().startswith("> **Speaker Note:**") or line.strip().startswith(">"):
                note_text = line.strip().lstrip(">").replace("**Speaker Note:**", "").strip()
                if note_text:
                    notes.append(note_text)
            else:
                body_lines.append(line)

        if notes:
            slide.notes_slide.notes_text_frame.text = "\n".join(notes)

        # Extract title and body
        title = ""
        body_content = []
        found_title = False
        for line in body_lines:
            if not found_title and (line.startswith("# ") or line.startswith("## ")):
                title = re.sub(r"^#+\s*", "", line).strip()
                found_title = True
            else:
                body_content.append(line)

        # Add Title Box
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.6), Inches(11.7), Inches(1.0))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title if title else "PPE Compliance Detection"
        p_title.font.size = Pt(28)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(15, 23, 42)

        # Add Body Box
        body_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
        tf_body = body_box.text_frame
        tf_body.word_wrap = True

        first = True
        clean_body = "\n".join(body_content).strip()
        for bline in clean_body.split("\n"):
            if not bline.strip():
                continue
            p = tf_body.paragraphs[0] if first else tf_body.add_paragraph()
            first = False
            clean_text = bline.replace("**", "").replace("*", "•") if bline.strip().startswith("* ") else bline.replace("**", "")
            p.text = clean_text
            p.font.size = Pt(15)
            p.font.color.rgb = RGBColor(51, 65, 85)

    prs.save(pptx_path)
    print(f"Saved {len(raw_slides)} slides to {pptx_path}")


if __name__ == "__main__":
    markdown_slides_to_pptx("SLIDES.md", "SLIDES.pptx")
