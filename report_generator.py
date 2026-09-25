"""Generate a downloadable PDF report for a cassava prediction."""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle,
)
from reportlab.lib.enums import TA_CENTER


def _pil_to_rl_image(pil_image, width_cm=7):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=width_cm * cm)


def _numpy_to_rl_image(np_array, width_cm=7):
    from PIL import Image as PILImage
    if np_array.dtype != "uint8":
        np_array = (np_array * 255).astype("uint8")
    pil = PILImage.fromarray(np_array)
    return _pil_to_rl_image(pil, width_cm)


def build_pdf_report(
    original_image,
    gradcam_image,
    predicted_class,
    confidence,
    all_probs,
    class_names,
    disease_info,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Cassava Leaf Disease Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=20, textColor=colors.HexColor("#1b5e20"),
        alignment=TA_CENTER, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=20,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=14, textColor=colors.HexColor("#2e7d32"),
        spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=4,
    )

    story = []
    story.append(Paragraph("Cassava Leaf Disease Report", title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        subtitle_style,
    ))

    # Prediction summary
    summary_data = [
        ["Prediction", predicted_class],
        ["Confidence", f"{confidence * 100:.1f}%"],
    ]
    summary_table = Table(summary_data, colWidths=[4 * cm, 12 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
        ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#f1f8e9")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1b5e20")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2e7d32")),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # Images
    story.append(Paragraph("Analysis Images", h2_style))
    img_table_data = [[
        Paragraph("<b>Original Image</b>", body_style),
        Paragraph("<b>Grad-CAM Heatmap</b>", body_style),
    ], [
        _pil_to_rl_image(original_image),
        _numpy_to_rl_image(gradcam_image) if gradcam_image is not None else Paragraph("Unavailable", body_style),
    ]]
    img_table = Table(img_table_data, colWidths=[8 * cm, 8 * cm])
    img_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(img_table)

    # Probabilities
    story.append(Paragraph("Class Probabilities", h2_style))
    prob_rows = [["Class", "Probability"]]
    for name, prob in zip(class_names, all_probs):
        prob_rows.append([name, f"{prob * 100:.1f}%"])
    prob_table = Table(prob_rows, colWidths=[12 * cm, 4 * cm])
    prob_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
    ]))
    story.append(prob_table)

    # Disease info
    if disease_info:
        story.append(Paragraph("Disease Description", h2_style))
        story.append(Paragraph(disease_info["description"], body_style))

        story.append(Paragraph("Symptoms", h2_style))
        for s in disease_info["symptoms"]:
            story.append(Paragraph(f"• {s}", body_style))

        story.append(Paragraph("Recommended Actions", h2_style))
        for a in disease_info["advice"]:
            story.append(Paragraph(f"• {a}", body_style))

    # Disclaimer
    story.append(Spacer(1, 20))
    disclaimer = ParagraphStyle(
        "Disclaimer", parent=body_style,
        fontSize=8, textColor=colors.grey,
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        "This report is a decision-support aid only. "
        "Always consult a local agricultural extension officer for confirmation.",
        disclaimer,
    ))

    doc.build(story)
    buf.seek(0)
    return buf
