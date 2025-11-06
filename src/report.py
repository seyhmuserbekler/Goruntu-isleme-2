# src/report.py
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import datetime

# ---- Font Kaydı (Türkçe karakterler için) ----
HERE = os.path.dirname(__file__)
FONT_DIR = os.path.normpath(os.path.join(HERE, "..", "fonts"))
DEJAVU = os.path.join(FONT_DIR, "DejaVuSans.ttf")
ARIAL = r"C:\Windows\Fonts\arial.ttf"  # Windows yedeği

FONT_NAME = None
if os.path.exists(DEJAVU):
    pdfmetrics.registerFont(TTFont("DejaVu", DEJAVU))
    FONT_NAME = "DejaVu"
elif os.path.exists(ARIAL):
    pdfmetrics.registerFont(TTFont("ArialTR", ARIAL))
    FONT_NAME = "ArialTR"
else:
    # Son çare: varsayılan font (Türkçe eksik çıkabilir)
    FONT_NAME = "Helvetica"

def tr_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(name="TitleTR", parent=styles["Title"], fontName=FONT_NAME),
        "h2": ParagraphStyle(name="H2TR", parent=styles["Heading2"], fontName=FONT_NAME),
        "h3": ParagraphStyle(name="H3TR", parent=styles["Heading3"], fontName=FONT_NAME),
        "body": ParagraphStyle(name="BodyTR", parent=styles["BodyText"], fontName=FONT_NAME, leading=14),
    }

def build_pdf(report_path, project_title, author, per_image_sections, references):
    # Hedef klasör yoksa oluştur
    os.makedirs(os.path.dirname(os.path.normpath(report_path)), exist_ok=True)

    doc = SimpleDocTemplate(
        os.path.normpath(report_path),
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    s = tr_styles()

    elements = []

    # Kapak
    elements.append(Paragraph(project_title, s["title"]))
    elements.append(Spacer(1, 0.5*cm))
    elements.append(Paragraph(f"Yazar: {author}", s["body"]))
    elements.append(Paragraph(f"Tarih: {datetime.date.today().strftime('%d.%m.%Y')}", s["body"]))
    elements.append(Spacer(1, 1*cm))

    # Özet
    elements.append(Paragraph("Özet", s["h2"]))
    elements.append(Paragraph(
        "Bu raporda üç görüntü üzerinde temel ve ileri düzey görüntü işleme adımları uygulanmıştır: "
        "boyutlandırma, normalizasyon, gri ton dönüşümü, HSV dönüşümü, Gauss bulanıklaştırma, "
        "kenar tespiti, histogram eşitleme, eşikleme ve geometrik dönüşümler, morfolojik işlemler "
        "(erozyon, dilatasyon, açma, kapama) ve bölütleme (K-Means, Watershed). "
        "Her adımın amacı kısa açıklamalarla verilmiş ve örnek çıktılar raporlanmıştır.", s["body"]))
    elements.append(Spacer(1, 0.5*cm))

    # Görsel bazlı bölümler
    for sec in per_image_sections:
        elements.append(PageBreak())
        elements.append(Paragraph(f"Görüntü: {os.path.basename(sec['image_name'])}", s["h2"]))
        elements.append(Spacer(1, 0.2*cm))

        for step_title, desc, img_paths in sec["steps"]:
            elements.append(Paragraph(step_title, s["h3"]))
            elements.append(Paragraph(desc, s["body"]))
            elements.append(Spacer(1, 0.2*cm))

            # Görselleri 2'şer sırala
            row = []
            for i, p in enumerate(img_paths):
                if os.path.exists(p):
                    row.append(Image(os.path.normpath(p), width=7*cm, height=7*cm))
                if len(row) == 2 or i == len(img_paths) - 1:
                    for im in row:
                        elements.append(im)
                    elements.append(Spacer(1, 0.3*cm))
                    row = []
            elements.append(Spacer(1, 0.2*cm))

    # Literatür
    elements.append(PageBreak())
    elements.append(Paragraph("Literatür", s["h2"]))
    for ref in references:
        elements.append(Paragraph(ref, s["body"]))
        elements.append(Spacer(1, 0.1*cm))

    doc.build(elements)
