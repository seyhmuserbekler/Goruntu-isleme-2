# src/main.py
import os
import glob
import cv2
from utils import (
    resize_image, normalize_image, to_grayscale, to_hsv, gaussian_blur,
    canny_edges, histogram_equalization, threshold_otsu, geometric_ops,
    morphological_ops, kmeans_segmentation, watershed_segmentation,
    imread_unicode, imwrite_unicode
)
from report import build_pdf

# ============== CONFIG ==============
CONFIG = {
    "input_dir": os.path.join(os.path.dirname(__file__), "..", "images"),
    "output_dir": os.path.join(os.path.dirname(__file__), "..", "outputs"),
    "report_path": os.path.join(os.path.dirname(__file__), "..", "reports", "odev_raporu.pdf"),
    "target_size": (512, 512),
    "gaussian_ksize": (5, 5),
    "gaussian_sigma": 0,
    "canny_low": 100,
    "canny_high": 200,
    "rotate_deg": 15,
    "scale": 1.0,
    "flip_code": 1,   # 1: yatay, 0: dikey, -1: ikisi
    "kmeans_K": 3,
    "author": "Şeyhmus Erbekler",
    "project_title": "Görüntü İşleme Teknikleri Ödevi"
}
# ====================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_step(img, path):
    ensure_dir(os.path.dirname(path))
    ok = imwrite_unicode(path, img)
    if not ok:
        print(f"[UYARI] Kaydedilemedi: {path}")

def process_image(img_path, cfg):
    """
    Görseli işler ve PDF raporu için (başlık, açıklama, görsel yol listeleri) içeren
    bir bölüm (section) objesi döndürür.
    """
    img_name = os.path.basename(img_path)
    base_out = os.path.join(cfg["output_dir"], os.path.splitext(img_name)[0])
    ensure_dir(base_out)
    print(f"[INFO] İşleniyor: {img_name}")

    # Unicode güvenli okuma
    bgr = imread_unicode(img_path)
    if bgr is None:
        print(f"[UYARI] Okunamadı: {img_path}")
        return None

    steps_desc = []

    # 1) Boyutlandırma
    resized = resize_image(bgr, cfg["target_size"])
    p1 = os.path.join(base_out, "01_resized.jpg")
    save_step(resized, p1)
    steps_desc.append((
        "Boyutlandırma (Resize)",
        "Görüntü, işlem sürekliliği ve karşılaştırılabilirlik için sabit (512×512) boyuta ölçeklendi.",
        [p1]
    ))

    # 2) Normalizasyon
    normalized = normalize_image(resized)
    p2 = os.path.join(base_out, "02_normalized.jpg")
    save_step(normalized, p2)
    steps_desc.append((
        "Normalizasyon (Min–Max)",
        "Piksel değerleri min–max yöntemi ile [0,255] aralığına yeniden ölçeklendi; kontrast dağılımı dengelendi.",
        [p2]
    ))

    # 3) Grayscale
    gray = to_grayscale(resized)
    p3 = os.path.join(base_out, "03_gray.jpg")
    save_step(gray, p3)
    steps_desc.append((
        "Gri Ton Dönüşümü (Grayscale)",
        "Renk kanallarından tek kanala indirgenerek yoğunluk bilgisi odaklı işleme uygun hale getirildi.",
        [p3]
    ))

    # 4) HSV (görüntüleme için tekrar BGR'e çevirip kaydediyoruz)
    hsv = to_hsv(resized)
    hsv_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    p4 = os.path.join(base_out, "04_hsv.jpg")
    save_step(hsv_vis, p4)
    steps_desc.append((
        "HSV Dönüşümü",
        "BGR uzayından HSV'ye dönüştürülerek ton/doygunluk/parlaklık bileşenleri ayrıştırıldı (görselleştirme için BGR'e çevrilerek kaydedildi).",
        [p4]
    ))

    # 5) Gaussian Blur
    blurred = gaussian_blur(gray, cfg["gaussian_ksize"], cfg["gaussian_sigma"])
    p5 = os.path.join(base_out, "05_gaussian_blur.jpg")
    save_step(blurred, p5)
    steps_desc.append((
        "Gaussian Blur",
        "Gürültüyü azaltmak ve kenar tespiti öncesinde pürüzleri yumuşatmak için Gauss filtresi uygulandı.",
        [p5]
    ))

    # 6) Kenar Tespiti (Canny)
    edges = canny_edges(blurred, cfg["canny_low"], cfg["canny_high"])
    p6 = os.path.join(base_out, "06_canny_edges.jpg")
    save_step(edges, p6)
    steps_desc.append((
        "Kenar Tespiti (Canny)",
        "Yoğunluk gradyanlarını kullanarak belirgin kenarlar tespit edildi.",
        [p6]
    ))

    # 7) Histogram Eşitleme
    hist_eq = histogram_equalization(gray)
    p7 = os.path.join(base_out, "07_hist_eq.jpg")
    save_step(hist_eq, p7)
    steps_desc.append((
        "Histogram Eşitleme",
        "Küresel histogram eşitleme ile kontrast iyileştirildi.",
        [p7]
    ))

    # 8) Eşikleme + Geometrik işlemler
    th = threshold_otsu(gray)
    geo = geometric_ops(th, cfg["rotate_deg"], cfg["scale"], cfg["flip_code"])
    p8 = os.path.join(base_out, "08_threshold_otsu.jpg")
    p9 = os.path.join(base_out, "09_geo_ops.jpg")
    save_step(th, p8)
    save_step(geo, p9)
    steps_desc.append((
        "Eşikleme (Otsu) ve Geometrik İşlemler",
        "Otsu ile otomatik eşikleme sonrası görüntü 15° döndürüldü ve yatay çevrildi.",
        [p8, p9]
    ))

    # 9) Morfolojik işlemler
    erosion, dilation, opening, closing = morphological_ops(th, 3)
    p10 = os.path.join(base_out, "10_morph_erosion.jpg")
    p11 = os.path.join(base_out, "11_morph_dilation.jpg")
    p12 = os.path.join(base_out, "12_morph_opening.jpg")
    p13 = os.path.join(base_out, "13_morph_closing.jpg")
    save_step(erosion, p10)
    save_step(dilation, p11)
    save_step(opening, p12)
    save_step(closing, p13)
    steps_desc.append((
        "Morfolojik İşlemler (açma, kapama, erozyon, dilatasyon)",
        "İkili görüntü üzerinde temel morfolojik işlemler ile gürültü temizleme ve boşluk doldurma yapıldı.",
        [p10, p11, p12, p13]
    ))

    # 10) Bölütleme
    kseg, _ = kmeans_segmentation(resized, K=cfg["kmeans_K"])
    wres, _ = watershed_segmentation(resized)
    p14 = os.path.join(base_out, "14_seg_kmeans.jpg")
    p15 = os.path.join(base_out, "15_seg_watershed.jpg")
    save_step(kseg, p14)
    save_step(wres, p15)
    steps_desc.append((
        "Bölütleme (Segmentation)",
        "K-Means (K=3) ile renk tabanlı bölütleme ve Watershed ile temas halindeki nesnelerin ayrıştırılması gerçekleştirildi.",
        [p14, p15]
    ))

    return {
        "image_name": img_path,
        "steps": steps_desc
    }

def main():
    input_dir = os.path.normpath(CONFIG["input_dir"])
    candidates = [os.path.join(input_dir, f"{i}.jpg") for i in (1, 2, 3)]
    imgs = [p for p in candidates if os.path.exists(p)]

    if not imgs:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(input_dir, ext)))
        imgs = sorted(files)[:3]

    if len(imgs) == 0:
        print("[UYARI] images/ klasöründe işlenecek görsel bulunamadı (1.jpg/2.jpg/3.jpg bekleniyordu).")
        return

    sections = []
    for p in imgs:
        sec = process_image(p, CONFIG)
        if sec:
            sections.append(sec)

    print(f"[OK] Tüm çıktılar: {os.path.normpath(CONFIG['output_dir'])}")

    # Literatür
    references = [
        "Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.",
        "OpenCV Documentation: https://docs.opencv.org/",
        "Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics.",
        "Vincent, L., & Soille, P. (1991). Watersheds in digital spaces: An efficient algorithm based on immersion simulations. IEEE PAMI."
    ]

    # PDF üret
    build_pdf(
        CONFIG["report_path"],
        CONFIG["project_title"],
        CONFIG["author"],
        sections,
        references
    )
    print(f"[OK] PDF oluşturuldu: {os.path.normpath(CONFIG['report_path'])}")

if __name__ == "__main__":
    main()
