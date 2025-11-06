import cv2
import numpy as np
import os

def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """
    Unicode (örn: Türkçe karakter içeren) Windows yollarında güvenli okuma.
    cv2.imread yerine np.fromfile + cv2.imdecode kullanır.
    """
    path = os.path.normpath(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None

def imwrite_unicode(path, img, params=None):
    """
    Unicode (örn: Türkçe karakter içeren) Windows yollarında güvenli yazma.
    cv2.imwrite yerine cv2.imencode + ndarray.tofile kullanır.
    """
    path = os.path.normpath(path)
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
        path = path + ext
    if params is None:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        buf.tofile(path)
        return True
    except Exception:
        return False

def resize_image(img, target_size=(512, 512)):
    return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    # Min-max normalization to [0, 255]
    img_float = img.astype(np.float32)
    min_val, max_val = img_float.min(), img_float.max()
    if max_val - min_val < 1e-6:
        norm = np.zeros_like(img_float)
    else:
        norm = (img_float - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)

def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def gaussian_blur(img, ksize=(5, 5), sigma=0):
    return cv2.GaussianBlur(img, ksize, sigmaX=sigma)

def canny_edges(gray, low=100, high=200):
    return cv2.Canny(gray, low, high)

def histogram_equalization(gray):
    return cv2.equalizeHist(gray)

def threshold_otsu(gray):
    # Otsu yöntemi ile otomatik eşikleme
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def geometric_ops(img, rotate_deg=15, scale=1.0, flip_code=1):
    """
    img ikili (binary) veya gri olabilir.
    flip_code: 1 yatay, 0 dikey, -1 her ikisi
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rotate_deg, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    flipped = cv2.flip(rotated, flip_code)
    return flipped

def morphological_ops(binary, ksize=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    erosion = cv2.erode(binary, kernel, iterations=1)
    dilation = cv2.dilate(binary, kernel, iterations=1)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return erosion, dilation, opening, closing

def kmeans_segmentation(img_bgr, K=3, attempts=10):
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img_bgr.shape)
    return segmented, labels.reshape(img_bgr.shape[:2])

def watershed_segmentation(img_bgr):
    gray = to_grayscale(img_bgr)
    blur = gaussian_blur(gray, (5, 5), 0)
    th = threshold_otsu(blur)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)
    result = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]  # sınırları kırmızı işaretle
    return result, markers
