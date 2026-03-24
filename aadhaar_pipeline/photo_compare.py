"""
Photo comparison between QR-embedded photo and card face crop.
Uses SSIM as primary metric with MSE as secondary.
"""
import cv2
import numpy as np


def _to_gray_resized(img, size=(200, 200)):
    """Convert image (numpy array or bytes) to grayscale 200x200."""
    if isinstance(img, (bytes, bytearray)):
        arr = np.frombuffer(bytes(img), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _tight_face_crop(img_bgr):
    """
    Try to detect and crop just the face region using OpenCV Haar cascade.
    Falls back to centre-crop if no face detected.
    """
    import os
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # try OpenCV's built-in frontal face cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cascade_path):
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            # pick largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # add 20% padding
            pad_x = int(w * 0.20)
            pad_y = int(h * 0.20)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img_bgr.shape[1], x + w + pad_x)
            y2 = min(img_bgr.shape[0], y + h + pad_y)
            return img_bgr[y1:y2, x1:x2]

    # fallback: centre-crop to square
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img_bgr[y0:y0+side, x0:x0+side]


def _expand_to_face(img_bgr, expand_ratio=0.5):
    """
    Pad the image outward so Haar cascade has enough context to detect a face.
    Used for QR photos which are already tightly cropped to the face.
    Pads with white (typical card background colour).
    """
    h, w = img_bgr.shape[:2]
    pad_h = int(h * expand_ratio)
    pad_w = int(w * expand_ratio)
    return cv2.copyMakeBorder(img_bgr, pad_h, pad_h, pad_w, pad_w,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def compare_photos(qr_photo, card_photo):
    """
    Compare QR-embedded photo with card face crop.
    Both images are face-aligned before SSIM comparison to handle
    framing differences (QR photo is tight crop, card photo has more context).
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return _fallback_mse_only(qr_photo, card_photo)

    # decode to BGR numpy arrays
    def to_bgr(img):
        if isinstance(img, (bytes, bytearray)):
            arr = np.frombuffer(bytes(img), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    bgr1 = to_bgr(qr_photo)
    bgr2 = to_bgr(card_photo)

    if bgr1 is None or bgr2 is None:
        return {
            "match": None, "confidence": 0.0, "ssim": 0.0,
            "mse": 0.0, "decision": "UNAVAILABLE", "fraud_contribution": 0,
            "error": "Could not decode one or both photos",
        }

    # QR photo is already a tight face crop — expand it so Haar has context,
    # then re-crop to face. Card photo has more context, crop directly.
    bgr1 = _tight_face_crop(_expand_to_face(bgr1, expand_ratio=0.5))
    bgr2 = _tight_face_crop(bgr2)

    # resize to same dims and convert to grayscale
    size = (200, 200)
    g1 = cv2.cvtColor(cv2.resize(bgr1, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(bgr2, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    score, _ = ssim(g1, g2, full=True)
    mse = float(np.mean((g1.astype(float) - g2.astype(float)) ** 2))

    if score > 0.75:
        decision, match, fraud_contribution = "MATCH", True, 20
    elif score >= 0.50:
        decision, match, fraud_contribution = "SUSPICIOUS", None, 10
    else:
        decision, match, fraud_contribution = "NO_MATCH", False, 0

    return {
        "match":              match,
        "confidence":         round(float(score), 4),
        "ssim":               round(float(score), 4),
        "mse":                round(mse, 2),
        "decision":           decision,
        "fraud_contribution": fraud_contribution,
        "error":              None,
    }


def _fallback_mse_only(qr_photo, card_photo):
    """Fallback when skimage is not available — use MSE only."""
    g1 = _to_gray_resized(qr_photo)
    g2 = _to_gray_resized(card_photo)

    if g1 is None or g2 is None:
        return {"match": None, "confidence": 0.0, "ssim": 0.0,
                "mse": 0.0, "decision": "UNAVAILABLE", "fraud_contribution": 0,
                "error": "skimage not available and could not decode photos"}

    mse = float(np.mean((g1.astype(float) - g2.astype(float)) ** 2))
    # normalise MSE to 0-1 similarity (lower MSE = more similar)
    sim = max(0.0, 1.0 - mse / 10000.0)

    if sim > 0.75:
        decision, match, fraud_contribution = "MATCH", True, 20
    elif sim >= 0.50:
        decision, match, fraud_contribution = "SUSPICIOUS", None, 10
    else:
        decision, match, fraud_contribution = "NO_MATCH", False, 0

    return {
        "match":              match,
        "confidence":         round(sim, 4),
        "ssim":               None,
        "mse":                round(mse, 2),
        "decision":           decision,
        "fraud_contribution": fraud_contribution,
        "error":              "skimage unavailable — MSE fallback used",
    }
