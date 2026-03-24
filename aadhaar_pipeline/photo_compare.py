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


def compare_photos(qr_photo, card_photo):
    """
    Compare QR-embedded photo with card face crop.

    Args:
        qr_photo:   bytes (JPEG/JP2) or numpy BGR array
        card_photo: bytes or numpy BGR array

    Returns dict:
        match: bool
        confidence: float 0-1
        ssim: float
        mse: float
        decision: "MATCH" | "SUSPICIOUS" | "NO_MATCH"
        fraud_contribution: int (0-20, higher = more genuine)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return _fallback_mse_only(qr_photo, card_photo)

    g1 = _to_gray_resized(qr_photo)
    g2 = _to_gray_resized(card_photo)

    if g1 is None or g2 is None:
        return {
            "match": None, "confidence": 0.0, "ssim": 0.0,
            "mse": 0.0, "decision": "UNAVAILABLE", "fraud_contribution": 0,
            "error": "Could not decode one or both photos",
        }

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
