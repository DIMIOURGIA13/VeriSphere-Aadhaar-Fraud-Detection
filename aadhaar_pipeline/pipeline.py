"""
Aadhaar Fraud Detection Pipeline
"""

import argparse
import json
import cv2
from difflib import SequenceMatcher

from aadhaar_pipeline.detector      import load_model, detect_regions, get_crops_by_class, draw_detections
from aadhaar_pipeline.ocr           import extract_all_text, normalize_aadhaar
from aadhaar_pipeline.validator     import validate_aadhaar_number
from aadhaar_pipeline.qr_validation import QRValidator
from aadhaar_pipeline.consistency   import run_all_checks
from aadhaar_pipeline.tampering     import load_tampering_model, predict_tampering
from aadhaar_pipeline.decision      import make_decision


def _refine_name_with_qr(ocr_name, qr_name):
    """
    OCR often picks up Hindi label noise before the actual name,
    e.g. 'Tangi fzRr Dheeraj Singha' when QR says 'Dheeraj Singha'.
    Slide a word-window over the OCR output and return the window
    that best matches the QR name (if it's a meaningful improvement).
    """
    ocr_words = ocr_name.split()
    n, m = len(ocr_words), len(qr_name.split())

    if n == 0 or m == 0:
        return None

    best_score, best_window = 0.0, None

    for size in range(max(1, m - 1), m + 2):
        for start in range(n - size + 1):
            window = " ".join(ocr_words[start:start + size])
            score  = SequenceMatcher(None, window.lower(), qr_name.lower()).ratio()
            if score > best_score:
                best_score, best_window = score, window

    orig_score = SequenceMatcher(None, ocr_name.lower(), qr_name.lower()).ratio()
    if best_window and best_score > orig_score and best_score >= 0.60:
        return best_window
    return None


def run_pipeline(image_path, yolo_weights, resnet_weights=None, device="cpu",
                 save_vis=None, verbose=True, yolo_model=None, resnet_tuple=None):
    """Run the full pipeline on a single Aadhaar card image."""

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't load image: {image_path}")
    log(verbose, f"Loaded image: {image_path}  ({image.shape[1]}x{image.shape[0]})")

    # step 1 — YOLO region detection
    yolo = yolo_model if yolo_model is not None else load_model(yolo_weights)
    detections = detect_regions(yolo, image)
    crops = get_crops_by_class(detections)
    log(verbose, f"YOLO found {len(detections)} regions: {[d['class_name'] for d in detections]}")

    if save_vis:
        cv2.imwrite(save_vis, draw_detections(image, detections))

    # step 2 — OCR
    ocr_fields = extract_all_text(crops)

    # step 3 — Verhoeff: pick the Aadhaar candidate that passes checksum
    candidates = ocr_fields.get("aadhaar_candidates", [])
    verified_aadhaar, verhoeff_passed = "", None

    for candidate in candidates:
        result = validate_aadhaar_number(candidate)
        if result["valid"]:
            verified_aadhaar, verhoeff_passed = candidate, result
            log(verbose, f"  Verhoeff PASSED: {candidate}")
            break
        else:
            log(verbose, f"  Verhoeff failed: {candidate} — {result['reason']}")

    if not verified_aadhaar:
        verified_aadhaar = ocr_fields.get("aadhaar_number", "")
        verhoeff_passed  = validate_aadhaar_number(verified_aadhaar)

    ocr_fields["aadhaar_number"] = verified_aadhaar
    log(verbose, f"OCR fields (pre-QR): {ocr_fields}")

    # step 4 — QR decode
    qr_validator = QRValidator()
    qr_crop      = crops.get("qr_code", [None])[0]
    qr_result    = qr_validator.validate_qr(image, ocr_fields, qr_crop=qr_crop)
    qr_fields    = qr_result.get("qr_parsed_data") or {}

    log(verbose, f"QR found: {qr_result['qr_found']} | valid: {qr_result['qr_valid']} | "
                 f"format: {qr_result.get('qr_format')} | match: {qr_result.get('match_score', 0):.0f}%")
    if qr_result.get("error"):
        log(verbose, f"  QR error: {qr_result['error']}")

    # step 4b — refine OCR name using QR name (removes Hindi label noise)
    # Must happen before consistency checks but comparison_details also needs updating
    qr_name = qr_fields.get("name", "")
    if qr_name and ocr_fields.get("name"):
        refined = _refine_name_with_qr(ocr_fields["name"], qr_name)
        if refined:
            log(verbose, f"  Name refined: '{ocr_fields['name']}' → '{refined}'")
            ocr_fields["name"] = refined
            # update comparison_details so the name check uses the refined value
            comp = qr_result.get("comparison_details", {})
            if "name" in comp:
                from difflib import SequenceMatcher as _SM
                qn = comp["name"].get("qr_value", qr_name)
                on = refined
                match = _SM(None, on.lower(), qn.lower()).ratio() >= 0.75 or \
                        on.lower() in qn.lower() or qn.lower() in on.lower()
                comp["name"] = {"qr_value": qn, "ocr_value": on, "match": match}
                qr_result["comparison_details"] = comp
                # recalculate match_score from updated comparison_details
                total = len(comp)
                matched = sum(1 for v in comp.values() if v.get("match") is True)
                qr_result["match_score"] = (matched / total * 100) if total else 0

    # step 5 — consistency checks
    verhoeff = verhoeff_passed
    log(verbose, f"Verhoeff: {verhoeff['reason']} ({verified_aadhaar})")

    consistency = run_all_checks(ocr_fields, qr_fields)
    consistency["qr_available"]   = qr_result["qr_found"] and qr_result["qr_valid"]
    consistency["qr_format"]      = qr_result.get("qr_format")
    consistency["qr_match_score"] = qr_result.get("match_score", 0)
    consistency["qr_comparison"]  = qr_result.get("comparison_details", {})
    consistency["qr_error"]       = qr_result.get("error")
    log(verbose, f"Consistency: {consistency['overall_score']} | QR match: {consistency['qr_match_score']:.0f}%")

    # step 6 — tampering detection
    resnet_tuple = resnet_tuple if resnet_tuple is not None else load_tampering_model(resnet_weights, device=device)
    tampering = predict_tampering(resnet_tuple, image, device=device)
    log(verbose, f"Tampering: {tampering['label']} ({tampering['confidence']:.2f})")

    # step 7 — final decision
    decision = make_decision(detections, verhoeff, consistency, tampering)
    log(verbose, f"\n{'='*45}\n{decision}\n{'='*45}")

    return {
        "image_path":         image_path,
        "detections":         [{k: v for k, v in d.items() if k != "crop"} for d in detections],
        "ocr_fields":         ocr_fields,
        "aadhaar_candidates": candidates,
        "qr_result":          {k: v for k, v in qr_result.items() if k not in ("qr_parsed_data",)},
        "qr_fields":          qr_fields,
        "verhoeff":           verhoeff,
        "consistency":        consistency,
        "tampering":          tampering,
        "verdict":            decision.verdict,
        "fraud_score":        decision.fraud_score,
        "confidence":         decision.confidence,
        "reasons":            decision.reasons,
    }


def log(verbose, msg):
    if verbose:
        print(msg)


def main():
    parser = argparse.ArgumentParser(description="Aadhaar Fraud Detection")
    parser.add_argument("--image",    required=True)
    parser.add_argument("--yolo",     required=True)
    parser.add_argument("--resnet",   default=None)
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--save-vis", default=None)
    parser.add_argument("--json",     action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        image_path=args.image, yolo_weights=args.yolo,
        resnet_weights=args.resnet, device=args.device,
        save_vis=args.save_vis, verbose=not args.json,
    )
    if args.json:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
