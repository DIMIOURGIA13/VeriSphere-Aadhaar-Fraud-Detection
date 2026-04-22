"""
Simple Batch Testing Script for Aadhaar Fraud Detection
Tests all images in a directory and generates a summary report
"""

import os
import json
from pathlib import Path
from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model


def simple_batch_test(image_dir, output_file="batch_test_results.json", max_images=20):
    """
    Test all images in a directory and save results
    
    Args:
        image_dir: Path to directory containing Aadhaar card images
        output_file: Path to save results JSON
        max_images: Maximum number of images to test (default 20 for quick testing)
    """
    
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"❌ Image directory not found: {image_dir}")
        return
    
    # Find all image files
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    
    if not image_files:
        print(f"❌ No images found in: {image_dir}")
        return
    
    # Limit to max_images for faster testing
    if len(image_files) > max_images:
        print(f"⚠ Found {len(image_files)} images, testing first {max_images} only")
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to test\n")
    
    # Load models once (faster than reloading for each image)
    print("Loading models...")
    yolo = load_model("aadhaar_best.pt")
    resnet = load_tampering_model(None, device="cpu")
    print("✓ Models loaded\n")
    
    results = {
        "total_tested": 0,
        "verdicts": {"Genuine": 0, "Suspicious": 0, "Fake": 0},
        "avg_fraud_score": 0,
        "details": []
    }
    
    fraud_scores = []
    
    print("="*70)
    print("TESTING AADHAAR CARDS")
    print("="*70)
    
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {img_path.name}")
        print("-"*70)
        
        try:
            result = run_pipeline(
                image_path=str(img_path),
                yolo_weights="aadhaar_best.pt",
                device="cpu",
                verbose=False,
                yolo_model=yolo,
                resnet_tuple=resnet
            )
            
            verdict = result["verdict"]
            fraud_score = result["fraud_score"]
            confidence = result["confidence"]
            reasons = result.get("reasons", [])
            
            # Update counters
            results["verdicts"][verdict] += 1
            results["total_tested"] += 1
            fraud_scores.append(fraud_score)
            
            # Store details
            results["details"].append({
                "filename": img_path.name,
                "verdict": verdict,
                "fraud_score": fraud_score,
                "confidence": confidence,
                "reasons": reasons
            })
            
            # Print result
            verdict_emoji = {"Genuine": "✓", "Suspicious": "⚠", "Fake": "✗"}[verdict]
            verdict_color = {"Genuine": "GREEN", "Suspicious": "YELLOW", "Fake": "RED"}[verdict]
            
            print(f"  Verdict: {verdict_emoji} {verdict} ({verdict_color})")
            print(f"  Fraud Score: {fraud_score:.3f}")
            print(f"  Confidence: {confidence:.3f}")
            if reasons:
                print(f"  Reasons:")
                for reason in reasons[:3]:  # Show first 3 reasons
                    print(f"    • {reason}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["details"].append({
                "filename": img_path.name,
                "error": str(e)
            })
    
    # Calculate statistics
    if fraud_scores:
        results["avg_fraud_score"] = sum(fraud_scores) / len(fraud_scores)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("BATCH TEST SUMMARY")
    print("="*70)
    print(f"Total images tested: {results['total_tested']}")
    print(f"\nVerdict Distribution:")
    print(f"  ✓ Genuine:    {results['verdicts']['Genuine']:3d} ({results['verdicts']['Genuine']/results['total_tested']*100:.1f}%)")
    print(f"  ⚠ Suspicious: {results['verdicts']['Suspicious']:3d} ({results['verdicts']['Suspicious']/results['total_tested']*100:.1f}%)")
    print(f"  ✗ Fake:       {results['verdicts']['Fake']:3d} ({results['verdicts']['Fake']/results['total_tested']*100:.1f}%)")
    print(f"\nAverage Fraud Score: {results['avg_fraud_score']:.3f}")
    print(f"\n✓ Detailed results saved to: {output_file}")
    print("="*70)
    
    # Generate sample data for evaluate_model.py
    print("\n" + "="*70)
    print("FOR EVALUATION (if you know ground truth):")
    print("="*70)
    print("If you manually verify these results, you can create evaluation data:")
    print("\nExample:")
    print("# Assuming first 10 are genuine, next 10 are fraud")
    print("y_true = [0]*10 + [1]*10")
    print("y_pred = [", end="")
    for i, detail in enumerate(results["details"][:20]):
        if "verdict" in detail:
            pred = 0 if detail["verdict"] == "Genuine" else 1
            print(f"{pred}", end=", " if i < 19 else "")
    print("]")
    print("\nThen run: py evaluate_model.py")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    
    # Default to Kaggle dataset test images
    default_dir = r"C:\Users\Asus\Desktop\Project\Kaggle aadhaar dataset\test\images"
    image_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    max_images = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    print("="*70)
    print("AADHAAR FRAUD DETECTION - SIMPLE BATCH TESTING")
    print("="*70)
    print(f"Image directory: {image_dir}")
    print(f"Max images: {max_images}")
    print("="*70)
    print()
    
    results = simple_batch_test(image_dir, max_images=max_images)
    
    if results:
        print("\n✓ Testing complete!")
        print("\nNext steps:")
        print("1. Review batch_test_results.json for detailed results")
        print("2. If you know which cards are genuine/fraud, create y_true and y_pred arrays")
        print("3. Update evaluate_model.py with your data")
        print("4. Run: py evaluate_model.py")
