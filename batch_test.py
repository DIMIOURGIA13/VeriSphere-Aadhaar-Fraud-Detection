"""
Batch Testing Script for Aadhaar Fraud Detection
Automatically tests multiple images and generates predictions for evaluation
"""

import os
import json
from pathlib import Path
from aadhaar_pipeline.pipeline import run_pipeline
from aadhaar_pipeline.detector import load_model
from aadhaar_pipeline.tampering import load_tampering_model


def batch_test(test_dir="test_images", output_file="test_results.json"):
    """
    Test all images in test_dir and save results
    
    Expected directory structure:
    test_images/
        genuine/
            card1.jpg
            card2.jpg
        fraud/
            fake1.jpg
            fake2.jpg
    
    Args:
        test_dir: Path to test images directory
        output_file: Path to save results JSON
    """
    
    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"❌ Test directory not found: {test_dir}")
        print("\nExpected structure:")
        print("  test_images/")
        print("    genuine/")
        print("      card1.jpg")
        print("      card2.jpg")
        print("    fraud/")
        print("      fake1.jpg")
        print("      fake2.jpg")
        return
    
    # Load models once (faster than reloading for each image)
    print("Loading models...")
    yolo = load_model("aadhaar_best.pt")
    resnet = load_tampering_model(None, device="cpu")
    print("✓ Models loaded\n")
    
    results = {
        "y_true": [],
        "y_pred": [],
        "details": []
    }
    
    # Test genuine cards
    genuine_dir = test_path / "genuine"
    if genuine_dir.exists():
        print(f"Testing genuine cards from: {genuine_dir}")
        for img_path in sorted(genuine_dir.glob("*.jpg")) + sorted(genuine_dir.glob("*.png")):
            print(f"  Processing: {img_path.name}...", end=" ")
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
                
                # Convert verdict to binary (0=Genuine, 1=Fraud)
                # You can adjust this logic based on how you want to treat "Suspicious"
                pred = 0 if verdict == "Genuine" else 1
                
                results["y_true"].append(0)  # ground truth: genuine
                results["y_pred"].append(pred)
                results["details"].append({
                    "filename": img_path.name,
                    "ground_truth": "Genuine",
                    "predicted": verdict,
                    "fraud_score": fraud_score,
                    "correct": pred == 0
                })
                
                status = "✓" if pred == 0 else "✗"
                print(f"{status} {verdict} (score: {fraud_score:.3f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        print(f"⚠ No genuine/ subdirectory found in {test_dir}")
    
    print()
    
    # Test fraud cards
    fraud_dir = test_path / "fraud"
    if fraud_dir.exists():
        print(f"Testing fraud cards from: {fraud_dir}")
        for img_path in sorted(fraud_dir.glob("*.jpg")) + sorted(fraud_dir.glob("*.png")):
            print(f"  Processing: {img_path.name}...", end=" ")
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
                
                # Convert verdict to binary
                pred = 0 if verdict == "Genuine" else 1
                
                results["y_true"].append(1)  # ground truth: fraud
                results["y_pred"].append(pred)
                results["details"].append({
                    "filename": img_path.name,
                    "ground_truth": "Fraud",
                    "predicted": verdict,
                    "fraud_score": fraud_score,
                    "correct": pred == 1
                })
                
                status = "✓" if pred == 1 else "✗"
                print(f"{status} {verdict} (score: {fraud_score:.3f})")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        print(f"⚠ No fraud/ subdirectory found in {test_dir}")
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary
    total = len(results["y_true"])
    correct = sum(1 for i in range(total) if results["y_true"][i] == results["y_pred"][i])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print("\n" + "="*60)
    print("BATCH TEST SUMMARY")
    print("="*60)
    print(f"Total images tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("To generate full evaluation report, copy these arrays to evaluate_model.py:")
    print(f"y_true = {results['y_true']}")
    print(f"y_pred = {results['y_pred']}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    import sys
    
    # Allow custom test directory via command line
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    
    print("="*60)
    print("AADHAAR FRAUD DETECTION - BATCH TESTING")
    print("="*60)
    print()
    
    results = batch_test(test_dir)
    
    if results and len(results["y_true"]) > 0:
        print("\nNext steps:")
        print("1. Copy the y_true and y_pred arrays printed above")
        print("2. Paste them into evaluate_model.py (replace sample data)")
        print("3. Run: py evaluate_model.py")
        print("4. Use generated confusion_matrix.png and metrics_visualization.png in your report")
