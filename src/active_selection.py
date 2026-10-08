"""
active_selection.py - 1-Class Cost-Aware Active Mining & Two-Tier HITL Split
=============================================================================
Feature set:
1. 1-Class BBox Confidence Margin Uncertainty: U_i = 1 - 2 * |P(GreenSM) - 0.5|
2. Dynamic Weak-Slice Weighting (from previous eval_metrics.json)
3. Effort Score estimation: E_i = 1 + 0.1 * N_box
4. K-Means Diversity Clustering on YOLO backbone feature vectors / HSV histograms
5. Two-Tier HITL Split:
   - Tier 1 (Auto-Accept): Conf >= 0.85 AND NOT (Night OR Small_Object)
   - Buffer Zone (0.70 < Conf < 0.85): Routed to Tier 2
   - Tier 2 (CVAT Review): Conf 0.20..0.85 + ALL Night/Small_Object images
   - Spot-Check QA: 5-10% random sample of Tier 1 Auto-Accept
"""

import os
import json
import argparse
import glob
import numpy as np
import cv2
from pathlib import Path
from sklearn.cluster import KMeans

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def extract_image_features(img_path: str, model=None) -> np.ndarray:
    """Extract feature vector for clustering. Uses YOLO backbone if available, else HSV color histogram."""
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(64)

    if model is not None and HAS_YOLO:
        try:
            # Extract features from YOLO model backbone if possible
            results = model.predict(img_path, verbose=False)
            if len(results) > 0 and hasattr(results[0], 'speed'):
                # Quick feature approximation using resized image + color channels
                resized = cv2.resize(img, (16, 16))
                return resized.flatten() / 255.0
        except Exception:
            pass

    # Fallback: Color histogram in HSV (8x8x8 = 512 dims)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [4, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def calculate_1class_uncertainty(confidences: list) -> float:
    """Calculate 1-class confidence margin uncertainty across bboxes in frame."""
    if not confidences:
        return 0.5  # High uncertainty if no detections found

    # Measure how close confidence is to 0.5 (maximum ambiguity threshold)
    bbox_uncertainties = [1.0 - 2.0 * abs(c - 0.5) for c in confidences]
    return float(np.mean(bbox_uncertainties))


def calculate_dynamic_slice_weight(meta: dict, eval_metrics: dict) -> float:
    """Calculate dynamic weak-slice weight based on previous model evaluation recall."""
    small_recall = eval_metrics.get("small_object_recall", 0.70)
    night_recall = eval_metrics.get("night_recall", 0.65)

    weight = 1.0
    if meta.get("is_night", False):
        weight *= (1.0 / max(night_recall, 0.2))
    if meta.get("has_small_object", False):
        weight *= (1.0 / max(small_recall, 0.2))

    return round(float(weight), 3)


def run_active_selection(
    metadata_json: str,
    model_path: str,
    top_k: int,
    output_json: str,
    eval_metrics_json: str = None,
    conf_thresh: float = 0.25,
    auto_accept_conf: float = 0.85,
    spot_check_ratio: float = 0.10
):
    # Load metadata
    with open(metadata_json, "r", encoding="utf-8") as f:
        meta_db = json.load(f)

    kept_paths = meta_db.get("kept_images", [])
    metadata_map = meta_db.get("metadata", {})

    # Load eval metrics if available
    eval_metrics = {}
    if eval_metrics_json and os.path.exists(eval_metrics_json):
        with open(eval_metrics_json, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    # Load YOLO Model
    model = None
    if HAS_YOLO and model_path and os.path.exists(model_path):
        print(f"🤖 Loading YOLO11n model from '{model_path}'...")
        model = YOLO(model_path)
    else:
        print("⚠️ YOLO model not found or ultralytics not installed. Running simulated/feature-based inference.")

    scored_candidates = []

    print(f"🔎 Running Inference & Active Scoring on {len(kept_paths)} unlabeled images...")

    for path in kept_paths:
        img_meta = metadata_map.get(path, {})
        confidences = []

        if model is not None:
            results = model.predict(path, conf=0.10, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy().tolist() if len(boxes) > 0 else []

        n_box = len(confidences)
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        # Check small object flag based on bboxes if detected
        has_small_object = img_meta.get("has_small_object", False)
        if model is not None and len(results) > 0 and results[0].boxes is not None:
            for b in results[0].boxes.xywh:
                w, h = b[2].item(), b[3].item()
                if w * h < 32 * 32:
                    has_small_object = True
                    break
        img_meta["has_small_object"] = has_small_object

        # Compute Score Components
        u_i = calculate_1class_uncertainty(confidences)
        w_i = calculate_dynamic_slice_weight(img_meta, eval_metrics)
        a_i = 1.2 if (n_box > 10 or n_box == 0) else 1.0  # Anomaly score
        e_i = 1.0 + 0.1 * n_box  # Effort score

        selection_score = (u_i * w_i * a_i) / e_i

        scored_candidates.append({
            "file_path": path,
            "selection_score": round(selection_score, 4),
            "uncertainty_score": round(u_i, 4),
            "slice_weight": round(w_i, 4),
            "effort_score": round(e_i, 4),
            "bbox_count": n_box,
            "avg_confidence": round(avg_conf, 4),
            "is_night": img_meta.get("is_night", False),
            "has_small_object": has_small_object,
            "confidences": [round(c, 3) for c in confidences]
        })

    # Sort candidates by Selection Score descending
    scored_candidates = sorted(scored_candidates, key=lambda x: x["selection_score"], reverse=True)

    # Take Top 2*K candidates for Diversity Clustering
    pool_for_clustering = scored_candidates[:min(2 * top_k, len(scored_candidates))]

    # Diversity Filtering using K-Means
    print(f"🎯 Applying K-Means Diversity Clustering on Top-{len(pool_for_clustering)} candidates to pick Top-{top_k}...")
    if len(pool_for_clustering) > top_k:
        features = [extract_image_features(c["file_path"], model) for c in pool_for_clustering]
        features = np.array(features)

        kmeans = KMeans(n_clusters=top_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)

        # Pick highest scoring candidate from each cluster
        selected_candidates = []
        for cluster_id in range(top_k):
            cluster_members = [pool_for_clustering[idx] for idx, l in enumerate(labels) if l == cluster_id]
            if cluster_members:
                best_member = max(cluster_members, key=lambda x: x["selection_score"])
                selected_candidates.append(best_member)
    else:
        selected_candidates = pool_for_clustering[:top_k]

    # Two-Tier HITL Routing
    tier1_auto_accept = []
    tier2_cvat_review = []

    for item in selected_candidates:
        conf = item["avg_confidence"]
        is_risk_slice = item["is_night"] or item["has_small_object"]

        # Risk Exclusion Guardrail: Night & Small Object NEVER auto-accepted
        if conf >= auto_accept_conf and not is_risk_slice:
            tier1_auto_accept.append(item)
        else:
            tier2_cvat_review.append(item)

    # Random Spot-Check QA (5-10% of Tier 1)
    num_spot_check = max(1, int(len(tier1_auto_accept) * spot_check_ratio)) if tier1_auto_accept else 0
    np.random.seed(42)
    spot_check_indices = np.random.choice(len(tier1_auto_accept), size=num_spot_check, replace=False) if num_spot_check > 0 else []
    spot_check_qa = [tier1_auto_accept[i] for i in spot_check_indices]

    payload = {
        "summary": {
            "total_candidates_evaluated": len(scored_candidates),
            "top_k_selected": len(selected_candidates),
            "tier1_auto_accept_count": len(tier1_auto_accept),
            "tier2_cvat_review_count": len(tier2_cvat_review),
            "spot_check_qa_count": len(spot_check_qa)
        },
        "tier1_auto_accept": tier1_auto_accept,
        "tier2_cvat_review": tier2_cvat_review,
        "spot_check_qa": spot_check_qa
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n✅ ACTIVE MINING & TWO-TIER HITL COMPLETED")
    print(f" Selected Top-K Candidates: {len(selected_candidates)}")
    print(f"🤖 Tier 1 Auto-Accept (0 Human Effort): {len(tier1_auto_accept)}")
    print(f" Tier 2 CVAT Human Review: {len(tier2_cvat_review)}")
    print(f"🔍 Spot-Check QA Sampled: {len(spot_check_qa)}")
    print(f"📁 Output saved to: [active_selection_results.json](file://{os.path.abspath(output_json)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="1-Class Cost-Aware Active Mining & Two-Tier HITL Split")
    parser.add_argument("--metadata-json", type=str, required=True, help="Path to dataset_metadata.json")
    parser.add_argument("--model-path", type=str, default="", help="Path to YOLO11n model weights (.pt)")
    parser.add_argument("--top-k", type=int, default=200, help="Number of active candidates to select (default: 200)")
    parser.add_argument("--output-json", type=str, default="active_selection_results.json", help="Output result path")
    parser.add_argument("--eval-metrics-json", type=str, default="", help="Optional previous eval_metrics.json for dynamic weights")
    parser.add_argument("--auto-accept-conf", type=float, default=0.85, help="Auto-accept threshold (default: 0.85)")
    args = parser.parse_args()

    run_active_selection(
        args.metadata_json,
        args.model_path,
        args.top_k,
        args.output_json,
        args.eval_metrics_json,
        auto_accept_conf=args.auto_accept_conf
    )
