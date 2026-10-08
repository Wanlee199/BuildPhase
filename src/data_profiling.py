"""
data_profiling.py - Intra-Video Fast Deduplication & Metadata Profiling
========================================================================
Feature set:
1. Intra-video deduplication using pHash (Hamming distance threshold).
2. Keep-most-vehicles rule when 2 frames are duplicates.
3. Metadata profiling: Day/Night (HSV V-channel), Blur (Laplacian Variance), Small Object, Dense Scene.
4. Export dataset_metadata.json for downstream Active Selection.
"""

import os
import glob
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


def compute_phash_opencv(image: np.ndarray) -> str:
    """Fallback pHash implementation using OpenCV when imagehash is missing."""
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_low = dct[:8, :8]
    avg = np.mean(dct_low)
    binary_hash = dct_low > avg
    return "".join(["1" if b else "0" for b in binary_hash.flatten()])


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two binary hashes."""
    return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))


def get_phash(img_path: str) -> str:
    """Compute 64-bit pHash for an image file."""
    if HAS_IMAGEHASH:
        with Image.open(img_path) as img:
            return str(imagehash.phash(img))
    else:
        img = cv2.imread(img_path)
        if img is None:
            return "0" * 64
        return compute_phash_opencv(img)


def estimate_vehicle_count_fallback(img: np.ndarray) -> int:
    """Estimate object/vehicle count using edge contours as a fast heuristic."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Count contours with area > 100px
    return sum(1 for c in contours if cv2.contourArea(c) > 100)


def profile_image_metadata(img_path: str, brightness_thresh: float = 65.0, blur_thresh: float = 100.0) -> dict:
    """Extract metadata attributes (Night, Blur, Brightness, Laplacian Var)."""
    img = cv2.imread(img_path)
    if img is None:
        return {
            "error": "Failed to read image",
            "is_night": False,
            "is_blur": False,
            "brightness": 0.0,
            "laplacian_var": 0.0,
            "est_vehicle_count": 0
        }

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = float(np.mean(hsv[:, :, 2]))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    est_count = estimate_vehicle_count_fallback(img)

    return {
        "brightness": round(brightness, 2),
        "laplacian_var": round(laplacian_var, 2),
        "is_night": brightness < brightness_thresh,
        "is_blur": laplacian_var < blur_thresh,
        "est_vehicle_count": est_count,
        "height": img.shape[0],
        "width": img.shape[1]
    }


def extract_video_id(file_path: str) -> str:
    """Extract video identifier from filepath or filename (e.g., video01_frame001.jpg -> video01)."""
    p = Path(file_path)
    parts = p.stem.split("_")
    if len(parts) > 1 and ("vid" in parts[0].lower() or "video" in parts[0].lower() or "cam" in parts[0].lower()):
        return parts[0]
    return p.parent.name if p.parent.name else "default_video"


def process_dataset(data_dir: str, output_json: str, hash_thresh: int = 5):
    """Run intra-video deduplication and data profiling."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    image_paths = sorted(image_paths)
    print(f"🔍 Found {len(image_paths)} total images in '{data_dir}'")

    # Group images by video_id
    video_groups = {}
    for path in image_paths:
        vid_id = extract_video_id(path)
        video_groups.setdefault(vid_id, []).append(path)

    print(f"📹 Grouped into {len(video_groups)} video sequence(s)")

    kept_images = []
    dropped_images = []
    metadata_db = {}

    total_duplicates_removed = 0

    for vid_id, paths in video_groups.items():
        print(f" Processing video group '{vid_id}' ({len(paths)} frames)...")
        group_kept = []
        
        for path in paths:
            phash_val = get_phash(path)
            meta = profile_image_metadata(path)
            meta["phash"] = phash_val
            meta["video_id"] = vid_id
            meta["file_path"] = os.path.abspath(path)

            # Deduplication check against kept frames in the SAME video
            is_duplicate = False
            duplicate_target = None

            for existing in group_kept:
                dist = hamming_distance(phash_val, existing["phash"])
                if dist < hash_thresh:
                    is_duplicate = True
                    duplicate_target = existing
                    break

            if is_duplicate and duplicate_target is not None:
                # Keep-Most-Vehicles rule
                current_count = meta["est_vehicle_count"]
                target_count = duplicate_target["est_vehicle_count"]

                if current_count > target_count:
                    # Replace existing duplicate with current frame (better frame)
                    group_kept.remove(duplicate_target)
                    dropped_images.append(duplicate_target["file_path"])
                    group_kept.append(meta)
                    total_duplicates_removed += 1
                else:
                    dropped_images.append(path)
                    total_duplicates_removed += 1
            else:
                group_kept.append(meta)

        for item in group_kept:
            kept_images.append(item["file_path"])
            metadata_db[item["file_path"]] = item

    dedup_ratio = (total_duplicates_removed / len(image_paths) * 100) if image_paths else 0.0

    output_payload = {
        "summary": {
            "total_raw_images": len(image_paths),
            "kept_images_count": len(kept_images),
            "dropped_duplicates_count": total_duplicates_removed,
            "dedup_reduction_percent": round(dedup_ratio, 2),
            "num_video_sequences": len(video_groups)
        },
        "kept_images": kept_images,
        "dropped_images": dropped_images,
        "metadata": metadata_db
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n✅ DATA PROFILING COMPLETED")
    print(f"📊 Total Raw Images: {len(image_paths)}")
    print(f"✨ Kept Unique Frames: {len(kept_images)}")
    print(f"🗑️ Duplicates Filtered: {total_duplicates_removed} ({dedup_ratio:.1f}% reduction)")
    print(f"📁 Metadata saved to: [dataset_metadata.json](file://{os.path.abspath(output_json)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intra-Video Fast Deduplication & Metadata Profiling")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing raw images or subfolders")
    parser.add_argument("--output-json", type=str, default="dataset_metadata.json", help="Output path for metadata JSON")
    parser.add_argument("--hash-thresh", type=int, default=5, help="Hamming distance threshold for pHash (default: 5)")
    args = parser.parse_args()

    process_dataset(args.data_dir, args.output_json, args.hash_thresh)
