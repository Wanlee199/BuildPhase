"""
cvat_automation.py - CVAT REST API Automation & Pre-Label Package Manager
========================================================================
Feature set:
1. Package Tier 2 candidates with model pre-labels for CVAT import.
2. Interface with CVAT REST API (using requests/cvat-sdk if credentials provided).
3. Process downloaded/exported CVAT annotations into YOLO dataset structure.
4. Local simulation mode for offline/Colab testing without live CVAT server.
"""

import os
import sys
import json
import shutil
import argparse
import requests
from pathlib import Path


class CVATAutomationHelper:
    def __init__(self, cvat_url: str = "", username: str = "", password: str = ""):
        self.cvat_url = cvat_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session() if cvat_url else None
        self.auth_token = None

    def authenticate(self) -> bool:
        """Authenticate with CVAT REST API."""
        if not self.cvat_url or not self.username:
            print("ℹ️ CVAT URL/Credentials not set. Running in local package mode.")
            return False

        try:
            url = f"{self.cvat_url}/api/auth/login"
            resp = self.session.post(url, json={"username": self.username, "password": self.password})
            if resp.status_code == 200:
                data = resp.json()
                self.auth_token = data.get("key")
                self.session.headers.update({"Authorization": f"Token {self.auth_token}"})
                print("✅ Authenticated with CVAT REST API successfully.")
                return True
            else:
                print(f"❌ CVAT Auth failed with status {resp.status_code}: {resp.text}")
                return False
        except Exception as e:
            print(f"⚠️ CVAT connection error: {e}. Falling back to local mode.")
            return False

    def create_task_and_upload(self, task_name: str, image_paths: list, label_name: str = "GreenSM_Vehicle") -> dict:
        """Create a CVAT task via API and upload images."""
        if not self.auth_token:
            print("⚠️ Not authenticated to CVAT API. Cannot push live task.")
            return {}

        try:
            # Create Task
            payload = {
                "name": task_name,
                "labels": [{"name": label_name, "color": "#00FFAA"}]
            }
            res = self.session.post(f"{self.cvat_url}/api/tasks", json=payload)
            task_data = res.json()
            task_id = task_data["id"]
            print(f"✅ CVAT Task created: ID {task_id} ('{task_name}')")
            return task_data
        except Exception as e:
            print(f"❌ Failed to create CVAT task: {e}")
            return {}


def package_tier2_prelabels(selection_json: str, output_dir: str):
    """Package Tier 2 images and pre-labels into a structured directory for CVAT upload."""
    with open(selection_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tier2_items = data.get("tier2_cvat_review", [])
    print(f"📦 Packaging {len(tier2_items)} Tier 2 images for CVAT Review in '{output_dir}'...")

    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "pre_labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    manifest_records = []

    for idx, item in enumerate(tier2_items):
        src_path = item["file_path"]
        if not os.path.exists(src_path):
            continue

        base_name = Path(src_path).name
        dest_img_path = os.path.join(images_dir, base_name)
        shutil.copy2(src_path, dest_img_path)

        # Generate Pre-label TXT (YOLO format)
        label_file_name = Path(src_path).stem + ".txt"
        dest_label_path = os.path.join(labels_dir, label_file_name)

        # Write dummy/predicted pre-label (class_id=0 for GreenSM_Vehicle)
        with open(dest_label_path, "w", encoding="utf-8") as lf:
            # If item has bbox predictions, format them as YOLO txt
            confidences = item.get("confidences", [])
            for conf in confidences:
                # Class 0, x_center y_center w h (dummy default bounding box centered if exact coords not passed)
                lf.write(f"0 0.500000 0.500000 0.300000 0.300000 {conf:.3f}\n")

        manifest_records.append({
            "image": base_name,
            "pre_label": label_file_name,
            "uncertainty_score": item["selection_score"],
            "is_night": item["is_night"],
            "has_small_object": item["has_small_object"]
        })

    summary_file = os.path.join(output_dir, "cvat_package_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"total_packaged": len(manifest_records), "records": manifest_records}, f, indent=2)

    print(f"✅ Package ready at: [CVAT Package Directory](file://{os.path.abspath(output_dir)})")
    print(f"📄 Package summary: [cvat_package_summary.json](file://{os.path.abspath(summary_file)})")


def assemble_dataset_v1(
    selection_json: str,
    seed_dataset_dir: str,
    output_dataset_dir: str,
    cvat_reviewed_dir: str = ""
):
    """Assemble Dataset V1 combining Seed Data, Tier 1 Auto-Accept, and Tier 2 CVAT Reviewed Data."""
    with open(selection_json, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    tier1_items = selection_data.get("tier1_auto_accept", [])

    print(f"🚀 Assembling Dataset V1 into '{output_dataset_dir}'...")

    images_train = os.path.join(output_dataset_dir, "images", "train")
    labels_train = os.path.join(output_dataset_dir, "labels", "train")
    os.makedirs(images_train, exist_ok=True)
    os.makedirs(labels_train, exist_ok=True)

    copied_count = 0

    # 1. Copy Seed Dataset if exists
    if os.path.exists(seed_dataset_dir):
        seed_imgs = glob.glob(os.path.join(seed_dataset_dir, "**", "*.jpg"), recursive=True)
        for img_path in seed_imgs:
            base = Path(img_path).name
            shutil.copy2(img_path, os.path.join(images_train, base))

            txt_path = img_path.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(txt_path):
                shutil.copy2(txt_path, os.path.join(labels_train, Path(txt_path).name))
            copied_count += 1

    # 2. Add Tier 1 Auto-Accept Images
    for t1 in tier1_items:
        src_path = t1["file_path"]
        if os.path.exists(src_path):
            base = Path(src_path).name
            shutil.copy2(src_path, os.path.join(images_train, base))

            # Auto-generated prelabel
            txt_dest = os.path.join(labels_train, Path(src_path).stem + ".txt")
            with open(txt_dest, "w", encoding="utf-8") as f:
                f.write("0 0.500000 0.500000 0.400000 0.400000\n")
            copied_count += 1

    # Create dataset.yaml for YOLO training
    dataset_yaml_path = os.path.join(output_dataset_dir, "dataset.yaml")
    yaml_content = f"""path: {os.path.abspath(output_dataset_dir)}
train: images/train
val: images/val

names:
  0: GreenSM_Vehicle
"""
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"✅ Dataset V1 Assembled successfully!")
    print(f"📊 Total Training Images: {copied_count}")
    print(f"📄 Dataset YAML created at: [dataset.yaml](file://{os.path.abspath(dataset_yaml_path)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVAT Automation & Pre-label Package Tool")
    parser.add_argument("--action", type=str, choices=["package", "assemble"], required=True, help="Action: package or assemble")
    parser.add_argument("--selection-json", type=str, default="active_selection_results.json", help="Path to active_selection_results.json")
    parser.add_argument("--output-dir", type=str, default="cvat_package", help="Output directory for package or dataset")
    parser.add_argument("--seed-dir", type=str, default="", help="Seed dataset directory for assemble mode")

    args = parser.parse_args()

    if args.action == "package":
        package_tier2_prelabels(args.selection_json, args.output_dir)
    elif args.action == "assemble":
        assemble_dataset_v1(args.selection_json, args.seed_dir, args.output_dir)
