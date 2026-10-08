# HƯỚNG DẪN CHẠY GREENSM ACTIVE LEARNING PIPELINE TRÊN GOOGLE COLAB

Tất cả mã nguồn Python phục vụ quy trình tự động hóa đã được đóng gói hoàn chỉnh trong thư mục `src/`.

---

## 🛠️ CẤU TRÚC BỘ TOOL TRONG `src/`

- `src/data_profiling.py`: Lọc trùng lặp Intra-video (Keep-Most-Vehicles) & Gán tag Metadata tự động (Night, Blur, Small Object).
- `src/active_selection.py`: Khai phá ảnh khó 1-Class Confidence Margin Uncertainty, Dynamic Weak-slice Weight, Effort Score & K-Means Diversity Clustering.
- `src/cvat_automation.py`: Đóng gói pre-labels cho Tier 2 CVAT Review & Tự động đóng gói Dataset V1.
- `src/train_yolo11n.py`: Huấn luyện YOLO11n với Multi-seed Run & Tự động xuất file `manifest.yaml`.
- `src/evaluate_benchmark.py`: Đánh giá đa lát cắt trên Fixed Test Set, tính sai số dao động tự nhiên $\sigma_{seed}$ và xuất báo cáo `BAO_CAO_BENCHMARK_A_B_TEST.md`.
- `src/colab_runner.py`: Master runner chạy toàn bộ 5 bước nối tiếp chỉ bằng 1 dòng lệnh.

---

## 📋 HƯỚNG DẪN CÁC CELL CHẠY TRÊN GOOGLE COLAB

### Cell 1: Mount Google Drive & Cài đặt Thư viện
```python
from google.colab import drive
drive.mount('/content/drive')

# Cài đặt các thư viện cần thiết
!pip install -q ultralytics imagehash opencv-python-headless scikit-learn pyyaml requests
```

### Cell 2: Chạy Master Pipeline (Toàn bộ 5 Bước Tự Động)
```python
!python src/colab_runner.py \
    --data-dir /content/drive/MyDrive/GreenSM_Data/raw_images \
    --output-dir /content/drive/MyDrive/GreenSM_Outputs \
    --top-k 200 \
    --epochs 30
```

---

## ⚙️ CHẠY TỪNG BƯỚC THỦ CÔNG (NẾU MUỐN KIỂM SOÁT TỪNG GIAI ĐOẠN)

### Bước 1: Lọc trùng Intra-Video & Profiling
```bash
!python src/data_profiling.py \
    --data-dir /content/drive/MyDrive/GreenSM_Data/raw_images \
    --output-json outputs/dataset_metadata.json \
    --hash-thresh 5
```

### Bước 2: Chọn Ảnh Khó 1-Class & Phân tầng 2-Tier HITL
```bash
!python src/active_selection.py \
    --metadata-json outputs/dataset_metadata.json \
    --top-k 200 \
    --output-json outputs/active_selection_results.json
```

### Bước 3: Đóng gói Pre-labels CVAT / Tạo Dataset V1
```bash
!python src/cvat_automation.py \
    --action package \
    --selection-json outputs/active_selection_results.json \
    --output-dir outputs/cvat_package
```

### Bước 4: Train YOLO11n Multi-Seed (3 Seeds)
```bash
!python src/train_yolo11n.py \
    --data-yaml outputs/dataset_v1/dataset.yaml \
    --seeds 42 123 999 \
    --epochs 30 \
    --project-dir outputs/runs \
    --exp-id greensm_v1
```

### Bước 5: Đánh giá Benchmark & Xuất Báo cáo Markdown
```bash
!python src/evaluate_benchmark.py \
    --test-yaml outputs/dataset_v1/dataset.yaml \
    --seed-summary-json outputs/runs/greensm_v1_multi_seed_summary.json \
    --output-report outputs/BAO_CAO_BENCHMARK_A_B_TEST.md
```
