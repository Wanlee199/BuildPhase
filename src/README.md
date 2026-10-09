# HƯỚNG DẪN CHẠY GREENSM HUMAN-IN-THE-LOOP ACTIVE LEARNING PIPELINE (V4.0)

Tất cả mã nguồn Python phục vụ quy trình tự động hóa V4.0 (Mentor-Aligned & Enterprise-Ready) đã được đóng gói hoàn chỉnh trong thư mục [`src/`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src).

---

## 🛠️ CẤU TRÚC BỘ TOOL TRONG `src/`

- [`src/data_profiling.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/data_profiling.py): Lọc trùng lặp Dual-Stream FastDedup (pHash, Keep-Most-Vehicles) & Gán tag Metadata 5 lát cắt (Night, Blur, Small Object, Occlusion, Density).
- [`src/active_selection.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/active_selection.py): Khai phá ảnh khó Target-Class Confusion Margin ($U_i$), Dynamic Weak-slice Weight ($W_i$), Prediction Anomaly ($A_i$), Effort Score ($E_i$), **PCA 32D + k-Center Greedy Core-Set Algorithm** & **Cross-Model Consensus Oracle (RT-DETRv4 + YOLO11n)**.
- [`src/cvat_automation.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/cvat_automation.py): Đóng gói pre-labels cho Tier 2 CVAT Review & Tự động đóng gói Dataset V1 chuẩn **5 Classes (`greensm`, `car`, `motorcycle`, `bus`, `truck`)**.
- [`src/train_yolo11n.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/train_yolo11n.py): Huấn luyện YOLO11n với Head Restructuring, Backbone Freezing 10 epochs, Class-Weighted Loss ($\alpha_{\text{greensm}}=2.0$), AdamW, Multi-seed Run & Tự động xuất file niêm phong `manifest.yaml`.
- [`src/evaluate_benchmark.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/evaluate_benchmark.py): Đánh giá đa lát cắt trên Fixed Test Set, kiểm định ý nghĩa thống kê ($\Delta mAP > 2\sigma_{\text{seed}}$) trên 4 nhánh A/B và xuất báo cáo Khung 3 Trụ cột ROI `BAO_CAO_BENCHMARK_A_B_TEST.md`.
- [`src/export_quantize.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/export_quantize.py): Xuất mô hình sang ONNX FP16 & Lượng tử hóa **OpenVINO INT8** (1.6 MB, >122 FPS trên CPU Intel), đo đạc Benchmark Latency/FPS.
- [`src/colab_runner.py`](file:///c:/Users/Admin/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/src/colab_runner.py): Master runner chạy toàn bộ 6 bước nối tiếp chỉ bằng 1 dòng lệnh.

---

## 📋 HƯỚNG DẪN CHẠY TRÊN GOOGLE COLAB / SERVER

### Cell 1: Mount Google Drive & Cài đặt Thư viện
```python
from google.colab import drive
drive.mount('/content/drive')

# Cài đặt các thư viện cần thiết
!pip install -q ultralytics imagehash opencv-python-headless scikit-learn pyyaml requests openvino-dev
```

### Cell 2: Chạy Master Pipeline (Toàn bộ 6 Bước Tự Động V4.0)
```python
!python src/colab_runner.py \
    --data-dir /content/drive/MyDrive/GreenSM_Data/raw_images \
    --output-dir /content/drive/MyDrive/GreenSM_Outputs \
    --top-k 200 \
    --epochs 50
```

---

## ⚙️ CHẠY TỪNG BƯỚC THỦ CÔNG (TÙY CHỈNH CHUYÊN SÂU)

### Bước 1: Lọc trùng Dual-Stream FastDedup & Profiling
```bash
!python src/data_profiling.py \
    --data-dir /content/drive/MyDrive/GreenSM_Data/raw_images \
    --output-json outputs/dataset_metadata.json \
    --hash-thresh 5
```

### Bước 2: Chọn Ảnh Khó Multi-Class (PCA 32D + k-Center Greedy Core-Set)
```bash
!python src/active_selection.py \
    --metadata-json outputs/dataset_metadata.json \
    --top-k 200 \
    --output-json outputs/active_selection_results.json
```

### Bước 3: Đóng gói Pre-labels CVAT / Tạo Dataset V1 (5 Classes)
```bash
!python src/cvat_automation.py \
    --action assemble \
    --selection-json outputs/active_selection_results.json \
    --output-dir outputs/dataset_v1
```

### Bước 4: Train YOLO11n Multi-Seed (3 Seeds: 42 123 456)
```bash
!python src/train_yolo11n.py \
    --data-yaml outputs/dataset_v1/dataset.yaml \
    --seeds 42 123 456 \
    --epochs 50 \
    --freeze-layers 10 \
    --project-dir outputs/runs \
    --exp-id greensm_v1
```

### Bước 5: Đánh giá Benchmark 4 Nhánh & Xuất Báo cáo 3 Trụ Cột ROI
```bash
!python src/evaluate_benchmark.py \
    --test-yaml outputs/dataset_v1/dataset.yaml \
    --seed-summary-json outputs/runs/greensm_v1_multi_seed_summary.json \
    --output-report outputs/BAO_CAO_BENCHMARK_A_B_TEST.md
```

### Bước 6: Xuất Mô hình ONNX & Lượng tử hóa OpenVINO INT8 trên Edge CPU
```bash
!python src/export_quantize.py \
    --weights outputs/runs/greensm_v1_seed_42/weights/best.pt \
    --output-dir outputs/exported_models
```
