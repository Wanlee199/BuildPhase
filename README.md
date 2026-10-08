# GREENSM HUMAN-IN-THE-LOOP ACTIVE LEARNING LAB (M50)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model YOLO11n](https://img.shields.io/badge/Model-YOLO11n-green.svg)](https://docs.ultralytics.com/)
[![Active Learning V3.0](https://img.shields.io/badge/Pipeline-V3.0_Final-orange.svg)](PIPELINE_TOI_UU_GREENSM_HITL_ACTIVE_LEARNING.md)

Dự án nghiên cứu và phát triển hệ thống **End-to-End Human-in-the-Loop Active Learning** cho bài toán **Nhận diện xe điện GreenSM (Single-Class GreenSM Vehicle Detection)** trong dòng giao thông hỗn hợp tại Việt Nam.

---

## 🎯 1. MỤC TIÊU BÀI TOÁN & GIÁ TRỊ THỰC TIỄN

### 1.1. Mục tiêu Kỹ thuật (Data-Centric AI)
Thay vì liên tục thay đổi kiến trúc mô hình hoặc gán nhãn thủ công hàng ngàn bức ảnh ngẫu nhiên, dự án giải quyết bài toán:
> **Với cùng một ngân sách gán nhãn thủ công cố định, sử dụng quy trình Active Learning 2 Tầng (Two-Tier HITL) để chọn lọc chính xác các bức ảnh mang giá trị nhất cho mô hình YOLO11n, giúp tăng mAP nhanh nhất và giảm tối đa công sức nhân sự.**

### 1.2. Định vị Ứng dụng Thực tế (Enterprise & Commercial Value)
- **Giám sát Mật độ Đội xe GreenSM (Fleet Density & Hotspot Monitoring)**: Đo lường tần suất và mật độ hiện diện của xe taxi/ô tô điện GreenSM tại các điểm nóng giao thông (Sân bay, Trung tâm thương mại, nút giao trọng điểm).
- **Phân tích Thị phần Thương hiệu (OOH Impression Share)**: Đo lường tỉ lệ xuất hiện ngoài trời của xe GreenSM so với các hãng xe công nghệ / taxi truyền thống khác.
- **Giải quyết các Lát cắt Khó (Weak Slices)**: Xe nhỏ cự ly xa ($<32 \times 32\text{px}$), điều kiện thiếu sáng ban đêm (Night slice), che khuất một phần (Occlusion), và nhận diện chính xác tránh nhầm lẫn với ô tô cá nhân màu xanh lục khác.

---

## 🏗️ 2. KIẾN TRÚC PIPELINE TỔNG THỂ (TWO-TIER ACTIVE HITL V3.0)

```text
               RAW GREENSM DATA (Video Streams / Image Batches)
                                      │
                                      ▼
                 [INTRA-VIDEO FASTDEDUP: pHash / Embedding]
            (Lọc frame trùng trong cùng Video — Rule Keep-Most-Vehicles)
                                      │
                                      ▼
                   AUTOMATED DATA PROFILING & BASELINE MEASURE
               (Tagging: Day/Night, Blur, Small Object, Density)
               (Đo Baseline: Time/Image gán tay vs nhãn mồi CVAT)
                                      │
      ┌───────────────────────────────┴───────────────────────────────┐
      ▼                                                               ▼
 FIXED TEST SET                                                UNLABELED POOL
(Video-Aware Split)                                                   │
      │                                                               ▼
 QA & Freeze                                                   SEED SELECTION (~10%)
(Dùng chung V0..Vk)                                           (Stratified Sampling)
      │                                                               │
      │                                                               ▼
      │                                                         DATASET V0
      │                                                               │
      │                                                               ▼
      │                                                       TRAIN MODEL V0 (YOLO11n)
      │                                                               │
      └───────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
                          INFERENCE ON UNLABELED POOL
                                      │
                                      ▼
                       1-CLASS COST-AWARE MINING & SCORING
               (Score = Conf_Uncertainty × Dynamic_Slice_Weight × Anomaly / Effort)
                                      │
                                      ▼
                         DIVERSITY CLUSTERING (K-Means)
                   (Tránh dồn ảnh cùng bối cảnh / góc quay)
                                      │
                                      ▼
                      TWO-TIER HUMAN-IN-THE-LOOP (HITL)
                                      │
      ┌───────────────────────────────┼───────────────────────────────┐
      ▼                               ▼                               ▼
 HIGH CONFIDENCE               BUFFER ZONE (GAP)              LOW/MEDIUM CONFIDENCE
(Conf >= 0.85 &               (0.70 < Conf < 0.85)             (0.20 <= Conf <= 0.70)
 Exclude Night/Small)         (Chuyển sang Tier 2)            + Weak Slice (Night/Small)
      │                               │                               │
      ▼                               └───────────────┬───────────────┘
TIER 1: AUTO-ACCEPT                                   │
(Gán nhãn tự động)                                    ▼
      │                                     TIER 2: CVAT HUMAN REVIEW
      ├────────────────┐                    (Sửa nhãn mồi / Vẽ bổ sung)
      ▼                ▼                              │
[SPOT-CHECK QA]  [PRE-LABEL]                          │
(Random 5-10%)         │                              │
      │                └──────────────┬───────────────┘
      ▼                               │
(Measure Accuracy)                    ▼
                                 DATASET V1 (Lineage Manifest + Git Tag)
                                      │
                                      ▼
                            TRAIN MODEL V1 (YOLO11n)
                                      │
                                      ▼
                  EVALUATION & STATISTICAL SIGNIFICANCE CHECK
                    (Multi-Seed Run: Gain > Natural Variance)
                                      │
                                      ▼
                 A/B TESTING & ABLATION STUDY BENCHMARK
         (Random vs Active vs Active-No-AutoLabel on Fixed Test)
```

---

## 📁 3. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT STRUCTURE)

```text
Buildphase/
├── src/                                  # Toàn bộ Mã nguồn Python Tự động hóa Pipeline
│   ├── __init__.py                       # Package Init
│   ├── data_profiling.py                 # Lọc trùng Intra-Video & Metadata Profiling
│   ├── active_selection.py               # Active Mining 1-Class & Phân tầng 2-Tier HITL
│   ├── cvat_automation.py                # Đấu nối Pre-labels CVAT & Assembly Dataset V1
│   ├── train_yolo11n.py                  # Wrapper Training YOLO11n & Multi-Seed Run
│   ├── evaluate_benchmark.py             # Đánh giá Đa lát cắt & Xuất báo cáo A/B Test
│   ├── colab_runner.py                   # Master Pipeline Execution Script cho Colab
│   └── README.md                         # Hướng dẫn chi tiết mã Cell chạy trên Colab
├── PIPELINE_TOI_UU_GREENSM_HITL_ACTIVE_LEARNING.md  # Tài liệu Thiết kế Pipeline V3.0 Final
├── BAO_CAO_TONG_KET_SELECT_MODEL.md      # Báo cáo chọn mô hình YOLO11n
├── mentorTalk1.md                        # Bảng tổng hợp góp ý Mentor
├── .gitignore                            # Cấu hình GitIgnore chỉ đẩy phần SRC & Markdown
└── README.md                             # Tài liệu tổng quan dự án (File này)
```

---

## 🚀 4. HƯỚNG DẪN SỬ DỤNG VÀ CHẠY TRÊN GOOGLE COLAB

### 4.1. Chạy Tự động Toàn bộ Pipeline (Master Script)
Chỉ cần 2 Cell lệnh trên Notebook Google Colab:

```python
# Cell 1: Mount Google Drive & Cài thư viện
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics imagehash opencv-python-headless scikit-learn pyyaml requests
```

```python
# Cell 2: Chạy Master Execution Script
!python src/colab_runner.py \
    --data-dir /content/drive/MyDrive/GreenSM_Data/raw_images \
    --output-dir /content/drive/MyDrive/GreenSM_Outputs \
    --top-k 200 \
    --epochs 30
```

### 4.2. Chạy Từng Bước Độc lập (Step-by-Step Execution)

```bash
# Bước 1: Intra-Video Deduplication & Profiling
python src/data_profiling.py --data-dir path/to/images --output-json dataset_metadata.json

# Bước 2: 1-Class Active Mining & 2-Tier HITL Split
python src/active_selection.py --metadata-json dataset_metadata.json --top-k 200

# Bước 3: Assembly Dataset V1 & Pre-labels
python src/cvat_automation.py --action assemble --selection-json active_selection_results.json --output-dir dataset_v1

# Bước 4: Train YOLO11n Multi-Seed (3 Seeds)
python src/train_yolo11n.py --data-yaml dataset_v1/dataset.yaml --seeds 42 123 999 --epochs 30

# Bước 5: Đánh giá Benchmark & Sinh Báo cáo Markdown
python src/evaluate_benchmark.py --test-yaml dataset_v1/dataset.yaml --output-report BAO_CAO_BENCHMARK_A_B_TEST.md
```

---

## 📊 5. CHỈ SỐ ROI & HIỆU QUẢ DỰ DỰ KIẾN (TARGET METRICS)

- **Mô hình YOLO11n V1**: Target $mAP_{50-95} \ge 78\%$, Small Object Recall $\ge 90\%$.
- **Tiết kiệm Công sức Nhân sự**: Tiết kiệm **$\ge 65\%$ tổng thời gian gán nhãn** (Thời gian review nhãn mồi CVAT $\approx 12\text{s/ảnh}$ so với $45\text{s/ảnh}$ gán vẽ từ đầu).
- **Tái lập & Thống kê**: Đạt khả năng tái lập 100% qua file `manifest.yaml` và kiểm định mức tăng $\Delta mAP > \sigma_{seed}$ (dao động tự nhiên giữa các seed).

---

## 📜 6. PHÂN CÔNG VAI TRÒ VÀ ĐỘC LẬP QA

- **Active Learning & Training Lead**: Phụ trách viết code pipeline, mining và huấn luyện mô hình.
- **Fixed Test QA Lead (Độc lập)**: Phụ trách kiểm định chất lượng, đóng băng tập Test và ngẫu nhiên Spot-check (5-10%) chất lượng nhãn máy tự duyệt Tier 1.
- **Reproducibility Lead**: Phụ trách quản lý file `manifest.yaml`, git tags và checkpoints.

---
*Dự án thuộc Cohort Build Phase — GreenSM Vehicle Active Learning Lab (M50).*
