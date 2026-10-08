# PIPELINE TỔNG THỂ TỐI ƯU — GREENSM HUMAN-IN-THE-LOOP ACTIVE LEARNING LAB (M50)

> **Dự án**: End-to-End Human-in-the-Loop Annotation Lab — GreenSM Vehicle Detection  
> **Phiên bản**: V3.0 Final (Mentor-Aligned & Enterprise-Ready)  
> **Ngày cập nhật**: 08/10/2026  
> **Mô hình cố định (Locked Model)**: **YOLO11n** (mAP50-95 Target: >78%, Small Object Recall: >90%, Speed: 21.4 FPS, Size: 5.5 MB)

---

## 1. MỤC TIÊU BÀI TOÁN & TẦM NHÌN THỰC TẾ

### 1.1. Bài toán kỹ thuật cốt lõi (Data-Centric AI — 1-Class Object Detection)
Bài toán đặt ra không phải là xây dựng hệ thống MLOps phức tạp hay liên tục thay đổi kiến trúc mô hình. Focus chính là bài toán **Nhận diện xe GreenSM (Single-Class GreenSM Vehicle Detection)** trong dòng giao thông hỗn hợp tại Việt Nam.

Mục tiêu cốt lõi:
> **Với cùng một ngân sách gán nhãn thủ công cố định của con người (Annotation Budget), áp dụng quy trình Active Learning & Two-Tier HITL để chọn lọc đúng các mẫu ảnh có giá trị nhất, giúp mô hình YOLO11n học nhanh nhất, đạt mAP cao nhất và giảm tối đa thời gian gán nhãn.**

### 1.2. Định vị bài toán thực tế (Enterprise & Commercial Value)
Để giải pháp mang giá trị thương mại và thực tiễn cao, hệ thống được ứng dụng vào **Hệ thống Quản lý Đội xe Thông minh & Phân tích Thị phần Giao thông Đô thị (EV Fleet Operations & OOH Analytics)**:
1. **Giám sát Mật độ Đội xe GreenSM (Fleet Density & Hotspot Monitoring)**: Định vị và đo lường tần suất hiện diện của xe taxi/ô tô điện GreenSM tại các nút giao thông trọng điểm, sân bay, trung tâm thương mại theo thời gian thực.
2. **Phân tích Thị phần Thương hiệu Ngoài trời (OOH Impression Share)**: Đo lường tỷ lệ diện mạo thương mại của xe GreenSM so với các hãng xe công nghệ / taxi truyền thống khác trong dòng giao thông đô thị.
3. **Thách thức Kỹ thuật Thực tế**:
   - **Xe nhỏ ở xa (< 32x32px)**: Nhìn thấy từ camera giám sát góc rộng.
   - **Điều kiện thiếu sáng / Ban đêm (Night Slice)**: Màu xanh Cyan đặc trưng của GreenSM bị biến đổi màu dưới ánh đèn đường.
   - **Che khuất một phần (Occlusion)**: Bị chắn bởi xe máy, ô tô khác.
   - **Dương tính giả (False Positives)**: Dễ nhầm lẫn với các ô tô cá nhân màu xanh lục / xanh ngọc khác.

---

## 2. KIẾN TRÚC PIPELINE TỔNG THỂ (TWO-TIER ACTIVE HITL V3.0)

Sơ đồ quy trình được chuẩn hóa theo đúng các chỉ dẫn và góp ý chuyên sâu từ Mentor:

```text
                        RAW GREENSM DATASET (Video Streams / Image Batches)
                                        │
                                        ▼
                   [INTRA-VIDEO FASTDEDUP: pHash / Embedding]
              (Lọc frame trùng trong cùng Video — Ưu tiên giữ frame Nhiều Xe)
                                        │
                                        ▼
                     AUTOMATED DATA PROFILING & BASELINE MEASURE
                 (Tagging: Day/Night, Blur, Small Object, Density)
                 (Đo Baseline: Time/Image gán tay vs nhãn mồi trên CVAT)
                                        │
        ┌───────────────────────────────┴───────────────────────────────┐
        ▼                                                               ▼
 FIXED TEST SET                                                  UNLABELED POOL
(Video-Aware Split)                                                     │
        │                                                               ▼
 Human QA & Freeze                                               SEED SELECTION (~10%)
(Dùng chung V0..Vk)                                             (Stratified Sampling)
        │                                                               │
        │                                                               ▼
        │                                                         HUMAN LABELING
        │                                                               │
        │                                                               ▼
        │                                                         DATASET V0
        │                                                               │
        │                                                               ▼
        │                                                         TRAIN MODEL V0 (YOLO11n)
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
  HIGH CONFIDENCE                 BUFFER ZONE (GAP)              LOW/MEDIUM CONFIDENCE
 (Conf >= 0.85 &                 (0.70 < Conf < 0.85)             (0.20 <= Conf <= 0.70)
  Not Night/Small)               (Chuyển sang Tier 2)            + Weak Slice (Night/Small)
        │                               │                               │
        ▼                               └───────────────┬───────────────┘
 TIER 1: AUTO-ACCEPT                                    │
 (Gán nhãn tự động)                                     ▼
        │                                      TIER 2: CVAT HUMAN REVIEW
        ├────────────────┐                     (Sửa nhãn mồi / Vẽ bổ sung)
        ▼                ▼                              │
 [SPOT-CHECK QA]   [PRE-LABEL]                          │
 (Random 5-10%)          │                              │
        │                └──────────────┬───────────────┘
        ▼                               │
 (Measure Label Acc)                    ▼
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

## 3. CHI TIẾT 10 BƯỚC THỰC THI PIPELINE (MENTOR-ALIGNED)

### Bước 1 — Intra-Video Fast Deduplication (Lọc trùng lặp chuẩn xác)
* **Vấn đề cũ**: Lọc trùng lặp trên toàn bộ pool có thể xoá nhầm ảnh mang góc quay quan trọng hoặc làm sai lệch phân bố.
* **Cải tiến từ Mentor**:
  1. Chỉ thực hiện Deduplication **trong cùng 1 Video (Intra-video deduplication)** dựa trên Perceptual Hash (pHash) hoặc Cosine Similarity của feature vector YOLO backbone.
  2. **Quy tắc giữ lại (Keep-Most-Vehicles)**: Khi 2 frame bị trùng ($HD < 5$), **ưu tiên giữ lại frame có nhiều bbox xe GreenSM hơn** (tránh vô tình loại bỏ các xe nhỏ ở cự ly xa).
  3. Ghi lại tỷ lệ frame trùng thực tế (%) để làm mốc chỉ số Reproducibility.

### Bước 2 — Automated Data Profiling & CVAT Time Baseline
* **Tự động trích xuất Metadata**:
  * **Day / Night**: Giá trị trung bình kênh V (Value) trong HSV ($\text{Brightness} < 65 \rightarrow \text{Night}$).
  * **Blur**: Phương sai toán tử Laplacian ($\text{Variance} < 100 \rightarrow \text{Blur}$).
  * **Small Object**: Diện tích bbox dự đoán $< 32 \times 32 \text{px}$.
  * **Density**: Số lượng xe GreenSM / frame $> 6 \text{ xe} \rightarrow \text{Dense Scene}$.
* **Đo đạc Baseline thực tế (CVAT Measuring)**:
  * Gán thử 50 ảnh thủ công từ đầu (Manual Drawing) $\rightarrow$ Ghi nhận thời gian trung bình $T_{manual}$ (giây/ảnh).
  * Sửa nhãn mồi 50 ảnh (Pre-label Review) $\rightarrow$ Ghi nhận thời gian trung bình $T_{assisted}$ (giây/ảnh).
  * Tỉ lệ $T_{manual} / T_{assisted}$ là mốc baseline đo lường chính xác chỉ số **Time Saved**.

### Bước 3 — Tạo Fixed Test Set (Video-Aware Split & Freeze)
* **Quy tắc phân chia**: Chia dữ liệu theo **Video Sequence ID (Group-aware split)**, tuyệt đối không chia ngẫu nhiên cấp frame để tránh rò rỉ dữ liệu (Data Leakage) giữa Train và Test.
* **Đại diện Slice**: Kiểm tra đảm bảo tập Test chứa đủ các bối cảnh: Ban ngày, Ban đêm, Xe nhỏ xa, Che khuất, Mật độ đông.
* **Freeze Benchmark**: Gán nhãn tỉ mỉ, trải qua bước QA独立 và **khóa cố định (Freeze)**. Mọi mô hình ($V_0, V_1, \dots, V_k$, Random vs Active) đều đánh giá trên duy nhất tập Test này.

### Bước 4 — Chọn Seed Set & Train Model V0
* Chọn ~10% dữ liệu đại diện từ Unlabeled Pool bằng **Stratified Sampling** theo các Tag Metadata ở Bước 2.
* Gán nhãn thủ công 100% $\rightarrow$ Tạo **Dataset V0**.
* Huấn luyện mô hình **YOLO11n (Model V0)** cố định hyperparameters (epoch, lr, batch size).

### Bước 5 — 1-Class Cost-Aware Hard-Case Mining
Chạy Inference Model V0 trên Unlabeled Pool. Vì bài toán là **1-Class Object Detection (GreenSM Vehicle)**, công thức tính điểm ưu tiên được thiết kế lại:

1. **1-Class Confidence Uncertainty ($U_i$)**: Độ bất định đo trực tiếp trên Confidence Score của Bounding Box thay vì Cross-class Entropy:
   $$U_i = 1 - 2 \times |P(\text{GreenSM}) - 0.5|$$
   *(Bbox có confidence tiệm cận 0.5 có độ bất định cao nhất)*. Kết hợp với chỉ số **Flip Stability** (mức chênh lệch bbox khi lật ảnh).
2. **Dynamic Weak-Slice Weight ($W_i$)**: Trọng số cho nhóm xe nhỏ / ban đêm **được lấy tự động từ kết quả đánh giá (Eval) trên Test Set của vòng trước** (không hardcode cố định):
   $$W_i = \frac{1}{\text{Recall}_{slice\_i}(\text{Model } V_{k-1})}$$
3. **Prediction Anomaly ($A_i$)**: Phát hiện bất thường (tỉ lệ khung hình bbox dị biệt, vị trí bất thường).
4. **Estimated Labeling Effort ($E_i$)**: Công sức gán nhãn ước tính dựa trên số lượng bbox dự đoán ($N_{box}$):
   $$E_i = 1 + 0.1 \times N_{box}$$

👉 **Công thức điểm tổng hợp (Cost-Aware Active Score)**:
$$\text{Selection Score}_i = \frac{U_i \times W_i \times A_i}{E_i}$$

### Bước 6 — Diversity & Deduplication Filtering (Clustering)
* Lấy Top $2 \times K$ ảnh có `Selection Score` cao nhất.
* Trích xuất Feature Embedding từ YOLO backbone và áp dụng **K-Means Clustering** để chọn ra $K$ ảnh đại diện đa dạng nhất.
* **Mục tiêu**: Loại bỏ rủi ro chọn phải 30 bức ảnh gần như trùng bối cảnh cùng một góc đường.

### Bước 7 — Two-Tier Human-in-the-Loop & Quality Guardrails
Cơ chế duyệt 2 tầng được bổ sung các chốt chặn an toàn (Guardrails) theo góp ý của Mentor:

```text
                               Candidates Top-K
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
 HIGH CONFIDENCE                BUFFER ZONE                   LOW/MEDIUM CONF CONFIDENCE
 (Conf >= 0.85 &                (0.70 < Conf < 0.85)           (0.20 <= Conf <= 0.70)
  High Flip Stability)          (Loại bỏ vùng bỏ ngỏ)          + Small Object / Night Slice
  AND NOT (Small/Night)               │                             │
        │                             └──────────────┬──────────────┘
        ▼                                            │
 TIER 1: AUTO-ACCEPT PRELABEL                        ▼
 (Lưu thẳng nhãn máy tự động)              TIER 2: CVAT HUMAN REVIEW
        │                                  - Annotator duyệt & sửa nhãn mồi
        ▼                                  - Xóa False Positive / Vẽ bổ sung
 [RANDOM SPOT-CHECK QA]
 (Trích 5-10% kiểm định độc lập)
```

* **Vùng giữa (Buffer Zone $0.70 < Conf < 0.85$)**: Chuyển thẳng sang Tier 2 để không bỏ ngỏ dữ liệu mờ ranh giới.
* **Loại trừ rủi ro (Risk Exclusion)**: Ảnh thuộc tag **Small Object** hoặc **Night** bắt buộc chuyển sang Tier 2 duyệt tay, **không cho phép Auto-accept** vì nguy cơ False Negative cao.
* **Spot-Check QA (5-10%)**: Trích xuất ngẫu nhiên 5-10% ảnh Tier 1 Auto-accept đưa cho QA độc lập kiểm định để tính chỉ số **Machine Label Accuracy**.

### Bước 8 — Experiment Manifest & Data Lineage Tracking
Mỗi vòng Active Learning thu được dữ liệu mới sẽ xuất file `manifest.yaml` và gắn `git tag` tương ứng (ví dụ: `v1.0-active-exp1`):

```yaml
experiment_id: "EXP_GREENSM_V1_20261008"
git_tag: "v1.0-active"
dataset_summary:
  total_images: 500
  seed_images: 300
  active_tier1_auto: 120
  active_tier2_human: 80
selection_strategy:
  uncertainty_type: "1-class_confidence_margin"
  weak_slice_weights: {"night": 1.42, "small_object": 1.65}
  diversity_clustering: "kmeans_k80"
model_reproducibility:
  architecture: "YOLO11n"
  random_seed: 42
  config_file: "configs/yolo11n_greensm.yaml"
  checkpoint_path: "runs/train/v1/weights/best.pt"
```

### Bước 9 — Retrain Model V1 & Statistical Significance Check
* Huấn luyện mô hình **YOLO11n (Model V1)** trên **Dataset V1**.
* **Đo lường mức dao động tự nhiên (Natural Variance)**: Chạy lại thử nghiệm với 3 `random seed` khác nhau (ví dụ: Seed 42, 123, 999) trên cùng cấu hình.
* **Nguyên tắc chấp nhận**: Cải thiện $mAP$ của Active Learning chỉ được công nhận là thật nếu:
  $$\Delta mAP = mAP(V_1) - mAP(V_0) > \sigma_{seed\_variance}$$

### Bước 10 — A/B Testing & Ablation Study Benchmark
Đánh giá sức mạnh của Active Learning qua 3 nhánh độc lập trên cùng **Fixed Test Set**:

1. **Nhánh A (Random Sampling)**: Huấn luyện YOLO11n trên $N$ ảnh chọn ngẫu nhiên (Gán nhãn thủ công 100%).
2. **Nhánh B (Full Active Learning 2-Tier)**: Huấn luyện YOLO11n trên $N$ ảnh chọn qua Active Learning + Tier 1 Auto-accept.
3. **Nhánh C (Ablation — Active Selection Only)**: Huấn luyện YOLO11n trên $N$ ảnh chọn qua Active Mining nhưng **chỉ gán thủ công (không Auto-accept)** nhằm bóc tách riêng hiệu quả của thuật toán chọn ảnh khó vs hiệu quả của nhãn tự duyệt.

---

## 4. NGÂN SÁCH GÁN NHÃN & KẾ HOẠCH KHẢ THI (3 TUẦN)

Để chứng minh đề tài hoàn toàn vừa sức trong thời gian **3 tuần**, ngân sách gán nhãn được lập chi tiết như sau:

### 4.1. Khối lượng công việc gán nhãn thực tế (Workload Estimation)

| Thành phần dữ liệu | Số lượng ảnh | Phương thức gán nhãn | Ước tính thời gian/ảnh | Tổng thời gian (Giờ) |
| :--- | :---: | :--- | :---: | :---: |
| **Fixed Test Set** | 150 ảnh | Gán thủ công 100% (Kỹ lưỡng) | 60 giây | 2.5 giờ |
| **Seed Set (Model V0)** | 250 ảnh | Gán thủ công 100% | 45 giây | 3.1 giờ |
| **Active Cycle V1 (Tier 2)** | 120 ảnh | Review & Sửa nhãn mồi CVAT | 15 giây | 0.5 giờ |
| **Active Cycle V1 (Tier 1)** | 180 ảnh | Auto-Accept (Máy duyệt) | 0 giây | 0.0 giờ |
| **Spot-Check QA Tier 1** | 20 ảnh | QA Độc lập kiểm thử | 15 giây | 0.1 giờ |
| **A/B Test Random Set** | 120 ảnh | Gán thủ công 100% | 45 giây | 1.5 giờ |
| **TỔNG CỘNG** | **840 ảnh** | *(Thực tế người làm: ~660 ảnh)* | **Trung bình** | **~7.7 giờ công** |

👉 **Kết luận Khả thi**: Tổng ngân sách gán nhãn cho cả dự án chỉ tốn khoảng **7.7 - 9.0 giờ công con người**, hoàn toàn khả thi để thực hiện tỉ mỉ trong 3 tuần mà không lo vỡ kế hoạch.

### 4.2. Lịch trình thực thi chi tiết (3 Tuần: 08/10 - 29/10/2026)

```text
Tuần 1 (08/10 - 14/10): Data Profiling, Baseline Measurement & Fixed Test Split
Tuần 2 (15/10 - 21/10): Seed Training (V0), 1-Class Mining & Two-Tier HITL Cycle (V1)
Tuần 3 (22/10 - 29/10): Multi-Seed Run, A/B & Ablation Benchmark, Complete Report
```

* **Tuần 1 (08/10 - 14/10)**:
  * Thu thập dữ liệu video xe GreenSM, chạy Intra-video Deduplication (Keep-most-vehicles).
  * Đo đạc baseline thời gian gán CVAT ($T_{manual}$ vs $T_{assisted}$).
  * Chia **Fixed Test Set** theo Video Sequence ID, QA & Freeze Test.
* **Tuần 2 (15/10 - 21/10)**:
  * Gán nhãn Seed set 250 ảnh $\rightarrow$ Train **Model V0 (YOLO11n)**.
  * Inference trên Unlabeled Pool, chạy 1-Class Confidence Margin Scoring & Diversity Clustering.
  * Thực hiện **Two-Tier HITL** (Tier 1 Auto + Tier 2 CVAT + Spot-check QA).
  * Xuất `manifest.yaml` $\rightarrow$ Train **Model V1 (YOLO11n)**.
* **Tuần 3 (22/10 - 29/10)**:
  * Chạy Multi-seed runs (3 seeds) để xác định mức dao động tự nhiên.
  * Thực hiện A/B Test (Random vs Active) và Ablation Study.
  * Hoàn thiện báo cáo ROI, biểu đồ mAP gain và time saved.

---

## 5. PHÂN CÔNG NHÂN SỰ & ĐỘC LẬP QA

Để đảm bảo tính độc lập và độ tin cậy khoa học cao nhất theo gợi ý của Mentor, trách nhiệm được phân công rõ ràng:

1. **Kỹ sư Mô hình & Active Learning Pipeline Lead**:
   - Chịu trách nhiệm viết code Profiling, Mining Score, Deduplication và Training YOLO11n.
   - Quản lý quá trình Inference và chuyển dữ liệu sang CVAT.
2. **Kỹ sư Kiểm định Chất lượng Tập Test & QA Lead (Tách biệt độc lập)**:
   - **Độc lập hoàn toàn với người train mô hình**.
   - Chịu trách nhiệm gán nhãn, kiểm tra chất lượng và khóa cố định **Fixed Test Set**.
   - Chịu trách nhiệm ngẫu nhiên Spot-check 5-10% ảnh thuộc Tier 1 Auto-accept để tính chỉ số `Machine Label Accuracy`.
3. **Kỹ sư Quản lý Tái lập & Manifest Lead**:
   - Quản lý các file `manifest.yaml`, lưu trữ config, seed, git tags và model checkpoints.
   - Đảm bảo 100% thí nghiệm có thể tái lập lại khi Mentor hoặc Hội đồng kiểm tra.

---

## 6. MA TRẬN CHỈ SỐ ĐO LƯỜNG HIỆU QUẢ (ROI & METRICS MATRIX)

Báo cáo kết quả cuối cùng sẽ được lập dựa trên **3 nhóm metric cốt lõi**:

### 6.1. Chỉ số Chất lượng Mô hình (Model Quality Gain)
* $\Delta mAP_{50-95} = mAP(\text{Model } V_1) - mAP(\text{Model } V_0)$
* $\Delta Recall_{small} = Recall_{small}(\text{Model } V_1) - Recall_{small}(\text{Model } V_0)$
* $\Delta Recall_{night} = Recall_{night}(\text{Model } V_1) - Recall_{night}(\text{Model } V_0)$
* **Target**: $mAP_{50-95}$ tăng **+4% đến +8%** so với Model V0.

### 6.2. Chỉ số Tiết kiệm Công sức & Chất lượng Nhãn (Labor Efficiency & Safety)
* **Thời gian trung bình/ảnh (Avg Sec/Img)**:
  $$\text{Time Saved \%} = 100\% \times \left(1 - \frac{T_{assisted}}{T_{manual}}\right)$$
* **Tỷ lệ giảm giờ công tổng thể (Total Hours Saved %)**: Tiết kiệm $\ge 60\%$ tổng giờ công so với gán nhãn thủ công toàn bộ.
* ** Machine Label Accuracy (Spot-check QA)**: Đạt $\ge 95\%$ chính xác trên các mẫu Tier 1 Auto-accept.

### 6.3. Chỉ số Ý nghĩa Thống kê & A/B Benchmark (Statistical Significance)
* **Natural Variance ($\sigma_{seed}$)**: Mức dao động $mAP$ giữa 3 random seeds.
* **Chỉ số cải thiện thực sự**: $\Delta mAP_{Active} > \sigma_{seed}$.
* **Active vs Random Efficiency Gap**:
  $$\text{Efficiency Gap} = mAP(\text{Active V1}) - mAP(\text{Random V1})$$

---

## 7. DANH SÁCH FILE VÀ ARTIFACTS CẦN TẠO TRONG PROJECT

1. `data_profiling.py`: Script tự động Intra-video Deduplication (pHash, Keep-most-vehicles) và gán tag metadata (Day/Night, Blur, Small Object).
2. `active_selection.py`: Script tính toán điểm 1-Class Confidence Margin Uncertainty, Dynamic Weak-slice Weight, và Diversity Clustering.
3. `train_yolo11n.py`: Pipeline huấn luyện YOLO11n với cấu hình hyperparameter cố định và Multi-seed support.
4. `evaluate_benchmark.py`: Script đánh giá mô hình trên Fixed Test Set theo từng slice và tính toán mức dao động $mAP$.
5. `manifest.yaml`: Template quản lý lineage dữ liệu và tái lập thí nghiệm.
6. `BAO_CAO_BENCHMARK_A_B_TEST.md`: Báo cáo kết quả A/B Test & Ablation Study kèm chứng minh ROI.
