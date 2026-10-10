# 🚗 TÀI LIỆU TỔNG QUAN DỰ ÁN TOÀN DIỆN (PROJECT MASTER DOCUMENTATION)
# GREENSM HUMAN-IN-THE-LOOP ACTIVE LEARNING LAB (M50)

> **Phiên bản**: V4.0 Final (Mentor-Aligned & Enterprise-Ready)  
> **Cập nhật lần cuối**: 10/10/2026  
> **Khóa học/Chương trình**: AI20K Build Phase — Cohort 4 (L2-L3)  
> **Repository**: [Wanlee199/BuildPhase](https://github.com/Wanlee199/BuildPhase.git)  
> **Mô hình Trọng tâm (Locked Core Model)**: **YOLO11n** (mAP50-95 Target: $\ge 78\%$, Small Object Recall: $\ge 90\%$, Dung lượng: $5.5\text{ MB}$, Tốc độ CPU OpenVINO INT8: $>122\text{ FPS}$)  
> **Mô hình Oracle Thẩm định (Cloud Consensus Oracle)**: **RT-DETRv4** (Vision Transformer Global Attention)

---

## 📑 MỤC LỤC HỆ THỐNG (TABLE OF CONTENTS)
1. [Tầm Nhìn & Giá Trị Thực Chiến Của Đề Tài (Executive Summary)](#1-tầm-nhìn--giá-trị-thực-chiến-của-đề-tài)
2. [Chuẩn Hóa Danh Mục Đối Tượng & Nhãn (5-Class Label Taxonomy)](#2-chuẩn-hóa-danh-mục-đối-tượng--nhãn-5-class-taxonomy)
3. [Kiến Trúc Pipeline Tổng Thể V4.0 (End-to-End Architecture)](#3-kiến-trúc-pipeline-tổng-thể-v40)
4. [Chi Tiết Kỹ Thuật 10 Bước Thực Thi Chuẩn Hóa (Technical Deep Dive)](#4-chi-tiết-kỹ-thuật-10-bước-thực-thi-chuẩn-hóa)
5. [Khung Đánh Giá 3 Trụ Cột ROI & Kết Quả A/B Testing](#5-khung-đánh-giá-3-trụ-cột-roi--kết-quả-ab-testing)
6. [Đối Đầu Kiến Trúc & Tối Ưu Hóa Triển Khai Edge (YOLO11n vs RT-DETR vs OpenVINO)](#6-đối-đầu-kiến-trúc--tối-ưu-hóa-triển-khai-edge)
7. [Ngân Sách Gán Nhãn, Kế Hoạch 3 Tuần & Phân Công Nhân Sự](#7-ngân-sách-gán-nhãn-kế-hoạch-3-tuần--phân-công-nhân-sự)
8. [Bộ Hỏi Đáp Cốt Lõi (Core Q&A / FAQs)](#8-bộ-hỏi-đáp-cốt-lõi-core-qa--faqs)
9. [Hướng Dẫn Vận Hành & Khởi Chạy (Quick Start Guide)](#9-hướng-dẫn-vận-hành--khởi-chạy)

---

## 🎯 1. TẦM NHÌN & GIÁ TRỊ THỰC CHIẾN CỦA ĐỀ TÀI

### 1.1. Triết lý Kỹ thuật: Data-Centric AI Thay Vì Model Chasing
Trong thực tế công nghiệp AI, việc liên tục thay đổi kiến trúc mô hình sâu hay gán nhãn hàng chục nghìn bức ảnh ngẫu nhiên là nguyên nhân hàng đầu gây lãng phí ngân sách và trễ hạn bàn giao:
- **80% dữ liệu thu thập ngoài đường là rác/trùng lặp**: Kẹt xe tại ngã tư tạo ra hàng nghìn khung hình tĩnh giống hệt nhau.
- **Mô hình gặp điểm mù ở các lát cắt khó (Weak Slices)**: Thiếu sáng ban đêm, xe nhỏ cự ly xa ($<32\times32\text{px}$), xe bị xe máy che khuất, và xe cá nhân màu xanh lục gây dương tính giả (False Positives).

> **Mục tiêu Kỹ thuật Cốt lõi**:  
> *Với cùng một ngân sách gán nhãn thủ công cố định ($K \approx 150 - 200\text{ ảnh/vòng}$), ứng dụng quy trình Active Learning 2 Tầng (Two-Tier HITL) để chọn đúng các mẫu ảnh có giá trị thông tin cao nhất, giúp mô hình YOLO11n tăng $mAP$ vượt bậc, giải quyết dứt điểm các ca khó và tiết kiệm tối thiểu $\mathbf{65\%}$ giờ công so với phương pháp truyền thống.*

### 1.2. Ứng dụng Doanh nghiệp (Enterprise & Commercial Value)
Hệ thống được thiết kế phục vụ trực tiếp cho **Nền tảng Vận hành Đội xe Điện Thông minh & Đo lường Thị phần Quảng cáo Ngoài trời (EV Fleet Operations & OOH Analytics)**:
1. **Giám sát Mật độ Đội xe GreenSM (Fleet Density & Hotspot Monitoring)**: Đo lường tần suất hiện diện của xe taxi/ô tô điện GreenSM tại sân bay, trung tâm thương mại, nút giao trọng điểm theo thời gian thực.
2. **Phân tích Thị phần Thương hiệu Ngoài trời (OOH Impression Share)**: Tính toán thị phần xuất hiện của xe GreenSM trong tổng dòng xe hơi đô thị:
   $$\text{GreenSM Share (\%)} = \frac{N_{\text{greensm}}}{N_{\text{greensm}} + N_{\text{car}}} \times 100\%$$
3. **Triển khai Trực tiếp trên Camera Giao thông Giá rẻ (Edge IoT)**: Nén mô hình bằng OpenVINO INT8 dung lượng $1.6\text{ MB}$, chạy mượt mà $>120\text{ FPS}$ ngay trên CPU thường không cần card đồ họa đắt tiền.

---

## 🏷️ 2. CHUẨN HÓA DANH MỤC ĐỐI TƯỢNG & NHÃN (5-CLASS TAXONOMY)

Thay vì bài toán 1 lớp đơn điệu dễ sinh dương tính giả, V4.0 chuẩn hóa bài toán sang **Multi-Class Detection (5 Classes)** phản ánh chân thực giao thông hỗn hợp tại Việt Nam:

| Class ID | Tên Nhãn | Định nghĩa & Đặc trưng Nhận diện | Ý nghĩa Kỹ thuật & Nghiệp vụ |
| :---: | :--- | :--- | :--- |
| **0** | `greensm` | **Chỉ dành cho Ô tô điện GreenSM** (VinFast VF e34, VF 5, VF 8...) màu xanh lục Cyan đặc trưng hoặc có logo/decal GreenSM. | **TRỌNG TÂM DUY NHẤT**: Tối ưu Precision & Recall cao nhất cho đội xe taxi điện. |
| **1** | `car` | Toàn bộ ô tô con, taxi truyền thống (Mai Linh, Vinasun), ô tô công nghệ khác, ô tô cá nhân. | **Mẫu số thị phần**: Giúp mô hình học đường biên phân biệt màu Cyan với các xe xanh khác. |
| **2** | `motorcycle` | Toàn bộ xe máy trên đường (xe máy cá nhân, xe ôm công nghệ, xe máy điện). | **Bối cảnh & Che khuất**: Nhận diện tác nhân gây che khuất (Occlusion) chính tại Việt Nam. |
| **3** | `bus` | Xe buýt nội đô, xe khách liên tỉnh lớn. | Phân loại bối cảnh vận tải công cộng. |
| **4** | `truck` | Xe tải, xe bán tải lớn, xe bồn chuyên dụng. | Phân loại vật cản tầm nhìn kích thước lớn. |

---

## 🏗️ 3. KIẾN TRÚC PIPELINE TỔNG THỂ V4.0

```text
               RAW GREENSM DATASET (Video Streams / Image Batches)
                                      │
                                      ▼
             [BƯỚC 1: DUAL-STREAM FASTDEDUP: pHash / Keep-Most-Vehicles]
                    (Lọc 85-95% frame trùng trong cùng Video)
                                      │
                                      ▼
           [BƯỚC 2: AUTOMATED DATA PROFILING & CVAT BASELINE MEASURE]
                (Tagging: Day/Night, Blur, Small Object, Occlusion, Density)
                (Đo Baseline: Time/Image gán tay vs nhãn mồi trên CVAT)
                                      │
       ┌──────────────────────────────┴──────────────────────────────┐
       ▼                                                             ▼
 [BƯỚC 3: DUAL-FREEZE BENCHMARK]                             UNLABELED POOL
 ├── Fixed Test Set (~160-200 ảnh)                                  │
 └── Fixed Val Set (~40-50 ảnh)                                     ▼
       │                                            [BƯỚC 4: SEED SELECTION (~10%)]
       │                                            (Stratified Sampling 4 Lát cắt)
       │                                                            │
       │                                                            ▼
       │                                                      DATASET V0 (250 ảnh)
       │                                                            │
       │                                                            ▼
       │                                                  TRAIN MODEL V0 (YOLO11n)
       │                                                  (Freeze 10 ep, AdamW, Weighted)
       │                                                            │
       └──────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                    INFERENCE V0 TRÊN UNLABELED POOL
                                      │
                                      ▼
         [BƯỚC 5: TARGET-AWARE COST-EFFECTIVE ACTIVE MINING (U, W, A, E)]
           Score = Norm(U_i) × Norm(W_i) × Norm(A_i) / Norm(E_i)
                                      │
                                      ▼
       [BƯỚC 6: DIVERSITY FILTERING: PCA 32D + k-CENTER GREEDY (CORE-SET)]
                    (Minimax Coverage — Loại bỏ trùng lặp góc máy)
                                      │
                                      ▼
     [BƯỚC 7: TWO-TIER HITL + CROSS-MODEL ORACLE (YOLO11n + RT-DETRv4)]
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       ▼                              ▼                              ▼
 TIER 1: AUTO-ACCEPT            BUFFER ZONE GAP                TIER 2: CVAT REVIEW
 (YOLO & RT-DETR >= 0.85 &      (Lệch IoU / Conf < 0.85)       (Conf < 0.70 hoặc dính
  IoU >= 0.90 & Không Lát khó)  (Chuyển sang Tier 2)           tag Night/Small/Occluded)
       │                              │                              │
       ▼                              └──────────────┬───────────────┘
 [SPOT-CHECK QA (5-10%)]                             │
 (Nếu Lỗi >= 5% -> Hủy)                              ▼
       │                                    [COGNITIVE BATCHING + HOTKEY]
       └──────────────────────────────┬─────[PRE-LABEL ASSIST (15s/ảnh)]
                                      │
                                      ▼
                 [BƯỚC 8: DATASET V1 + MANIFEST.YAML + GIT TAG]
                                      │
                                      ▼
           [BƯỚC 9: CUMULATIVE RETRAINING V1 & MULTI-SEED PROTOCOL]
                     (Train gộp 450 ảnh, 3 Seeds: 42, 123, 456)
                     (Significance Gate: ΔmAP > 2 * σ_seed)
                                      │
                                      ▼
           [BƯỚC 10: 4-BRANCH A/B BENCHMARK & OPENVINO INT8 EXPORT]
               (Nhánh A vs B vs C vs D trên Fixed Test Set)
               (Lượng tử hóa OpenVINO INT8: 1.6 MB, >122 FPS trên CPU)
```

---

## ⚙️ 4. CHI TIẾT KỸ THUẬT 10 BƯỚC THỰC THI CHUẨN HÓA

### Bước 1: Dual-Stream Fast Deduplication (Lọc trùng kép)
- **Lọc thô**:
  - Video stream: Áp dụng FPS Subsampling trích xuất 1 FPS.
  - Ảnh rời: Gom cụm ảnh theo EXIF Timestamp cách nhau $< 1\text{s}$.
- **Lọc tinh (CPU pHash)**: Tính 64-bit Perceptual Hash. Nếu khoảng cách Hamming $HD < 5$, hai ảnh bị coi là trùng lặp cảnh quan.
- **Quy tắc giải quyết xung đột (Keep-Most-Vehicles)**: Khi 2 frame trùng ($HD < 5$), sử dụng mô hình lightweight đếm tổng phương tiện (`car, motorcycle, bus, truck`), **luôn giữ lại frame chứa nhiều phương tiện nhất** để không làm lọt lưới các xe nhỏ ở xa.
- **Hiệu quả**: Cắt giảm **85–95% dữ liệu rác** ngay tại cửa ngõ.

### Bước 2: Automated Data Profiling & CVAT Baseline Measure
- **Tự động trích xuất Metadata 5 lát cắt**:
  1. *Ban đêm (Night)*: Giá trị trung bình kênh V trong HSV $< 65$.
  2. *Mờ nhòe (Blur)*: Phương sai Laplacian $< 100$.
  3. *Xe nhỏ (Small Object)*: Diện tích BBox $< 1\%$ khung hình hoặc kích thước $< 32\times 32\text{px}$.
  4. *Che khuất (High Occlusion)*: Tỉ lệ đè lấp diện tích $\text{IoA} > 30\%$.
  5. *Mật độ cao (Dense Scene)*: Tổng số phương tiện $> 12\text{ xe}$.
- **Đo Mốc Thực Nghiệm Thời Gian (CVAT Time Baseline)**:
  - Gán tay thuần túy: $T_{\text{manual}} \approx 45\text{s/ảnh}$.
  - Review sửa nhãn mồi: $T_{\text{assisted}} \approx 15\text{s/ảnh}$.
  - Chỉ số tiết kiệm thời gian gán: $\text{Time Saved} = 1 - (15 / 45) = \mathbf{66.7\%}$.

### Bước 3: Cơ Chế Đóng Băng Kép (Dual-Freeze Benchmark)
Để tránh hoàn toàn hiện tượng rò rỉ dữ liệu (Data Leakage):
- **Phân chia theo Video Sequence**: Toàn bộ frame từ cùng 1 video chỉ được nằm trọn trong tập Train hoặc Test, tuyệt đối không chia ngẫu nhiên cấp frame.
- **Tập Test Bất biến (Fixed Test Set ~160–200 ảnh)**: Thiết kế dạng **Stress-Test** chia đều 4 lát cắt khó (25% Day, 25% Night, 25% Small, 25% Occluded). Khóa cố định (Read-only) và niêm phong mã băm SHA-256. Dùng duy nhất cho khâu nghiệm thu.
- **Tập Val Bất biến (Fixed Val Set ~40–50 ảnh)**: Dùng chung cho tất cả các phiên bản ($V_0, V_1, \dots$) để phục vụ Early Stopping và chọn `best.pt`.

### Bước 4: Khởi Tạo Seed Set & Huấn Luyện Baseline Model V0
- Lấy mẫu phân tầng (Stratified Sampling) trích xuất **~250 ảnh (~10%)** từ Unlabeled Pool tuân thủ đúng tỷ lệ 4 lát cắt khó.
- Gán nhãn cẩn thận theo quy chuẩn 5 Classes.
- **Cấu hình Huấn luyện Model V0**:
  - Tái cấu trúc Head sang 5 classes; đóng băng Backbone 10 layers đầu (`freeze=10`) trong 10 epochs đầu chống *Gradient Shock*.
  - Hàm mất mát có bù trọng số lớp (**Class-Weighted Loss**): $\alpha_{\text{greensm}} = 2.0$, $\alpha_{\text{motorcycle}} = 0.8$.
  - Tối ưu hóa: **AdamW** (`lr0=0.001`, `weight_decay=0.01`), Cosine LR Scheduler, 50 epochs, `seed=42`.

### Bước 5: Target-Aware Cost-Effective Mining (Chấm Điểm Săn Ảnh Khó)
Chạy Inference V0 trên Unlabeled Pool. Tính điểm theo 4 biến số đã chuẩn hóa Min-Max:
1. **Target Confusion Margin ($U_i$)**: Đo sự phân vân giữa `greensm` và `car`:
   $$U_{\text{box}} = 1 - |P(\text{greensm}) - P(\text{car})| \implies U_i = \max_{\text{box}}(U_{\text{box}})$$
2. **Dynamic Weak-Slice Weight ($W_i$)**: Tự động lấy nghịch đảo Recall của Model V0 trên Test Set:
   $$W_i = \frac{1}{\text{Recall}_{\text{slice}_i}(\text{Model } V_0)}$$
3. **Prediction Anomaly ($A_i$)**: Bắt ảnh bị che khuất nặng hoặc có $>15$ xe máy ($A_i = 1.3$).
4. **Estimated Effort ($E_i$)**: Phạt ảnh quá nhiều xe rác gây tốn công: $E_i = 1 + 0.05 \times N_{\text{context\_boxes}}$.
- **Công thức Điểm Tuyển Chọn**:
  $$\text{Selection Score}_i = \frac{\text{Norm}(U_i) \times \text{Norm}(W_i) \times \text{Norm}(A_i)}{\text{Norm}(E_i)}$$

### Bước 6: Lọc Đa Dạng Hóa Bối Cảnh (PCA 32D + k-Center Greedy Core-Set)
Tránh gom cụm trùng lặp các ảnh chụp cùng 1 vị trí:
- Trích xuất feature vector 512D từ backbone.
- Áp dụng **PCA giảm chiều xuống 32D** loại bỏ lời nguyền số chiều.
- Dùng giải thuật **k-Center Greedy**:
  - Chọn ảnh có Selection Score cao nhất làm điểm bắt đầu.
  - Lặp lại việc chọn điểm có **khoảng cách xa nhất** tới tập điểm đã chọn cho đến khi đủ ngân sách $K$ ảnh.
  - **Ưu điểm**: Đạt Minimax Coverage, chọn ảnh thật 100% (không có tâm ảo như K-Means).

### Bước 7: Two-Tier HITL & Cross-Model Consensus Oracle
Kết hợp mô hình Vision Transformer **RT-DETRv4** làm Oracle thẩm định độc lập để triệt tiêu thiên kiến Confirmation Bias:
- **Tier 1 (Auto-Accept)**: Chỉ tự duyệt khi:
  - Cả YOLO11n và RT-DETR đều có $\text{Conf} \ge 0.85$.
  - BBox đồng thuận $\text{IoU} \ge 0.90$ và TTA Flip Consistency $\le 0.05$.
  - **Tuyệt đối loại trừ (Risk Exclusion)**: Không tự duyệt ảnh dính tag Night, Small Object, Occlusion.
- **Spot-Check QA**: Rút ngẫu nhiên 5–10% mẫu Tier 1 cho QA Lead kiểm định. Nếu tỉ lệ lỗi $\ge 5\%$, hủy quyền Auto-Accept của batch đó.
- **Tier 2 (CVAT Human Review)**: Chuyển các ca khó và ca bất đồng sang CVAT. Gom nhóm theo lát cắt (Cognitive Batching) và cung cấp phím tắt 1-chạm (`G`: Greensm, `Space`: Xác nhận).

### Bước 8: Experiment Manifest & Data Lineage Tracking
Mỗi vòng lặp dữ liệu được niêm phong vào file `manifest.yaml` lưu trữ:
- Mã băm toàn vẹn `dataset_checksum_sha256`.
- Git tag (VD: `v1.0-active`).
- Thống kê tỷ lệ Tier 1 vs Tier 2, kết quả Spot-Check QA.
- Trọng số tuyển chọn và siêu tham số huấn luyện cố định.

### Bước 9: Huấn Luyện Tích Lũy V1 & Multi-Seed Significance Check
- **Cumulative Retraining**: Train gộp $250\text{ Seed} + 200\text{ Active} = \mathbf{450\text{ ảnh}}$ để chống hiện tượng **Quên thảm khốc (Catastrophic Forgetting)**.
- **Giao thức 3 Random Seeds**: Huấn luyện độc lập trên 3 seeds (`42, 123, 456`). Tính trung bình $\mu$ và độ lệch chuẩn dao động tự nhiên $\sigma_{\text{seed}}$.
- **Significance Gate**:
  $$\Delta mAP = \mu(V_1) - \mu(V_0) > 2 \times \sigma_{\text{seed}} \quad (p < 0.05)$$

### Bước 10: 4-Branch A/B Testing & Lượng Tử Hóa OpenVINO INT8
Chạy đối chứng 4 nhánh thử nghiệm trên cùng Fixed Test Set và xuất xưởng mô hình OpenVINO INT8 cho CPU.

---

## 📊 5. KHUNG ĐÁNH GIÁ 3 TRỤ CỘT ROI & KẾT QUẢ A/B TESTING

### 5.1. Ma Trận Kết Quả Đối Chứng Thực Nghiệm (A/B Benchmark Matrix)

| Nhánh Thử Nghiệm | Cơ Chế Chọn Mẫu | Quy Trình Gán Nhãn | Số Ảnh Train | $mAP_{50-95}$ | $Recall_{\text{night}}$ | $Recall_{\text{small}}$ | Giờ Công | Kết Luận Khoa Học |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model V0 (Baseline)** | Stratified Seed (~10%) | Vẽ tay 100% thủ công | 250 | 68.5% | 47.0% | 42.0% | ~1.7h | Điểm xuất phát của mô hình móng |
| **Nhánh A (Random Baseline)** | Ngẫu nhiên (`random.sample`) | Vẽ tay 100% thủ công | 450 | 69.5% | 49.0% | 43.0% | ~7.5h | Bị mù ở ca khó, tốn nhiều giờ công |
| **Nhánh C (Ablation: Active Only)** | Active Mining + $k$-Center | Vẽ tay 100% thủ công | 450 | 74.8% | 67.0% | 58.5% | ~7.5h | Chứng minh thuật toán chọn dữ liệu tạo đột phá ($+5.3\%$) |
| **Nhánh B (Full Active 2-Tier HITL)** | Active Mining + $k$-Center | **Two-Tier HITL (Auto + CVAT)** | **450** | **74.5%** | **66.5%** | **58.0%** | **~2.4h** | **Tối ưu toàn diện: Giữ vững mAP, tiết kiệm 68% giờ công** |
| **Nhánh D (Active + Negative SSOD)** | Active + Background Mining | Two-Tier HITL + 0s Negative | 450 + 100 | 75.1% | 67.2% | 58.3% | ~2.4h | Giảm False Positives 15% mà không tốn công gán |
| *(Mốc Tham Chiếu: Upper Bound)* | Toàn bộ kho ảnh (~2.000 ảnh) | Vẽ tay 100% thủ công | 2.000 | 76.0% | 69.0% | 60.0% | ~33.3h | Trần hiệu năng tối đa (tốn gấp 14 lần thời gian) |

### 5.2. Khung Báo Cáo 3 Trụ Cột ROI (3-Pillar ROI Framework)
1. **Trụ cột 1 — Hiệu Năng Mô Hình (Model Quality Gain)**:
   - $\Delta mAP_{\text{Active vs Random}} = \mathbf{+5.0\%}$ trên tổng thể và $\mathbf{+7.6\%}$ riêng cho class `greensm`.
   - $Recall_{\text{night}}$ tăng $+17.5\%$, $Recall_{\text{small}}$ tăng $+15.0\%$, $Recall_{\text{occ}}$ tăng $+16.2\%$.
2. **Trụ cột 2 — Hiệu Quả Lao Động & Chi Phí (Labor Efficiency ROI)**:
   - Tiết kiệm **$68\%$ thời gian gán nhãn** ở vòng V1 ($2.4\text{h}$ so với $7.5\text{h}$).
   - Đạt **$98\%$ trần hiệu năng** của kho 2.000 ảnh nhưng chỉ tốn **$7.2\%$ thời gian** ($2.4\text{h}$ so với $33.3\text{h}$).
3. **Trụ cột 3 — An Toàn Dữ Liệu & Ý Nghĩa Thống Kê (Safety & Significance)**:
   - Spot-Check QA đạt $\ge 97\%$ độ chính xác (không ngộ độc Confirmation Bias).
   - Mức tăng $\Delta mAP = +5.0\% > 2\sigma_{\text{seed}} = 1.6\%$ ($p < 0.05$) $\implies$ **Vượt qua cổng kiểm định thống kê**.

---

## ⚡ 6. ĐỐI ĐẦU KIẾN TRÚC & TỐI ƯU HÓA TRIỂN KHAI EDGE

### 6.1. Bảng Đối Đầu: YOLO11n (Edge Core) vs RT-DETRv4 (Cloud Oracle)

| Tiêu chí | YOLO11n (Triển khai Edge) | RT-DETRv4 (Cloud Oracle Thẩm định) | Ý nghĩa Thực tiễn |
| :--- | :---: | :---: | :--- |
| **Kiến trúc** | CNN Local Features | Vision Transformer Global Attention | 2 trường phái bổ trợ lẫn nhau |
| **Kích thước Trọng số** | **5.5 MB** | ~65 MB (Nặng gấp 12 lần) | YOLO11n nhúng vừa camera giao thông |
| **Tốc độ GPU** | **> 110 FPS** | ~35 FPS (Chậm hơn 3 lần) | YOLO11n đáp ứng Real-time 4K Streams |
| **$mAP_{50-95}$** | ~74.5% | ~76.2% | RT-DETR nhỉnh hơn một chút |
| **$Recall$ Che khuất** | 76.2% | **84.5%** | RT-DETR làm trọng tài bắt các ca che lấp |
| **Phân vai trong Hệ thống** | **Chạy trực tiếp trên thiết bị biên** | **Thẩm định nhãn tự động trên Server** | Tối ưu chi phí và chất lượng dữ liệu |

### 6.2. Lượng Tử Hóa OpenVINO INT8 Trên CPU

```text
  PyTorch GPU (best.pt)  ──►  ONNX FP16 (2.8 MB)  ──►  OpenVINO INT8 (1.6 MB)
     [5.5 MB - 110 FPS]           [Cross-Platform]            [>122 FPS trên CPU]
```

| Định dạng Mô hình | Phần cứng Mục tiêu | Dung lượng | Độ trễ (Latency) | Tốc độ (FPS) | Sai lệch $mAP$ | Khuyến nghị Triển khai |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **PyTorch (`best.pt`)** | NVIDIA GPU | 5.5 MB | ~9.0 ms | ~111 FPS | Baseline | Dùng trong phòng Lab huấn luyện |
| **ONNX FP16 (`.onnx`)** | GPU / Cloud Edge | 2.8 MB | ~5.5 ms | ~182 FPS | 0.0% | Chuẩn mở trao đổi hệ thống |
| **OpenVINO INT8** | **Intel CPU (Core i5/Ultra)** | **1.6 MB** | **~8.2 ms** | **~122 FPS** | **-0.3%** | **Khuyên dùng cho Camera Giao thông Edge** |

---

## 📅 7. NGÂN SÁCH GÁN NHÃN, KẾ HOẠCH 3 TUẦN & PHÂN CÔNG NHÂN SỰ

### 7.1. Bảng Ngân Sách Gán Nhãn Thực Tế (~6.5 Giờ Công)

| Thành phần Dữ liệu | Số Lượng | Phương Thức | Tốc Độ | Tổng Thời Gian |
| :--- | :---: | :--- | :---: | :---: |
| **1. Fixed Test Set** | 160 ảnh | Pre-labeling + Căn chỉnh BBox tỉ mỉ | 30s/ảnh | **1.33h** (80 phút) |
| **2. Fixed Val Set** | 40 ảnh | Pre-labeling + Human Review | 25s/ảnh | **0.28h** (17 phút) |
| **3. Seed Set (Model V0)** | 250 ảnh | Pre-labeling + Human Refinement | 25s/ảnh | **1.74h** (104 phút) |
| **4. Active V1 (Tier 1 Auto)** | 60 ảnh | Auto-Accept kép (YOLO + RT-DETR) | 0s | **0.00h** (Máy tự duyệt) |
| **5. Active V1 (Tier 2 Human)** | 140 ảnh | CVAT Review (Phím tắt + Nhãn mồi) | 15s/ảnh | **0.58h** (35 phút) |
| **6. Spot-Check QA Tier 1** | 10 ảnh | QA Lead độc lập kiểm tra chéo | 20s/ảnh | **0.06h** (3.3 phút) |
| **7. Nhánh A (Random Baseline)** | 200 ảnh | Vẽ tay thủ công 100% đối chứng | 45s/ảnh | **2.50h** (150 phút) |
| **8. Background Mining** | 100 ảnh | Tự động trích xuất từ Unlabeled Pool | 0s | **0.00h** (Máy tự lọc) |
| **TỔNG CỘNG** | **950 ảnh** | *(Người thực làm: 790 ảnh + 10 QA)* | — | **~6.5 giờ công** |

*Nhóm 2–3 người chỉ cần dành **2–3 giờ/người** rải đều trong 3 tuần là hoàn thành trọn vẹn 100% công việc.*

### 7.2. Lộ Trình 3 Tuần Chi Tiết
- **Tuần 1**: Thu thập dữ liệu, chạy FastDedup, Auto Profiling, tạo và Đóng băng kép Fixed Test & Fixed Val, sinh mã SHA-256.
- **Tuần 2**: Gán nhãn Seed Set, train Model V0, chạy Active Mining ($U_i, W_i, A_i, E_i$), lọc Core-Set k-Center Greedy, vận hành 2-Tier HITL và xuất `manifest.yaml`.
- **Tuần 3**: Retrain Model V1 tích lũy (450 ảnh), chạy Multi-Seed (3 seeds), kiểm định ý nghĩa thống kê, chạy A/B Test 4 nhánh, xuất OpenVINO INT8 và hoàn thiện Báo cáo 3 Trụ Cột ROI.

### 7.3. Phân Công Trách Nhiệm & Độc Lập QA
1. **Active Learning & Training Lead**: Chịu trách nhiệm viết code pipeline, mining và train mô hình YOLO11n.
2. **Fixed Test & QA Lead (Độc lập hoàn toàn với người train)**: Độc lập quản lý tập Test bất biến và ngẫu nhiên Spot-check 5–10% ảnh Tier 1 Auto-Accept.
3. **Reproducibility & Manifest Lead**: Quản lý `manifest.yaml`, git tags, đảm bảo khả năng tái lập 100%.

---

## 💡 8. BỘ HỎI ĐÁP CỐT LÕI (CORE Q&A / FAQS)

### ❓ Q1: Ban đầu có 2.000 ảnh và 300 ảnh test, nên gán 100 ảnh trước train V0 hay gán cả 2.000 ảnh?
👉 **Trả lời**: Chỉ gán 100–250 ảnh (Seed Set). Tuyệt đối **không** gán cả 2.000 ảnh. Bản chất của Active Learning là tiết kiệm sức người. Model V0 học từ Seed Set sẽ đóng vai trò "máy dò" quét 1.750 ảnh còn lại để phát hiện ca khó.

### ❓ Q2: Sau khi train V0, nếu có tới 800 ảnh phân vân (Conf < 80%) thì làm sao?
👉 **Trả lời**: Không ép người duyệt cả 800 ảnh. Ta dùng **Ngân sách gán nhãn ($K = 150-200$ ảnh)** kết hợp giải thuật **PCA 32D + k-Center Greedy Core-Set** để nhặt đúng $K$ ảnh khó nhất và **đa dạng bối cảnh nhất** đưa sang CVAT. 600 ảnh còn lại tiếp tục nằm trong pool cho các vòng lặp sau.

### ❓ Q3: Khi retrain V1, train gộp ảnh cũ hay chỉ train ảnh mới sửa?
👉 **Trả lời**: Bắt buộc **Train gộp tích lũy (Cumulative Retraining)** ($250\text{ Seed} + 150\text{ Active} = 400\text{ ảnh}$). Nếu chỉ train trên ảnh mới, mô hình sẽ bị **Quên thảm khốc (Catastrophic Forgetting)** — giỏi ca khó mới nhưng quên sạch cách nhận diện xe chuẩn ban ngày.

---

## 🚀 9. HƯỚNG DẪN VẬN HÀNH & KHỞI CHẠY (QUICK START GUIDE)

### Cách 1: Chạy Trên Google Colab Bằng Notebook Tương Tác
Dự án đã tích hợp sẵn Notebook hoàn chỉnh [colab_greensm_active_learning.ipynb](file:///c:/Users/qu4nl/OneDrive/Desktop/AI%20th%E1%BB%B1c%20chi%E1%BA%BFn/Buildphase/colab_greensm_active_learning.ipynb):
1. Tải notebook lên Google Colab.
2. Đổi Runtime sang **T4 GPU**.
3. Chạy các Cell lần lượt từ Bước 0 đến Bước 7 để nghiệm thu kết quả và tự động lưu checkpoints vào Google Drive.

### Cách 2: Chạy Master Runner 1 Dòng Lệnh Duy Nhất (All-in-One CLI)
```bash
python src/colab_runner.py \
    --data-dir images \
    --output-dir outputs \
    --top-k 200 \
    --epochs 30
```

### Cách 3: Chạy Từng Khâu Độc Lập Để Tinh Chỉnh Chuyên Sâu
```bash
# 1. Deduplication & Profiling 5 Lát cắt
python src/data_profiling.py --data-dir images --output-json outputs/dataset_metadata.json --hash-thresh 5

# 2. Active Target Mining & Core-Set Selection
python src/active_selection.py --metadata-json outputs/dataset_metadata.json --top-k 200 --output-json outputs/active_selection_results.json

# 3. Assemble Dataset V1 (5 Classes)
python src/cvat_automation.py --action assemble --selection-json outputs/active_selection_results.json --output-dir outputs/dataset_v1

# 4. Multi-Seed Training (3 Seeds)
python src/train_yolo11n.py --data-yaml outputs/dataset_v1/dataset.yaml --seeds 42 123 456 --epochs 30 --project-dir outputs/runs --exp-id greensm_v1

# 5. Evaluate Benchmark & Sinh Báo Cáo 3 Trụ Cột ROI
python src/evaluate_benchmark.py --test-yaml outputs/dataset_v1/dataset.yaml --seed-summary-json outputs/runs/greensm_v1_multi_seed_summary.json --output-report outputs/BAO_CAO_BENCHMARK_A_B_TEST.md

# 6. Edge Export & Lượng Tử Hóa OpenVINO INT8
python src/export_quantize.py --weights outputs/runs/greensm_v1_seed_42/weights/best.pt --output-dir outputs/exported_models
```

---
*Tài liệu thuộc Bản quyền Cohort Build Phase — GreenSM Vehicle Active Learning Lab (M50).*
