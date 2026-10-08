# 1. MỤC TIÊU BÀI TOÁN

## 1.1. Bài toán

Dự án xây dựng một quy trình **Human-in-the-Loop + Active Learning + MLOps/MLDOP** để huấn luyện và liên tục cải thiện **một mô hình duy nhất** dùng cho bài toán phát hiện xe GreenSM.

Bài toán cốt lõi không phải là xây dựng một hệ thống MLOps lớn.

Mục tiêu chính là:

> **Với cùng một ngân sách gán nhãn của con người, làm thế nào để chọn ra những ảnh có giá trị nhất để gán nhãn, từ đó giúp mô hình GreenSM cải thiện nhanh hơn so với việc chọn ảnh ngẫu nhiên?**

---

# 2. ĐẦU VÀO CỦA BÀI TOÁN

Hệ thống bắt đầu với:

### Raw Dataset

Một tập ảnh GreenSM chưa được gán nhãn:

```text
Raw GreenSM Images
 │
 ├── Image 001
 ├── Image 002
 ├── Image 003
 ├── ...
 └── Image N
```

Ngoài ảnh, hệ thống có thể thu thập thêm metadata nếu có:
- Scene / sequence ID
- Timestamp
- Camera
- Điều kiện môi trường
- Kích thước object
- Vị trí object
- Chất lượng ảnh

---

# 3. MỤC TIÊU ĐẦU RA

Hệ thống cuối cùng cần tạo ra **3 nhóm đầu ra chính**.

## Output 1 — Mô hình GreenSM tốt hơn

Sau mỗi vòng lặp:

```text
Model V0
  ↓
Model V1
  ↓
Model V2
  ↓
Model V3
  ↓
...
```

Tất cả đều là **các phiên bản của cùng một mô hình**, không phải nhiều mô hình khác nhau.

Mục tiêu:

```text
mAP ↑
Recall ↑
Precision ↑
F1 ↑
```

Đặc biệt phải cải thiện các trường hợp mô hình yếu:

```text
Small Object
Night
Rain
Occlusion
Blur
Dense Scene
...
```

## Output 2 — Dataset ngày càng tốt hơn

Sau mỗi vòng:

```text
Dataset V0
  ↓
Dataset V1
  ↓
Dataset V2
  ↓
Dataset V3
```

Dataset mới không đơn giản là “thêm ảnh”.

Nó phải là:

> **Tập dữ liệu được bổ sung những mẫu có giá trị cao đối với lỗi hiện tại của mô hình.**

Ví dụ:

```text
Model V0 yếu ở:
Small Object
Rain
Occlusion
  ↓
Active Learning tìm ảnh tương ứng
  ↓
Human Review
  ↓
Dataset V1
```

## Output 3 — Evidence chứng minh phương pháp hiệu quả

Đây là output rất quan trọng của đề tài.

Không chỉ nói:
> “Active Learning giúp model tốt hơn.”

Mà phải chứng minh bằng thực nghiệm:

```text
Cùng Model V0
Cùng Test Set
Cùng số lượng ảnh được gán nhãn
Cùng cấu hình train
  ↓
  ┌───────────────┬───────────────┐
  │                               │
  ▼                               ▼
Random Sampling         Intelligent Sampling
  │                               │
  ▼                               ▼
Model R1                        Model U1
  │                               │
  └───────────────┬───────────────┘
                  ▼
            Fixed Test Set
```

Ví dụ:

| Phương pháp | 200 ảnh | mAP |
| :--- | :---: | :---: |
| Random | 200 | 70% |
| Intelligent | 200 | **74%** |

Kết luận:

```text
Intelligent Sampling
→ cùng 200 ảnh được label
→ nhưng model tăng nhiều hơn
```

Đây mới là bằng chứng cho **data efficiency**.

---

# 4. PIPELINE TỔNG THỂ

Toàn bộ bài toán có thể nhìn thành:

```text
                  RAW GREENSM DATA
                         │
                         ▼
                   DATA PROFILING
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
  FIXED TEST SET                   UNLABELED POOL
         │                               │
  Human Annotation                       │
         │                               │
         ▼                               │
    QA / REVIEW                          │
         │                               │
         ▼                               │
 GROUND TRUTH TEST SET                   │
         │                               │
      FREEZE                             │
         │                               │
         │                               ▼
         │                         SEED SELECTION
         │                               │
         │                           ~10% DATA
         │                               │
         │                               ▼
         │                         HUMAN LABELING
         │                               │
         │                               ▼
         │                           DATASET V0
         │                               │
         │                               ▼
         │                         TRAIN MODEL V0
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
                 EVALUATE MODEL V0
                         │
                         ▼
                  MODEL DIAGNOSIS
                         │
             Find Weak Cases / Errors
                         │
                         ▼
                 INFERENCE ON POOL
                         │
                         ▼
               PRELABEL / PREDICTION
                         │
                         ▼
               UNCERTAINTY + ANOMALY
                         │
                         ▼
                ERROR-DRIVEN TARGET
                         │
                         ▼
             DEDUP + DIVERSITY FILTER
                         │
                         ▼
                SELECT TOP-K IMAGES
                         │
                         ▼
                   HUMAN REVIEW
                         │
                         ▼
                    DATASET V1
                         │
                         ▼
                  TRAIN MODEL V1
                         │
                         ▼
             EVALUATE ON FIXED TEST
                         │
                         ▼
                  MODEL DIAGNOSIS
                         │
                         ▼
                      REPEAT
```

---

# 5. BẢN CHẤT CỦA MỖI GIAI ĐOẠN

## Bước 1 — Data Profiling

Trước khi train phải hiểu dữ liệu.

Ví dụ:

```text
10,000 ảnh

Day           55%
Night         20%
Rain          10%
Small Object   8%
Occlusion      5%
Blur           2%
```

Đồng thời kiểm tra:
- duplicate
- near-duplicate
- sequence/scene
- image quality
- object size
- object position
- occlusion
- density
- viewpoint

Mục tiêu:
> **Biết dữ liệu đang có gì và tránh chia train/test một cách sai lệch.**

---

# 6. BƯỚC 2 — TẠO FIXED TEST SET

Một phần dữ liệu được chọn để tạo **Ground Truth Test Set**.

```text
Raw Data
  ↓
Stratified / Group-aware Split
  ↓
Test Candidates
  ↓
Human Annotation
  ↓
QA / Review
  ↓
Ground Truth Test Set
  ↓
FREEZE
```

Sau khi freeze:
- Test Set **không được dùng để train** và **không được dùng để chọn ảnh Active Learning**.
- Mọi Model V0, V1, V2... đều phải đánh giá trên **cùng Test Set**.

Nhờ vậy:

```text
Model V0 ─┐
Model V1 ─┤
Model V2 ─┼──→ SAME TEST SET
Model V3 ─┘
```

Có thể so sánh công bằng.

---

# 7. BƯỚC 3 — TẠO SEED DATASET

Khoảng 10% dữ liệu được chọn làm dữ liệu ban đầu.

Không nên chỉ:
```text
Random 10%
```

Mà nên:
```text
Stratified Sampling
+
Group-aware Sampling
```

để seed có đại diện cho:
- Day / Night
- Rain
- Small / Large object
- Occlusion
- Dense / Sparse
- Blur
- các case quan trọng khác.

Sau đó:

```text
Seed Images
  ↓
Human Annotation
  ↓
Dataset V0
```

---

# 8. BƯỚC 4 — TRAIN MODEL V0

Dataset V0 được dùng để huấn luyện **mô hình GreenSM đầu tiên**.

```text
Dataset V0
  ↓
Training
  ↓
Model V0
```

Model V0 là checkpoint khởi đầu của toàn bộ vòng đời.

---

# 9. BƯỚC 5 — ĐÁNH GIÁ MODEL V0

Model V0 được chạy trên Fixed Test Set.

Thu được:
```text
mAP
Precision
Recall
F1
IoU
```

Quan trọng hơn là **phân tích theo từng slice**.

Ví dụ:

```text
Day          Recall = 92%
Night        Recall = 78%
Rain         Recall = 65%
Small Object Recall = 58%
Occlusion    Recall = 61%
Blur         Recall = 63%
```

Từ đó hệ thống biết:
> **Model đang yếu ở đâu?**

Ví dụ:
```text
Small Object
Rain
Occlusion
Blur
```

---

# 10. BƯỚC 6 — TÌM HARD CASE TRONG UNLABELED POOL

Đây là phần **Active Learning**.

Model V0 chạy inference trên phần dữ liệu chưa được label:

```text
Unlabeled Pool
  ↓
Model V0
  ↓
Predictions
```

Sau đó tính các tín hiệu:

### Uncertainty
Ví dụ:
```text
Confidence thấp
```

### No Prediction
```text
Model không phát hiện object
```

### Prediction Anomaly
Ví dụ:
```text
BBox quá nhỏ
BBox bất thường
Số lượng prediction bất thường
```

### Prediction Instability
Chạy ảnh gốc và ảnh biến đổi nhẹ:

```text
Original
  ↓
Prediction A

Brightness Augmentation
  ↓
Prediction B

A ≠ B
→ Unstable
```

---

# 11. BƯỚC 7 — ERROR-DRIVEN TARGETING

Không chỉ chọn ảnh có confidence thấp.

Hệ thống còn dựa vào:
> **Model yếu ở đâu?**

Ví dụ:

Test Set cho thấy:
```text
Small Object = yếu nhất
Rain         = yếu
Occlusion    = yếu
```

Thì khi tìm trong Unlabeled Pool:

```text
Rain + Small Object + Uncertain
  ↓
  ưu tiên
```

Có thể hình dung:

```text
Candidate Score
=
Weak Case Relevance
×
Uncertainty
×
Anomaly
```

Mục tiêu:
> **Chọn những ảnh vừa đáng nghi, vừa liên quan đến lỗi mà model đang thực sự gặp phải.**

---

# 12. BƯỚC 8 — DEDUP + DIVERSITY

Sau khi lấy Top-N candidate:

```text
Top 1000 candidates
  ↓
Remove duplicate
  ↓
Remove near-duplicate
  ↓
Diversity selection
  ↓
Final 200 images
```

Mục tiêu tránh trường hợp:

```text
Frame 001
Frame 002
Frame 003
Frame 004
...
```

đều gần như cùng một cảnh.

Thay vì:
```text
200 ảnh ≈ 1 cảnh
```

muốn:
```text
200 ảnh
→ nhiều scene
→ nhiều điều kiện
→ nhiều kích thước object
→ nhiều mức độ khó
```

---

# 13. BƯỚC 9 — HUMAN-IN-THE-LOOP

Top-K ảnh được đưa cho annotator.

```text
Selected Images
  ↓
Prelabel
  ↓
CVAT
  ↓
Human Review / Correction
  ↓
Accepted Ground Truth
```

Guideline Chatbot hỗ trợ annotator:

```text
Annotator
 │
 ├── CVAT
 │
 └── Guideline Chatbot
       ↓
     Tra cứu guideline
```

Chatbot **không quyết định Ground Truth cuối cùng**. Con người vẫn là người xác nhận label.

---

# 14. BƯỚC 10 — TẠO DATASET V1

Sau khi human review xong:

```text
Dataset V0
  +
New Human-verified Labels
  ↓
Dataset V1
```

Dataset V1 phải lưu được:
- Frame ID
- Label
- Source
- Selection strategy
- Experiment ID
- Annotation status
- Dataset version

---

# 15. BƯỚC 11 — RETRAIN MODEL V1

```text
Dataset V1
  ↓
Same Model Architecture
  ↓
Training
  ↓
Model V1
```

Quan trọng:
> **Không đổi sang một model khác để “ăn điểm”.**

Model V0 → V1 → V2 là **cùng một model line**.

---

# 16. BƯỚC 12 — ĐÁNH GIÁ LẠI

Model V1 tiếp tục chạy trên Fixed Test Set:

```text
Model V0
  ↓
Test
  ↓
mAP = 68%

Model V1
  ↓
SAME TEST
  ↓
mAP = 73%
```

Khi đó:

```text
ΔmAP = 73% - 68%
     = +5%
```

Đồng thời kiểm tra:

```text
Rain Recall
Small Object Recall
Occlusion Recall
Blur Recall
```

Xem các lỗi cũ có thực sự được cải thiện hay không.

---

# 17. BƯỚC 13 — LẶP LẠI

Nếu vẫn còn dữ liệu chưa label:

```text
Model V1
  ↓
Test
  ↓
Diagnosis
  ↓
Find Weak Cases
  ↓
Unlabeled Pool
  ↓
Uncertainty
  ↓
Error-driven Selection
  ↓
Dedup
  ↓
Diversity
  ↓
Human Review
  ↓
Dataset V2
  ↓
Model V2
  ↓
Test
  ↓
...
```

Đây chính là:
> **Closed-loop Human-in-the-Loop Active Learning.**

---

# 18. MLOps / MLDOP NẰM Ở ĐÂU?

MLOps/MLDOP không phải một bước đứng riêng trong pipeline.

Nó là **lớp quản lý toàn bộ vòng đời**.

```text
 MLOps / MLDOP
─────────────────────────────────────────
Dataset V0
  ↓
Experiment
  ↓
Training
  ↓
Model V0
  ↓
Evaluation
  ↓
Selection
  ↓
Dataset V1
  ↓
Experiment
  ↓
Training
  ↓
Model V1
  ↓
...
```

Hệ thống phải biết:

```text
Model V2
  ↓
được train từ Dataset nào?

Dataset V2
  ↓
chứa frame nào?

Frame đó
  ↓
được chọn bằng chiến lược nào?

Experiment nào
  ↓
đã tạo ra Model V2?
```

Đó chính là **traceability / lineage / reproducibility**.

---

# 19. ĐẦU RA CUỐI CÙNG CỦA TOÀN BỘ ĐỀ TÀI

Sau khi hoàn thành MVP, hệ thống phải tạo được:

### 1. Một model GreenSM tốt hơn

```text
Model V0
  ↓
Model V1
  ↓
Model V2
```

với:

```text
mAP ↑
Recall ↑
```

### 2. Một dataset có version

```text
Dataset V0
Dataset V1
Dataset V2
...
```

### 3. Một pipeline Active Learning

```text
Evaluate
  ↓
Diagnose
  ↓
Select
  ↓
Label
  ↓
Retrain
  ↓
Evaluate
  ↓
Repeat
```

### 4. Một hệ thống Human-in-the-Loop

```text
AI
  ↓
Gợi ý / Prelabel
  ↓
Human
  ↓
Correction
  ↓
Ground Truth
```

### 5. Một lớp MLOps/MLDOP có traceability

```text
Data
  ↓
Selection
  ↓
Experiment
  ↓
Training
  ↓
Model
  ↓
Evaluation
```

### 6. Một báo cáo thực nghiệm chứng minh hiệu quả

Ví dụ:

| Metric | Baseline | Random | Intelligent |
| :--- | :---: | :---: | :---: |
| Labeled images | 1,000 | +200 | +200 |
| mAP | 68% | 70% | **73%** |
| Recall | 76% | 78% | **82%** |
| Small Object Recall | 58% | 61% | **68%** |
| Rain Recall | 65% | 67% | **72%** |
| Annotation budget | — | 200 | 200 |

Kết luận cần chứng minh:

> **Intelligent Sampling giúp mô hình đạt chất lượng cao hơn với cùng một lượng dữ liệu được con người gán nhãn.**

Đây là **mục tiêu khoa học/kỹ thuật chính của đề tài**, còn CVAT, Chatbot, MLOps, Dataset Versioning và Dashboard là các thành phần hỗ trợ để thực hiện và chứng minh mục tiêu đó.
