# BÁO CÁO KẾT QUẢ BENCHMARK MÔ HÌNH NHẬN DIỆN XE GREENSM
**Dự án**: GreenSM Vehicle Detection (Hệ thống Data-Centric & Active Learning)  
**Ngày thực hiện**: 06/10/2026  
**Môi trường thử nghiệm**: Thực nghiệm trực tiếp trên GPU Tesla T4 (Google Colab)

---

## 1. TÓM TẮT DÀNH CHO NGƯỜI QUẢN LÝ (EXECUTIVE SUMMARY)

* **Mục tiêu**: Tìm ra mô hình phát hiện xe GreenSM tối ưu nhất trên dữ liệu thực tế để **khóa cố định (lock)** cho toàn bộ vòng đời dự án (từ phiên bản V0 đến V3).
* **Kết quả lựa chọn**: **Khóa mô hình YOLO11n (Chiến thắng với điểm số 76.6 / 100)**.
* **Lý do chính**:
  1. **Nhận diện chính xác và ít bỏ sót**: Tìm đúng 91.5% số xe trên đường và 90.9% các xe ở xa (xe nhỏ).
  2. **Tốc độ nhanh nhất**: Đạt 21.4 hình/giây (nhanh gấp hơn 2 lần đối thủ RT-DETR), đáp ứng tốt yêu cầu xử lý video thực tế.
  3. **Siêu nhẹ và tiết kiệm**: File mô hình chỉ 5.5 MB (nhẹ hơn 12 lần so với RT-DETR), tốn ít bộ nhớ GPU khi huấn luyện.
  4. **Phù hợp nhất cho khâu lọc dữ liệu (Active Learning)**: Phân loại rất rạch ròi giữa ảnh "chắc chắn đúng" và ảnh "cần người kiểm tra lại", giúp tiết kiệm tối đa thời gian gán nhãn của con người.

---

## 2. BÀI TOÁN & ĐẶC ĐIỂM DỮ LIỆU THỰC TẾ

Dữ liệu gồm **217 bức ảnh chụp thực tế** với **243 xe GreenSM** trên đường:
* **Đơn lớp đối tượng**: Chỉ có duy nhất 1 đối tượng cần nhận diện là xe `GreenSM`.
* **Nhiều xe ở khoảng cách xa (41.6%)**: Hơn 40% xe xuất hiện dưới dạng vật thể nhỏ hoặc rất nhỏ, đòi hỏi mô hình phải "mắt tinh" để không bỏ sót.
* **Có ảnh nền không chứa xe (12 ảnh)**: Giúp kiểm tra mô hình có bị "nhìn gà hóa cuốc" (nhận nhầm xe khác thành GreenSM) hay không.
* **Chống gian lận dữ liệu (Data Leakage)**: Dữ liệu chứa các chuỗi ảnh cắt từ cùng một video. Chúng tôi đã gom các chuỗi này lại thành từng nhóm trước khi chia tập Huấn luyện (Train) và Kiểm tra (Test), đảm bảo tập Test hoàn toàn xa lạ với mô hình để phản ánh đúng thực tế.

---

## 3. BẢNG SO SÁNH KẾT QUẢ THỰC NGHIỆM

Toàn bộ 4 mô hình đều được huấn luyện trên cùng một tập dữ liệu, cùng kích thước ảnh và cùng phần cứng:

| Tiêu chí đánh giá | YOLOv8n | **YOLO11n** | YOLO26n | RT-DETR | Giải thích ý nghĩa |
|---|:---:|:---:|:---:|:---:|---|
| **Tỷ lệ bắt trúng xe (Recall)** | 89.2% | **91.5%** | 84.0% | **93.9%** | Càng cao càng ít bỏ sót xe trên đường |
| **Bắt xe ở khoảng cách xa (Small Recall)** | 86.4% | **90.9%** | 81.8% | **95.5%** | Khả năng thấy xe nhỏ/xa (rất quan trọng) |
| **Độ chính xác tổng thể (mAP50-95)** | 75.3% | **76.4%** | 75.5% | **77.2%** | Điểm số chuẩn đánh giá chất lượng phát hiện |
| **Tốc độ xử lý (FPS)** | 20.6 FPS | **21.4 FPS** | 18.7 FPS | 10.1 FPS | Tốc độ quét ảnh (YOLO11n nhanh nhất) |
| **Dung lượng mô hình** | 6.2 MB | **5.5 MB** | 5.4 MB | 66.2 MB | Càng nhẹ càng dễ cài đặt và chạy trên camera |
| **Bộ nhớ GPU tiêu tốn** | 2.0 GB | **2.3 GB** | 2.5 GB | 6.7 GB | Tốn ít tài nguyên thì chi phí vận hành càng rẻ |
| **Khả năng hỗ trợ lọc dữ liệu (Active Learning)** | Tốt | **Xuất sắc** | Khá | Kém | Phân loại rõ ràng ảnh dễ và ảnh khó |
| **ĐIỂM TỔNG HỢP (Thang 100)** | **47.9** | **76.6 (Hạng 1)** | **23.2** | **75.0 (Hạng 2)** | Đánh giá toàn diện theo 7 tiêu chí |

---

## 4. TẠI SAO CHỌN YOLO11n MÀ KHÔNG CHỌN CÁC MÔ HÌNH KHÁC?

### Vì sao không chọn RT-DETR (Dù độ chính xác nhỉnh hơn một chút)?
* RT-DETR đạt độ chính xác 77.2% (hơn YOLO11n đúng 0.8%), nhưng **phải trả giá quá đắt**:
  1. **Chậm gấp đôi**: Chỉ đạt 10.1 hình/giây so với 21.4 hình/giây của YOLO11n.
  2. **Nặng gấp 12 lần & ngốn GPU gấp 3 lần**: File nặng tới 66 MB và ngốn 6.7 GB GPU (dễ gây nghẽn hệ thống khi mở rộng).
  3. **Không dùng được cho lọc dữ liệu thông minh**: Khi dự đoán, có tới gần 90% số kết quả của RT-DETR rơi vào trạng thái "nửa vời / không tự tin". Người quản lý không thể biết được ảnh nào mô hình chắc chắn đúng để tự duyệt, ảnh nào model nghi ngờ để đưa người kiểm tra.

### Vì sao không chọn YOLO26n?
* Mặc dù mang tên gọi "mới hơn", nhưng khi chạy thực tế trên ảnh GreenSM thì:
  * Bỏ sót nhiều xe nhất trong 4 mô hình (chỉ bắt được 84% số xe, để lọt 6 xe trong bài kiểm tra).
  * Khả năng bắt xe ở xa rất kém (chỉ 81.8%).
  * Xếp hạng chót với điểm tổng hợp chỉ 23.2 / 100.

### Vì sao YOLO11n vượt trội so với bản tiền nhiệm YOLOv8n?
* YOLO11n nhận diện xe chuẩn hơn, bắt xe ở xa tốt hơn (+4.5%), chạy nhanh hơn và dung lượng nhẹ hơn 12% so với YOLOv8n.

---

## 5. ĐÁNH GIÁ TÍNH PHÙ HỢP CHO ACTIVE LEARNING (HUMAN-IN-THE-LOOP)

Mục tiêu tiếp theo của dự án là xây dựng hệ thống **Active Learning** (Mô hình tự lọc ra các ảnh khó để con người xem xét và gán nhãn thêm):
* **YOLO11n là mô hình lý tưởng nhất**:
  * **59.0% số trường hợp**: Mô hình tự tin tuyệt đối $\rightarrow$ Hệ thống tự động chấp nhận kết quả, không cần con người can thiệp.
  * **37.3% số trường hợp**: Mô hình nhận thấy đây là ca khó (bị che khuất, thiếu sáng, xe ở quá xa) $\rightarrow$ Tự động lọc ra đưa vào hàng đợi cho con người gán nhãn bổ sung.
  * Thử nghiệm đảo chiều ảnh (lật ngang) cho thấy kết quả dự đoán của YOLO11n đạt độ ổn định 100%.

---

## 6. QUYẾT ĐỊNH & HƯỚNG ĐI TIẾP THEO

> **KHÓA CỐ ĐỊNH (LOCK) MÔ HÌNH: YOLO11n**

1. **Nguyên tắc bất di bất dịch**: Từ phiên bản V0 $\rightarrow$ V1 $\rightarrow$ V2 $\rightarrow$ V3, toàn bộ dự án sẽ dùng cố định một cấu trúc mô hình **YOLO11n**. Tuyệt đối không đổi mô hình giữa chừng.
2. **Chứng minh hiệu quả**: Mọi sự tăng trưởng về độ chính xác ở các phiên bản V1, V2, V3 trong tương lai sẽ được đảm bảo 100% đến từ **chất lượng dữ liệu được con người bổ sung (Data-Centric)**, chứ không phải do đổi mô hình.
