Evidence & Baseline

Nhóm chưa có số liệu thời gian gán nhãn thực tế. Hướng gợi ý là đo ngay thủ công trước, thay vì ước lượng:
•    Gán thử một nhóm nhỏ ảnh trên CVAT theo hai cách, vẽ từ đầu và sửa nhãn mồi, rồi ghi lại thời gian trung bình mỗi ảnh. Đây sẽ là mốc để tính "time saved" về sau.
•    Ghi lại tỷ lệ frame trùng thực tế sau khi lọc, và mức độ chênh lệch kết quả khi train lại cùng một cấu hình. Đây là mốc cho chỉ số reproducibility.
Giải pháp

Các ý tưởng chính và hướng gợi ý
•    Two-Tier HITL (duyệt 2 tầng): ảnh mô hình rất chắc chắn thì tự duyệt, ảnh mô hình còn lưỡng lự thì đưa lên CVAT cho người sửa.
–    Nên chú ý vùng giữa hai ngưỡng tin cậy, để không có ảnh nào bị bỏ ngỏ.
–    Độ tin cậy chỉ phản ánh những xe mô hình đã thấy, không phản ánh xe bị sót. Có thể cân nhắc không tự duyệt những ảnh dễ sót xe (xe nhỏ, ban đêm) và kiểm tra ngẫu nhiên một phần ảnh tự duyệt để đo chất lượng nhãn máy.
•    Fast Deduplication: lọc frame gần trùng bằng pHash hoặc embedding để giảm khối lượng gán nhãn và suy luận.
–    Có thể chỉ so sánh frame trong cùng một video, và khi hai frame trùng thì ưu tiên giữ frame có nhiều xe hơn, tránh vô tình loại mất xe nhỏ ở xa.
•    Cost-Aware Hard-case Mining: chấm điểm mỗi ảnh để chọn ảnh "đáng gán" nhất.
–    Điểm có thể kết hợp nhiều yếu tố: đo sự quyết định lựa chọn của mô hình, nhóm dữ liệu mô hình đang yếu (xe nhỏ, ban đêm…), độ ổn định khi lật ảnh và công sức ước tính để gán.
–    Vì bài toán chỉ có một lớp, độ bất định nên đo trực tiếp trên độ tin cậy của box thay vì entropy giữa các lớp. Trọng số cho nhóm dữ liệu yếu nên lấy từ kết quả đánh giá thực tế thay vì đặt sẵn.
–    Có thể kết hợp thêm bước lọc đa dạng (ví dụ phân cụm ảnh) để ảnh được chọn không dồn vào cùng một bối cảnh.
ATOM5-T018 BuildPhase  •  12:27 AM
Forwarded
Tập test cố định: chia tập theo video để tránh rò rỉ dữ liệu, gán nhãn kỹ rồi giữ nguyên cho mọi mô hình -> đảm bảo sự cố định để mọi so sánh có ý nghĩa.
•    Experiment Manifest: mỗi mô hình đi kèm một file ghi lại danh sách ảnh, nguồn gốc nhãn, chiến lược chọn mẫu, cấu hình train, random seed và checkpoint. Một file YAML/JSON kèm git tag là đủ cho phạm vi đề tài.
•    Thí nghiệm đối chứng: so sánh nhánh Random và nhánh Active trên cùng số ảnh người gán. Nếu còn thời gian, có thể thêm một nhánh để tách riêng tác động của nhãn tự duyệt khỏi tác động của việc chọn ảnh khó.
Đầu ra

Cách đọc kết quả: tập test của đề tài không lớn, nên nhóm có thể chạy lại một vài lần với seed khác nhau để biết mức dao động tự nhiên, và chỉ coi một cải thiện là thật khi nó lớn hơn mức dao động này.
Kế hoạch

Nên ước tính sơ bộ tổng số ảnh cần người gán (tập test, seed, mỗi vòng cho từng nhánh) để chứng minh khối lượng vừa sức trong 3 tuần. Lịch trong phụ lục cũng nên được điều chỉnh cho khớp với kế hoạch này.
Phân công

Nên giao rõ người phụ trách kiểm tra chất lượng tập test (tốt nhất không trùng với người train mô hình) và người phụ trách manifest tái lập -> đây là hai phần quyết định độ tin cậy độc lập cho kết quả.