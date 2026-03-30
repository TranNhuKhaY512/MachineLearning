
## Hệ Thống Dự Đoán Khả Năng Tái Nhập Viện (Bệnh Tiêu Hóa & Tuần Hoàn)
#### Đồ án môn học: Học Máy Ứng Dụng 
#### Ngành: Trí Tuệ Nhân Tạo 

---

#### Giới thiệu dự án:
Dự án này là một hệ thống ứng dụng học máy nhằm dự đoán khả năng tái nhập viện của bệnh nhân mắc các bệnh về tiêu hóa và tuần hoàn. Hệ thống phân tích dữ liệu dựa trên hồ sơ y tế, thói quen sinh hoạt và lịch sử điều trị để đưa ra cảnh báo nguy cơ tái nhập viện. Sự kết hợp giữa thuật toán học máy và giao diện Web Chatbot giúp bệnh nhân và cơ sở y tế dễ dàng tương tác, qua đó tối ưu hóa công tác phòng ngừa và quản lý bệnh trạng.
1. Tính năng nổi bật
- Dự đoán bằng Machine Learning: Sử dụng thuật toán Random Forest Classifier (với 100 cây quyết định) để xử lý bài toán phân loại nhị phân, dự đoán xem bệnh nhân có nguy cơ tái nhập viện hay không.
- Trợ lý Y tế Ảo (Chatbot): Tích hợp Google Gemini API (Gemini 2.5 Flash Lite) để xử lý ngôn ngữ tự nhiên. Chatbot có khả năng giải thích kết quả dự đoán của học máy và đưa ra các lời khuyên y tế, chế độ dinh dưỡng dễ hiểu cho người dùng.
- Tinh chỉnh xác suất thông minh: Hệ thống được thiết lập các luật nghiệp vụ y khoa cơ bản để tinh chỉnh tỷ lệ phần trăm nguy cơ (ví dụ: tăng 10% rủi ro nếu bệnh nhân trên 60 tuổi, tăng 20% nếu tình trạng bệnh nghiêm trọng).
- Giao diện thân thiện: Cho phép người dùng nhập trực tiếp thông tin bằng ngôn ngữ tự nhiên hoặc tra cứu qua ID bệnh nhân.

---

#### Công nghệ sử dụng
- Ngôn ngữ lập trình: Python 3.11.5
- Backend Framework: Flask.
- Machine Learning: scikit-learn (triển khai RandomForestClassifier và LabelEncoder).
- Xử lý dữ liệu: pandas, numpy.AI API: google-generativeai (Sử dụng model Gemini 2.5 Flash Lite).
- Frontend: HTML, CSS, JavaScript.

---

#### Cấu trúc dữ liệu (Dataset)
- Dữ liệu được lưu trữ trong tệp patient_data.csv.
- Cấu trúc bảng dữ liệu bao gồm các trường: ID, Tên bệnh nhân, Tuổi, Bệnh (Bệnh chính), Bệnh nền, Số lần nhập viện, và Nghiêm trọng (Đánh giá mức độ, 0 hoặc 1).
- Dữ liệu phân loại dạng chuỗi được chuyển đổi sang số nguyên thông qua LabelEncoder trước khi đưa vào huấn luyện.
- Độ chính xác :
<img width="1180" height="519" alt="image" src="https://github.com/user-attachments/assets/c8cb12a9-bfc4-481e-a22f-62d1782ca783" />
- Giao diện người dùng :
<img width="1221" height="826" alt="image" src="https://github.com/user-attachments/assets/50a89bff-44e5-43a4-9415-fd1f6590e36a" />



#### Hướng dẫn cài đặt và chạy
1. Clone repository:
```Bash
git clone https://github.com/TranNhuKhaY512/MedRisk_AI_Readmission_Predictor.git
cd MedRisk_AI_Readmission_Predictor
```
2. Thiết lập môi trường ảo và cài đặt thư viện:
- Chạy lệnh sau để cài đặt các gói cần thiết:
```
Bash
pip install flask pandas numpy scikit-learn google-generativeai
```
3. Cấu hình API Key:
- Bạn cần có khóa API của Google Gemini để kích hoạt tính năng Chatbot. Hãy tích hợp khóa bảo mật vào mã nguồn (sử dụng thư viện os để bảo mật).
4. Khởi động Server:
- Chạy ứng dụng bằng lệnh:
```
Bash
python app.py
```
[Lưu ý: Mô hình Random Forest sẽ tự động được huấn luyện khi khởi động server Flask].
5. Trải nghiệm ứng dụng:
- Mở trình duyệt web và truy cập vào địa chỉ mặc định http://127.0.0.1:5050 để sử dụng Trợ lý Y tế Ảo.

---

## 📞 Thông tin liên hệ

Nếu bạn có bất kỳ câu hỏi nào về dự án hoặc muốn trao đổi thêm, vui lòng liên hệ với đại diện nhóm:

* **Trần Như Khả Ý**
* **Email:** trannhukhayy0512@gmail.com 
* **Điện thoại:** 0364551205 
* **GitHub:** [TranNhuKhaY512](https://github.com/TranNhuKhaY512) 
* **LinkedIn:** [linkedin.com/in/trannhukhay051205](https://linkedin.com/in/trannhukhay051205)
