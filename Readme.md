# Digital Watermarking App

Nghiên cứu kỹ thuật thủy vân số và xây dựng ứng dụng bảo vệ bản quyền ảnh số
- Tìm hiểu về các kỹ thuật che giấu tập tin
- Tìm hiểu phương pháp và mô hình thủy vân số
- Tìm hiểu về các thuật toán thủy vân theo miền không gian ảnh (SW; WU- LEE; LBS; PCT,...)
- Tìm hiểu về các thuật toán thủy vân theo mền tần số (DCT; DWT)
- Xây dựng chương trình thử nghiệm cài đặt một số thuật toán thủy vân nhằm ứng dụng vào việc xác thực thông tin và bảo vệ bản quyền cho dữ liệu ảnh số

## Cài đặt
```bash
pip install -r requirements.txt
```

## 🗿 Chạy chương trình
```bash
python watermark_app.py
```

## 🏷️ Các thuật toán được cài đặt
Ứng dụng này cài đặt 4 thuật toán thủy vân khác nhau:
- PCT (Parity-Check-Based Technique)
- Wu-Lee
- SW (Simple Watermarking)
- LSB (Least Significant Bit)

## 🏷️Thuật toán PCT
Giải thích Thuật toán PCT
```bash
Thuật toán PCT là phương pháp giấu thông tin vào ảnh nhị phân (ảnh đen trắng).

## Mục đích
- Giấu một ảnh thủy vân vào trong ảnh gốc mà không làm thay đổi quá nhiều chất lượng ảnh
- Cho phép trích xuất lại ảnh thủy vân bất cứ lúc nào khi biết khóa bí mật

## Cách hoạt động
1. "Chia nhỏ ảnh": Chia ảnh gốc thành nhiều ô vuông nhỏ (gọi là "khối")
2. "Giấu dữ liệu": Trong mỗi khối, thay đổi một vài điểm ảnh (thường chỉ 1-2 điểm) để mã hóa một phần nhỏ của thông điệp
3. "Sử dụng toán học": Dùng phép XOR và ma trận trọng số để quyết định những điểm ảnh nào cần thay đổi
4. "Khóa bí mật": Sử dụng hai ma trận bí mật (K và W) làm khóa để đảm bảo chỉ người có khóa mới trích xuất được thông tin

Cốt lõi của thuật toán là tìm cách thay đổi ít nhất các điểm ảnh mà vẫn đảm bảo giấu được đủ thông tin cần thiết, đồng thời cho phép khôi phục chính xác thông tin đã giấu.
```

## 🏷️Thuật toán LSB (Least Significant Bit)
```bash
Thuật toán LSB là phương pháp giấu thông tin vào bit ít quan trọng nhất của mỗi pixel trong ảnh.

## Mục đích
- Giấu thông tin (văn bản, hình ảnh) trong ảnh mà không gây thay đổi nhìn thấy được
- Tận dụng sự thiếu nhạy cảm của mắt người với những thay đổi nhỏ trong giá trị màu sắc

## Cách hoạt động
1. "Phân tích bit": Dựa trên tần suất xuất hiện của bit 0 và 1 trong ảnh gốc và thông điệp
2. "Đặt cờ": Xác định xem có cần đảo bit thông điệp hay không trước khi nhúng
3. "Nhúng thông tin": Thay đổi bit cuối cùng (LSB) của từng byte màu (thường là kênh Blue) trong ảnh
4. "Đánh dấu EOF": Sử dụng marker để xác định kết thúc thông điệp khi trích xuất

## Ưu điểm
- Đơn giản, dễ cài đặt
- Khả năng giấu lượng thông tin lớn (có thể lên đến 1/8 kích thước ảnh)
- Thay đổi rất nhỏ về mặt thị giác (PSNR cao)

## Hạn chế
- Dễ bị phát hiện bằng phân tích thống kê
- Không chống được với nén mất dữ liệu (JPEG, ...)
- Không chống được với các biến đổi hình học (xoay, cắt, ...)

Đây là thuật toán cơ bản, nhưng hiệu quả cho các ứng dụng steganography đơn giản. Tuy nhiên, trong môi trường thực, thường cần kết hợp với các phương pháp khác để tăng tính bảo mật.
```

## 🏷️Thuật toán SW

## 🏷️Thuật toán WU-LEE

## Các chức năng của ứng dụng
1. **Nhúng thủy vân**:
   - Chọn ảnh gốc
   - Chọn ảnh hoặc văn bản thủy vân
   - Chọn thuật toán thủy vân
   - Điều chỉnh các tham số của thuật toán
   - Xem ảnh đã nhúng thủy vân

2. **Trích xuất thủy vân**:
   - Tải ảnh đã nhúng thủy vân
   - Trích xuất thông tin thủy vân
   - Hiển thị thủy vân đã trích xuất

3. **Đánh giá chất lượng**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - Số pixel đã sửa đổi
   - Độ chính xác trích xuất
   - Thời gian xử lý

4. **Lưu kết quả**:
   - Lưu ảnh đã nhúng thủy vân
   - Lưu thủy vân đã trích xuất
   - Lưu báo cáo kết quả