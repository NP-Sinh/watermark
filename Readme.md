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
Quy trình thực hiện:
```bash
### 1. Quá trình khởi tạo
- Thuật toán sử dụng các tham số: kích thước khối (m×n) và số bit r cần giấu trong mỗi khối
- Ràng buộc: 2^r - 1 ≤ m×n
- Tạo ma trận khóa K (nhị phân kích thước m×n) ngẫu nhiên
- Tạo ma trận trọng số W (m×n) với các giá trị thuộc {1, 2, ..., 2^r-1}

### 2. Quá trình nhúng thủy vân
1. Chia ảnh gốc thành các khối kích thước m×n
2. Chuyển đổi thông điệp cần nhúng thành các đoạn r-bit
3. Với mỗi khối F:
   - Tính T = F ⊕ K (XOR giữa khối và khóa)
   - Tính S = ∑(T×W) mod 2^r (tổng có trọng số)
   - Xây dựng các tập Z chứa các vị trí có thể thay đổi 
   - Tính d = b - S mod 2^r (b là giá trị thập phân của r-bit cần nhúng)
   - Nếu d=0: không cần sửa đổi
   - Nếu d≠0: thay đổi 1-2 bit phù hợp trong khối để S'=b

"Quá trình trích xuất thủy vân" 
1. Chia ảnh đã nhúng thành các khối m×n
2. Với mỗi khối F':
   - Tính T' = F' ⊕ K
   - Tính S' = ∑(T'×W) mod 2^r
   - Chuyển S' thành biểu diễn nhị phân để tạo r-bit trích xuất

### 4. Các bước xử lý tổng thể
1. Đọc và xử lý ảnh gốc:
   - Chuyển thành ảnh nhị phân (0-1)
   - Điều chỉnh kích thước thành bội số của m×n
2. Đọc và xử lý ảnh thủy vân:
   - Chuyển thành dãy bit
   - Cắt bớt nếu dài hơn dung lượng có thể nhúng
3. Thực hiện nhúng thủy vân
4. Đánh giá kết quả:
   - PSNR (Peak Signal-to-Noise Ratio)
   - Số pixel đã thay đổi
   - Thời gian xử lý
5. Trích xuất và đánh giá độ chính xác:
   - Tỷ lệ bit lỗi (BER)
   - Độ chính xác phục hồi

Thuật toán này đảm bảo khả năng trích xuất thủy vân mà không cần ảnh gốc, thông qua việc sử dụng các ma trận khóa K và ma trận trọng số W.

```
## 🏷️Thuật toán SW
```bash
python watermark_app.py
```
## 🏷️Thuật toán WU-LEE
```bash
python watermark_app.py
```