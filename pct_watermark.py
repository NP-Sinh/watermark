import numpy as np
import cv2
import math
import random
import time

class PCTWatermark:
    """
    Thuật toán thủy vân PCT (Chen, Pan, Tseng) cho ảnh nhị phân.
    """
    
    def __init__(self, block_size_m, block_size_n, r):
        """
        Khởi tạo thuật toán thủy vân PCT.
        
        Tham số:
            block_size_m (int): Chiều cao khối
            block_size_n (int): Chiều rộng khối
            r (int): Số bit cần giấu trong mỗi khối (phải thỏa mãn 2^r - 1 <= m*n)
        """
        self.m = block_size_m
        self.n = block_size_n
        self.r = r
        
        # Kiểm tra ràng buộc: 2^r - 1 <= m*n
        if 2**r - 1 > block_size_m * block_size_n:
            raise ValueError(f"r phải thỏa mãn: 2^r - 1 <= m*n, hiện tại 2^{r} - 1 = {2**r - 1} > {block_size_m * block_size_n}")
        
        # Tạo ma trận khóa K (ma trận nhị phân kích thước m x n)
        self.K = np.random.randint(0, 2, size=(block_size_m, block_size_n), dtype=np.uint8)
        
        # Tạo ma trận trọng số W
        self.W = self._generate_weight_matrix()
    
    def _generate_weight_matrix(self):
        """
        Tạo ma trận trọng số W kích thước m x n với các phần tử thuộc tập {1, 2, ..., 2^r - 1}
        Mỗi giá trị trong dải phải xuất hiện ít nhất một lần
        """
        # Tạo danh sách các giá trị cần thiết (từ 1 đến 2^r - 1)
        required_values = list(range(1, 2**self.r))
        
        # Khởi tạo ma trận trọng số với các giá trị 0
        W = np.zeros((self.m, self.n), dtype=np.int32)
        
        # Đầu tiên, đặt mỗi giá trị cần thiết ít nhất một lần
        indices = list(np.ndindex(self.m, self.n))
        random.shuffle(indices)
        
        for i, value in enumerate(required_values):
            if i < len(indices):
                row, col = indices[i]
                W[row, col] = value
        
        # Điền các vị trí còn lại bằng các giá trị ngẫu nhiên từ tập giá trị cần thiết
        for idx in range(len(required_values), len(indices)):
            row, col = indices[idx]
            W[row, col] = random.choice(required_values)
            
        return W
    
    def embed(self, binary_image, message_bits):
        """
        Nhúng chuỗi bit vào ảnh nhị phân sử dụng thuật toán PCT.
        
        Tham số:
            binary_image (numpy.ndarray): Ảnh nhị phân (giá trị 0 hoặc 1)
            message_bits (list hoặc numpy.ndarray): Chuỗi bit cần giấu
            
        Trả về:
            numpy.ndarray: Ảnh đã được nhúng thủy vân
        """
        # Tạo bản sao của ảnh đầu vào
        watermarked_image = np.copy(binary_image)
        
        # Tính số lượng khối trong ảnh
        height, width = binary_image.shape
        blocks_y = height // self.m
        blocks_x = width // self.n
        
        # Chuyển đổi chuỗi bit thành các đoạn r-bit
        bit_chunks = []
        for i in range(0, len(message_bits), self.r):
            chunk = message_bits[i:i+self.r]
            # Đệm nếu cần thiết
            if len(chunk) < self.r:
                chunk = np.pad(chunk, (0, self.r - len(chunk)), 'constant')
            bit_chunks.append(chunk)
        
        # Xử lý từng khối
        block_idx = 0
        for i in range(blocks_y):
            for j in range(blocks_x):
                if block_idx >= len(bit_chunks):
                    break
                
                # Trích xuất khối hiện tại F
                y_start = i * self.m
                x_start = j * self.n
                F = binary_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Lấy đoạn bit cần giấu
                bit_chunk = bit_chunks[block_idx]
                
                # Chuyển đổi đoạn bit thành giá trị thập phân (b)
                b = sum(bit * (2**(self.r-1-idx)) for idx, bit in enumerate(bit_chunk))
                
                # Áp dụng thuật toán thủy vân cho khối
                F_modified = self._embed_in_block(F, b)
                
                # Thay thế khối trong ảnh đã nhúng thủy vân
                watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n] = F_modified
                
                block_idx += 1
        
        return watermarked_image
    
    def _embed_in_block(self, F, b):
        """
        Nhúng r bit (biểu diễn dưới dạng số thập phân b) vào một khối F
        
        Tham số:
            F (numpy.ndarray): Khối nhị phân kích thước m x n
            b (int): Biểu diễn thập phân của r bit cần giấu
            
        Trả về:
            numpy.ndarray: Khối đã được sửa đổi với các bit ẩn
        """
        # Tạo bản sao của F để làm việc
        F_modified = np.copy(F)
        
        # Bước 1: Tính T = F + K (phép toán XOR)
        T = np.bitwise_xor(F, self.K)
        
        # Bước 2: Tính S = SUM(T * W) (nhân các phần tử tương ứng và tính tổng)
        S = np.sum(T * self.W) % (2**self.r)
        
        # Bước 3: Xây dựng các tập Z
        Z_sets = self._build_z_sets(T)
        
        # Bước 4: Xác định các sửa đổi cần thiết để thỏa mãn S' = b (mod 2^r)
        
        # Tính d = b - S (mod 2^r)
        d = (b - S) % (2**self.r)
        
        if d == 0:
            # Không cần sửa đổi nếu S đã bằng b
            return F_modified
        
        # Trường hợp khi d > 0
        if d > 0:
            if len(Z_sets[d]) > 0:
                # Chỉ thay đổi một bit nếu có thể
                j, k = random.choice(Z_sets[d])
                F_modified[j, k] = 1 - F_modified[j, k]  # Đảo bit
            else:
                # Cần thay đổi hai bit
                h = 1
                while h < 2**self.r:
                    hd = (h * d) % (2**self.r)
                    one_minus_h_d = ((1-h) * d) % (2**self.r)
                    
                    if one_minus_h_d == 0:
                        one_minus_h_d = 2**self.r - 1
                    
                    if len(Z_sets[hd]) > 0 and len(Z_sets[one_minus_h_d]) > 0:
                        # Tìm thấy h hợp lệ
                        j, k = random.choice(Z_sets[hd])
                        u, v = random.choice(Z_sets[one_minus_h_d])
                        
                        # Đảo cả hai bit
                        F_modified[j, k] = 1 - F_modified[j, k]
                        F_modified[u, v] = 1 - F_modified[u, v]
                        break
                    
                    h += 1
        
        # Trường hợp khi d < 0 (hoặc thực tế ta có thể xem như d > 0 trong số học modulo)
        else:
            d_plus_2r = (d + 2**self.r) % (2**self.r)
            
            if len(Z_sets[d_plus_2r]) > 0:
                # Chỉ thay đổi một bit
                j, k = random.choice(Z_sets[d_plus_2r])
                F_modified[j, k] = 1 - F_modified[j, k]  # Đảo bit
            else:
                # Cần thay đổi hai bit
                h = 1
                while h < 2**self.r:
                    hd = (h * d) % (2**self.r)
                    one_minus_h_d_plus_2r = ((1-h) * d + 2**self.r) % (2**self.r)
                    
                    if len(Z_sets[hd]) > 0 and len(Z_sets[one_minus_h_d_plus_2r]) > 0:
                        # Tìm thấy h hợp lệ
                        j, k = random.choice(Z_sets[hd])
                        u, v = random.choice(Z_sets[one_minus_h_d_plus_2r])
                        
                        # Đảo cả hai bit
                        F_modified[j, k] = 1 - F_modified[j, k]
                        F_modified[u, v] = 1 - F_modified[u, v]
                        break
                    
                    h += 1
        
        return F_modified
    
    def _build_z_sets(self, T):
        """
        Xây dựng các tập Z cho tất cả các giá trị có thể từ 1 đến 2^r - 1
        
        Tham số:
            T (numpy.ndarray): Ma trận T (F + K)
            
        Trả về:
            dict: Từ điển với các khóa từ 1 đến 2^r-1, giá trị là danh sách các chỉ số (j,k)
        """
        Z = {alpha: [] for alpha in range(1, 2**self.r)}
        
        for j in range(self.m):
            for k in range(self.n):
                alpha = self.W[j, k]
                if alpha < 2**self.r:
                    if T[j, k] == 0:
                        # Khi T[j,k] = 0, đảo F[j,k] làm tăng S thêm alpha
                        Z[alpha].append((j, k))
                    else:
                        # Khi T[j,k] = 1, đảo F[j,k] làm giảm S đi alpha
                        # Điều đó tương đương với việc tăng S thêm 2^r - alpha (mod 2^r)
                        Z[(2**self.r - alpha) % (2**self.r)].append((j, k))
        
        return Z
    
    def extract(self, watermarked_image):
        """
        Trích xuất thông điệp ẩn từ ảnh đã nhúng thủy vân
        
        Tham số:
            watermarked_image (numpy.ndarray): Ảnh đã nhúng thủy vân
            
        Trả về:
            numpy.ndarray: Các bit thông điệp đã trích xuất
        """
        height, width = watermarked_image.shape
        blocks_y = height // self.m
        blocks_x = width // self.n
        
        extracted_bits = []
        
        for i in range(blocks_y):
            for j in range(blocks_x):
                # Trích xuất khối hiện tại F'
                y_start = i * self.m
                x_start = j * self.n
                F_prime = watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Tính T' = F' + K (phép toán XOR)
                T_prime = np.bitwise_xor(F_prime, self.K)
                
                # Tính S' = SUM(T' * W) (nhân các phần tử tương ứng)
                S_prime = np.sum(T_prime * self.W) % (2**self.r)
                
                # Chuyển đổi S' sang biểu diễn nhị phân (r bit)
                bits = [(S_prime >> i) & 1 for i in range(self.r-1, -1, -1)]
                extracted_bits.extend(bits)
                
        return np.array(extracted_bits)

def load_binary_image(path):
    """
    Đọc ảnh và chuyển đổi thành ảnh nhị phân (0 và 1)
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {path}")
    
    # Chuyển đổi thành ảnh nhị phân (0 và 1)
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary_img

def save_binary_image(path, binary_img):
    """
    Lưu ảnh nhị phân (với giá trị 0 và 1) thành tệp (với giá trị 0 và 255)
    """
    # Chuyển từ 0/1 sang 0/255 để hiển thị đúng
    img_to_save = binary_img * 255
    cv2.imwrite(path, img_to_save)

def calculate_psnr(original, watermarked):
    """
    Tính toán Peak Signal-to-Noise Ratio (PSNR) giữa ảnh gốc và ảnh đã nhúng thủy vân
    
    Tham số:
        original (numpy.ndarray): Ảnh gốc
        watermarked (numpy.ndarray): Ảnh đã nhúng thủy vân
        
    Trả về:
        float: Giá trị PSNR (dB)
    """
    # Chuyển đổi ảnh nhị phân sang dạng 0-255 để tính MSE
    orig_255 = original * 255
    wm_255 = watermarked * 255
    
    # Tính MSE (Mean Squared Error)
    mse = np.mean((orig_255 - wm_255) ** 2)
    if mse == 0:  # Nếu ảnh giống hệt nhau
        return float('inf')
    
    # Giá trị pixel tối đa (255 cho ảnh 8-bit)
    max_pixel = 255.0
    
    # Tính PSNR
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", output_path="PCT/ket_qua_thuy_van.png", block_size_m=8, block_size_n=8, r=3):
    """
    Áp dụng thuật toán thủy vân PCT cho ảnh cover và watermark có sẵn
    
    Tham số:
        cover_path (str): Đường dẫn đến file ảnh gốc
        watermark_path (str): Đường dẫn đến file ảnh thủy vân
        output_path (str): Đường dẫn để lưu ảnh kết quả
        block_size_m (int): Kích thước khối theo chiều cao
        block_size_n (int): Kích thước khối theo chiều rộng
        r (int): Số bit giấu trong mỗi khối
    """
    # Đọc ảnh gốc và chuyển sang nhị phân
    try:
        print("\n--- THÔNG TIN ẢNH GỐC ---")
        cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
        if cover_img is None:
            raise ValueError(f"Không thể đọc file ảnh {cover_path}")
            
        # Chuyển đổi thành ảnh nhị phân (0 và 1)
        _, binary_cover = cv2.threshold(cover_img, 127, 1, cv2.THRESH_BINARY)
        
        # Đảm bảo kích thước ảnh là bội số của kích thước khối
        height, width = binary_cover.shape
        new_height = (height // block_size_m) * block_size_m
        new_width = (width // block_size_n) * block_size_n
        
        print(f"Kích thước ảnh gốc: {width}x{height} pixels")
        
        if new_height != height or new_width != width:
            print(f"Cắt ảnh từ {width}x{height} thành {new_width}x{new_height} để đảm bảo chia khối đều")
            binary_cover = binary_cover[:new_height, :new_width]
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh gốc: {e}")
        return
    
    # Đọc ảnh thủy vân và chuyển thành dãy bit
    try:
        print("\n--- THÔNG TIN ẢNH THỦY VÂN ---")
        watermark_img = cv2.imread(watermark_path)
        if watermark_img is None:
            raise ValueError(f"Không thể đọc file ảnh thủy vân {watermark_path}")
        
        # Hiển thị kích thước ảnh thủy vân
        wm_height, wm_width = watermark_img.shape[:2]
        print(f"Kích thước ảnh thủy vân: {wm_width}x{wm_height} pixels")
        
        # Chuyển thành ảnh xám
        if len(watermark_img.shape) == 3:  # Ảnh màu
            watermark_gray = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
        else:
            watermark_gray = watermark_img
            
        # Chuyển thành ảnh nhị phân
        _, binary_watermark = cv2.threshold(watermark_gray, 127, 1, cv2.THRESH_BINARY)
        
        # Chuyển ma trận thành dãy bit
        watermark_bits = binary_watermark.flatten()
        
        # Tính số khối có thể nhúng trong ảnh gốc
        num_blocks = (new_height // block_size_m) * (new_width // block_size_n)
        
        # Tính số bit có thể nhúng (mỗi khối nhúng r bit)
        max_bits = num_blocks * r
        
        print(f"Số khối ảnh: {num_blocks} (kích thước {block_size_m}x{block_size_n})")
        print(f"Dung lượng tối đa có thể nhúng: {max_bits} bits ({max_bits/8:.0f} bytes)")
        
        # Nếu thông điệp quá dài, cắt bớt
        if len(watermark_bits) > max_bits:
            print(f"Thông điệp thủy vân quá dài ({len(watermark_bits)} bits), cắt xuống {max_bits} bits")
            watermark_bits = watermark_bits[:max_bits]
        else:
            print(f"Thông điệp thủy vân: {len(watermark_bits)} bits ({len(watermark_bits)/8:.0f} bytes)")
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh thủy vân: {e}")
        return
    
    print("\n--- TIẾN HÀNH NHÚNG THỦY VÂN ---")
    
    # Khởi tạo thuật toán thủy vân PCT
    pct = PCTWatermark(block_size_m, block_size_n, r)
    
    # Đo thời gian nhúng thủy vân
    start_time_embed = time.time()
    
    # Nhúng thủy vân
    print("Đang nhúng thủy vân...")
    watermarked_image = pct.embed(binary_cover, watermark_bits)
    
    # Kết thúc đo thời gian nhúng
    embed_time = time.time() - start_time_embed
    print(f"Thời gian nhúng thủy vân: {embed_time:.3f} giây")
    
    # Tính PSNR
    psnr_value = calculate_psnr(binary_cover, watermarked_image)
    print(f"PSNR (Peak Signal-to-Noise Ratio): {psnr_value:.2f} dB")
    
    # Đếm số pixel đã thay đổi
    modified_pixels = np.sum(binary_cover != watermarked_image)
    total_pixels = new_height * new_width
    percent_modified = (modified_pixels / total_pixels) * 100
    print(f"Số pixel đã sửa đổi: {modified_pixels}/{total_pixels} ({percent_modified:.4f}%)")
    
    # Lưu ảnh kết quả
    save_binary_image(output_path, watermarked_image)
    print(f"Đã lưu ảnh thủy vân tại: {output_path}")
    
    # Lưu ảnh để hiển thị sự khác biệt
    diff_image = np.abs(binary_cover - watermarked_image)
    diff_path = "PCT/pixel_thay_doi.png"
    save_binary_image(diff_path, diff_image)
    print(f"Đã lưu ảnh hiển thị các pixel thay đổi tại: {diff_path}")
    
    print("\n--- TRÍCH XUẤT THỦY VÂN ---")
    
    # Đo thời gian trích xuất
    start_time_extract = time.time()
    
    # Trích xuất thủy vân
    extracted_bits = pct.extract(watermarked_image)[:len(watermark_bits)]
    
    # Kết thúc đo thời gian trích xuất
    extract_time = time.time() - start_time_extract
    print(f"Thời gian trích xuất thủy vân: {extract_time:.3f} giây")
    
    # Tính độ chính xác trích xuất
    bit_errors = np.sum(watermark_bits != extracted_bits)
    accuracy = (1 - bit_errors / len(watermark_bits)) * 100
    print(f"Độ chính xác trích xuất: {accuracy:.2f}% ({bit_errors} bit lỗi)")
    bit_error_rate = bit_errors / len(watermark_bits)
    if bit_errors > 0:
        print(f"Tỷ lệ lỗi bit (BER): {bit_error_rate:.6f}")
    
    # Lưu ảnh trích xuất để kiểm tra
    if len(watermark_bits) > 0:
        try:
            # Tạo lại ảnh từ các bit trích xuất
            extracted_shape = binary_watermark.shape
            min_size = min(len(extracted_bits), extracted_shape[0] * extracted_shape[1])
            reshaped_bits = extracted_bits[:min_size].reshape(
                extracted_shape[0], -1)[:extracted_shape[0], :extracted_shape[1]]
            
            # Lưu ảnh trích xuất
            extracted_path = "PCT/thuy_van_trich_xuat.png"
            save_binary_image(extracted_path, reshaped_bits)
            print(f"Đã lưu ảnh thủy vân trích xuất tại: {extracted_path}")
        except Exception as e:
            print(f"Không thể lưu ảnh thủy vân trích xuất: {e}")
    
    # Tạo file báo cáo kết quả
    try:
        report_path = "PCT/bao_cao_ket_qua.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO KẾT QUẢ THỦY VÂN PCT\n")
            f.write("===============================\n\n")
            f.write(f"Ảnh gốc: {cover_path} - {width}x{height} pixels\n")
            f.write(f"Ảnh thủy vân: {watermark_path} - {wm_width}x{wm_height} pixels\n\n")
            
            f.write("THÔNG SỐ THUẬT TOÁN\n")
            f.write(f"- Kích thước khối: {block_size_m}x{block_size_n}\n")
            f.write(f"- Số bit nhúng mỗi khối: {r}\n")
            f.write(f"- Số khối ảnh: {num_blocks}\n")
            f.write(f"- Dung lượng tối đa có thể nhúng: {max_bits} bits ({max_bits/8:.0f} bytes)\n\n")
            
            f.write("KẾT QUẢ\n")
            f.write(f"- Thời gian nhúng thủy vân: {embed_time:.3f} giây\n")
            f.write(f"- Thời gian trích xuất thủy vân: {extract_time:.3f} giây\n")
            f.write(f"- PSNR: {psnr_value:.2f} dB\n")
            f.write(f"- Số pixel đã sửa đổi: {modified_pixels}/{total_pixels} ({percent_modified:.4f}%)\n")
            f.write(f"- Độ chính xác trích xuất: {accuracy:.2f}% ({bit_errors} bit lỗi)\n")
            if bit_errors > 0:
                f.write(f"- Tỷ lệ lỗi bit (BER): {bit_error_rate:.6f}\n")
        
        print(f"\nĐã tạo báo cáo kết quả tại: {report_path}")
    except Exception as e:
        print(f"Không thể tạo báo cáo kết quả: {e}")
    
    return {
        'psnr': psnr_value,
        'embed_time': embed_time,
        'extract_time': extract_time,
        'modified_pixels': modified_pixels,
        'total_pixels': total_pixels,
        'accuracy': accuracy
    }

if __name__ == "__main__":
    # Chạy trực tiếp với file cover.png và watermark.jpg
    print("Thuật toán thủy vân PCT (Chen, Pan, Tseng) cho ảnh nhị phân")
    print("Đang xử lý ảnh cover.png và watermark.jpg...")
    
    # Gọi hàm áp dụng thủy vân
    apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", 
                           output_path="PCT/ket_qua_thuy_van.png", 
                           block_size_m=8, block_size_n=8, r=3)
