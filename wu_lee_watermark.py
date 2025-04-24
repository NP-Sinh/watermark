import numpy as np
import cv2
import math
import random
import time

class WuLeeWatermark:
    """
    Thuật toán thủy vân Wu-Lee (M.Y. Wu và J.H. Lee) cho ảnh nhị phân.
    Thuật toán này sử dụng khóa bí mật K để tăng tính bảo mật và cải tiến hiệu quả nhúng.
    """
    
    def __init__(self, block_size_m, block_size_n, secret_key=None, alpha=5.0):
        """
        Khởi tạo thuật toán thủy vân Wu-Lee.
        
        Tham số:
            block_size_m (int): Chiều cao khối
            block_size_n (int): Chiều rộng khối
            secret_key (int, optional): Số nguyên sử dụng để tạo khóa bí mật
            alpha (float, optional): Tham số alpha điều chỉnh độ mạnh của việc nhúng
        """
        self.m = block_size_m
        self.n = block_size_n
        
        # Tạo khóa bí mật K
        if secret_key is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(secret_key)
            
        # Tạo ma trận khóa K kích thước m x n với tỷ lệ bit 1 khoảng 40-60%
        density = 0.4 + (alpha / 20)  # Điều chỉnh mật độ bit 1 trong khóa dựa trên alpha
        density = min(max(density, 0.3), 0.7)  # Giới hạn trong khoảng 30-70%
        
        self.K = np.random.choice([0, 1], size=(block_size_m, block_size_n), p=[1-density, density]).astype(np.uint8)
        self.sum_K = np.sum(self.K)
        
        # Đảm bảo ma trận K không toàn 0 hoặc toàn 1
        if self.sum_K == 0:
            self.K[0, 0] = 1
            self.sum_K = 1
        elif self.sum_K == block_size_m * block_size_n:
            self.K[0, 0] = 0
            self.sum_K -= 1
            
        self.alpha = alpha
        
    def embed(self, binary_image, message_bits):
        """
        Nhúng chuỗi bit vào ảnh nhị phân sử dụng thuật toán Wu-Lee.
        
        Tham số:
            binary_image (numpy.ndarray): Ảnh nhị phân (giá trị 0 hoặc 1)
            message_bits (list hoặc numpy.ndarray): Chuỗi bit cần nhúng
            
        Trả về:
            numpy.ndarray: Ảnh đã được nhúng thủy vân
            int: Số bit đã nhúng thành công
        """
        # Tạo bản sao của ảnh đầu vào
        watermarked_image = np.copy(binary_image)
        
        # Tính số lượng khối trong ảnh
        height, width = binary_image.shape
        blocks_y = height // self.m
        blocks_x = width // self.n
        total_blocks = blocks_y * blocks_x
        
        # Chuẩn bị message_bits
        message_bits = np.array(message_bits, dtype=np.uint8)
        
        # Biến theo dõi số bit đã nhúng thành công
        embedded_bits = 0
        bit_index = 0
        
        # Xử lý từng khối
        for i in range(blocks_y):
            for j in range(blocks_x):
                if bit_index >= len(message_bits):
                    break
                
                # Trích xuất khối hiện tại Fi
                y_start = i * self.m
                x_start = j * self.n
                Fi = binary_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Kiểm tra điều kiện nhúng (0 < SUM(Fi ^ K) < SUM(K))
                Fi_xor_K = np.bitwise_xor(Fi, self.K)
                sum_Fi_xor_K = np.sum(Fi_xor_K)
                
                if 0 < sum_Fi_xor_K and sum_Fi_xor_K < self.sum_K:
                    # Lấy bit cần nhúng
                    bit_to_embed = message_bits[bit_index]
                    
                    # Nhúng bit vào khối
                    modified_block = self._embed_in_block(Fi, sum_Fi_xor_K, bit_to_embed)
                    
                    # Cập nhật khối trong ảnh đã nhúng
                    watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n] = modified_block
                    
                    # Tăng số bit đã nhúng và chỉ số bit
                    embedded_bits += 1
                    bit_index += 1
                
        return watermarked_image, embedded_bits
    
    def _embed_in_block(self, Fi, sum_Fi_xor_K, bit):
        """
        Nhúng 1 bit vào một khối Fi theo thuật toán Wu-Lee
        
        Tham số:
            Fi (numpy.ndarray): Khối ảnh kích thước m x n
            sum_Fi_xor_K (int): Tổng các bit trong Fi ^ K
            bit (int): Bit cần nhúng (0 hoặc 1)
            
        Trả về:
            numpy.ndarray: Khối đã được sửa đổi để nhúng bit
        """
        # Tạo bản sao của khối để làm việc
        Fi_modified = np.copy(Fi)
        
        # Kiểm tra trường hợp 1: bit đã thỏa mãn, không cần thay đổi
        if sum_Fi_xor_K % 2 == bit:
            return Fi_modified
        
        # Các trường hợp còn lại cần thay đổi 1 bit
        Fi_xor_K = np.bitwise_xor(Fi, self.K)
        
        # Trường hợp 2: SUM(Fi ^ K) = 1
        if sum_Fi_xor_K == 1:
            # Tìm vị trí (j,k) mà Fi(j,k)=0 và K(j,k)=1 để đảo bit
            positions = []
            for j in range(self.m):
                for k in range(self.n):
                    if Fi[j, k] == 0 and self.K[j, k] == 1:
                        positions.append((j, k))
            
            if positions:
                j, k = random.choice(positions)
                Fi_modified[j, k] = 1
        
        # Trường hợp 3: SUM(Fi ^ K) = SUM(K) - 1
        elif sum_Fi_xor_K == self.sum_K - 1:
            # Tìm vị trí (j,k) mà Fi(j,k)=1 và K(j,k)=1 để đảo bit
            positions = []
            for j in range(self.m):
                for k in range(self.n):
                    if Fi[j, k] == 1 and self.K[j, k] == 1:
                        positions.append((j, k))
            
            if positions:
                j, k = random.choice(positions)
                Fi_modified[j, k] = 0
        
        # Trường hợp 4: 1 < SUM(Fi ^ K) < SUM(K) - 1
        else:
            # Tìm vị trí (j,k) mà K(j,k)=1 để đảo bit Fi(j,k)
            positions = []
            for j in range(self.m):
                for k in range(self.n):
                    if self.K[j, k] == 1:
                        positions.append((j, k))
            
            if positions:
                j, k = random.choice(positions)
                Fi_modified[j, k] = 1 - Fi_modified[j, k]  # Đảo bit
        
        return Fi_modified
    
    def extract(self, watermarked_image):
        """
        Trích xuất thông điệp ẩn từ ảnh đã nhúng thủy vân
        
        Tham số:
            watermarked_image (numpy.ndarray): Ảnh đã nhúng thủy vân
            
        Trả về:
            numpy.ndarray: Các bit thông điệp đã trích xuất
        """
        # Tính số lượng khối trong ảnh
        height, width = watermarked_image.shape
        blocks_y = height // self.m
        blocks_x = width // self.n
        
        extracted_bits = []
        
        # Xử lý từng khối
        for i in range(blocks_y):
            for j in range(blocks_x):
                # Trích xuất khối hiện tại Fi'
                y_start = i * self.m
                x_start = j * self.n
                Fi_prime = watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Kiểm tra điều kiện nhúng (0 < SUM(Fi' ^ K) < SUM(K))
                Fi_xor_K = np.bitwise_xor(Fi_prime, self.K)
                sum_Fi_xor_K = np.sum(Fi_xor_K)
                
                if 0 < sum_Fi_xor_K and sum_Fi_xor_K < self.sum_K:
                    # Trích xuất bit: b = [SUM(Fi' ^ K)] mod 2
                    bit = sum_Fi_xor_K % 2
                    extracted_bits.append(bit)
        
        return np.array(extracted_bits, dtype=np.uint8)

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

def apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", output_path="WU_LEE/ket_qua_thuy_van.png", block_size_m=8, block_size_n=8, secret_key=77337, alpha=5.0):
    """
    Áp dụng thuật toán thủy vân Wu-Lee cho ảnh cover và watermark có sẵn
    
    Tham số:
        cover_path (str): Đường dẫn đến file ảnh gốc
        watermark_path (str): Đường dẫn đến file ảnh thủy vân
        output_path (str): Đường dẫn để lưu ảnh kết quả
        block_size_m (int): Kích thước khối theo chiều cao
        block_size_n (int): Kích thước khối theo chiều rộng
        secret_key (int): Khóa bí mật dùng để tạo ma trận K
        alpha (float): Tham số điều chỉnh độ mạnh của thuật toán
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
        blocks_y = new_height // block_size_m
        blocks_x = new_width // block_size_n
        num_blocks = blocks_y * blocks_x
        
        # Ước tính số bit có thể nhúng (không phải tất cả khối đều có thể nhúng bit)
        est_embedding_capacity = int(num_blocks * 0.7)  # Giả sử khoảng 70% khối có thể nhúng
        
        print(f"Số khối ảnh: {num_blocks} (kích thước {block_size_m}x{block_size_n})")
        print(f"Ước tính dung lượng có thể nhúng: khoảng {est_embedding_capacity} bits")
        
        # Nếu thông điệp quá dài, cắt bớt
        if len(watermark_bits) > est_embedding_capacity:
            print(f"Thông điệp thủy vân quá dài ({len(watermark_bits)} bits), cắt xuống {est_embedding_capacity} bits")
            watermark_bits = watermark_bits[:est_embedding_capacity]
        else:
            print(f"Thông điệp thủy vân: {len(watermark_bits)} bits ({len(watermark_bits)/8:.0f} bytes)")
    
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh thủy vân: {e}")
        return
    
    print("\n--- TIẾN HÀNH NHÚNG THỦY VÂN ---")
    
    # Khởi tạo thuật toán thủy vân Wu-Lee
    wu_lee = WuLeeWatermark(block_size_m, block_size_n, secret_key, alpha)
    
    # Thông tin về ma trận khóa K
    sum_K = np.sum(wu_lee.K)
    print(f"Ma trận khóa K có kích thước {block_size_m}x{block_size_n}, tổng số bit 1: {sum_K}")
    print(f"Tỷ lệ bit 1 trong K: {(sum_K / (block_size_m * block_size_n)) * 100:.2f}%")
    
    # Đo thời gian nhúng thủy vân
    start_time_embed = time.time()
    
    # Nhúng thủy vân
    print("Đang nhúng thủy vân...")
    watermarked_image, embedded_bits = wu_lee.embed(binary_cover, watermark_bits)
    
    # Kết thúc đo thời gian nhúng
    embed_time = time.time() - start_time_embed
    print(f"Thời gian nhúng thủy vân: {embed_time:.3f} giây")
    
    print(f"Số bit đã nhúng thành công: {embedded_bits}/{len(watermark_bits)}")
    embedding_rate = (embedded_bits / len(watermark_bits)) * 100 if len(watermark_bits) > 0 else 0
    print(f"Tỷ lệ nhúng thành công: {embedding_rate:.2f}%")
    
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
    diff_path = "WU_LEE/pixel_thay_doi.png"
    save_binary_image(diff_path, diff_image)
    print(f"Đã lưu ảnh hiển thị các pixel thay đổi tại: {diff_path}")
    
    print("\n--- TRÍCH XUẤT THỦY VÂN ---")
    
    # Đo thời gian trích xuất
    start_time_extract = time.time()
    
    # Trích xuất thủy vân
    extracted_bits = wu_lee.extract(watermarked_image)
    
    # Kết thúc đo thời gian trích xuất
    extract_time = time.time() - start_time_extract
    print(f"Thời gian trích xuất thủy vân: {extract_time:.3f} giây")
    
    # Cắt kết quả trích xuất để khớp với số bit đã nhúng thành công
    extracted_bits = extracted_bits[:embedded_bits]
    
    # So sánh với dữ liệu nhúng ban đầu
    original_bits = watermark_bits[:embedded_bits]
    bit_errors = np.sum(original_bits != extracted_bits)
    accuracy = (1 - bit_errors / len(extracted_bits)) * 100 if len(extracted_bits) > 0 else 0
    print(f"Độ chính xác trích xuất: {accuracy:.2f}% ({bit_errors} bit lỗi)")
    
    bit_error_rate = bit_errors / len(extracted_bits) if len(extracted_bits) > 0 else 0
    if bit_errors > 0:
        print(f"Tỷ lệ lỗi bit (BER): {bit_error_rate:.6f}")
    
    # Lưu ảnh trích xuất để kiểm tra
    if len(extracted_bits) > 0:
        try:
            # Tạo lại ảnh từ các bit trích xuất
            # Lưu ý rằng kích thước có thể không khớp với ảnh thủy vân gốc
            # vì không phải tất cả các bit đều được nhúng thành công
            
            # Tạo ảnh hình chữ nhật gần với kích thước ban đầu
            ext_width = min(wm_width, int(math.sqrt(len(extracted_bits) * wm_width / wm_height)))
            ext_height = min(wm_height, len(extracted_bits) // ext_width + (1 if len(extracted_bits) % ext_width > 0 else 0))
            
            reshaped_bits = np.zeros((ext_height, ext_width), dtype=np.uint8)
            for i in range(min(len(extracted_bits), ext_height * ext_width)):
                row = i // ext_width
                col = i % ext_width
                if row < ext_height and col < ext_width:
                    reshaped_bits[row, col] = extracted_bits[i]
            
            # Lưu ảnh trích xuất
            extracted_path = "WU_LEE/thuy_van_trich_xuat.png"
            save_binary_image(extracted_path, reshaped_bits)
            print(f"Đã lưu ảnh thủy vân trích xuất tại: {extracted_path}")
        except Exception as e:
            print(f"Không thể lưu ảnh thủy vân trích xuất: {e}")
    
    # Tạo file báo cáo kết quả
    try:
        report_path = "WU_LEE/bao_cao_ket_qua.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO KẾT QUẢ THỦY VÂN WU-LEE\n")
            f.write("===============================\n\n")
            f.write(f"Ảnh gốc: {cover_path} - {width}x{height} pixels\n")
            f.write(f"Ảnh thủy vân: {watermark_path} - {wm_width}x{wm_height} pixels\n\n")
            
            f.write("THÔNG SỐ THUẬT TOÁN\n")
            f.write(f"- Kích thước khối: {block_size_m}x{block_size_n}\n")
            f.write(f"- Khóa bí mật: {secret_key}\n")
            f.write(f"- Alpha: {alpha}\n")
            f.write(f"- Tỷ lệ bit 1 trong K: {(sum_K / (block_size_m * block_size_n)) * 100:.2f}%\n")
            f.write(f"- Số khối ảnh: {num_blocks}\n")
            f.write(f"- Ước tính dung lượng nhúng: {est_embedding_capacity} bits\n\n")
            
            f.write("KẾT QUẢ\n")
            f.write(f"- Số bit đã nhúng thành công: {embedded_bits}/{len(watermark_bits)} ({embedding_rate:.2f}%)\n")
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
        'accuracy': accuracy,
        'embedded_bits': embedded_bits
    }

if __name__ == "__main__":
    # Chạy trực tiếp với file cover.png và watermark.jpg
    print("Thuật toán thủy vân Wu-Lee (M.Y. Wu và J.H. Lee) cho ảnh nhị phân")
    print("Đang xử lý ảnh cover.png và watermark.jpg...")
    
    # Gọi hàm áp dụng thủy vân
    apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", 
                           output_path="WU_LEE/ket_qua_thuy_van.png", 
                           block_size_m=8, block_size_n=8, secret_key=77337, alpha=5.0)
