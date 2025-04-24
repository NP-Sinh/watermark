import numpy as np
import cv2
import math
import random
import time

class SWWatermark:
    """
    Thuật toán thủy vân Simple Watermark (SW) cho ảnh nhị phân.
    Chia ảnh thành các khối và nhúng/trích xuất bit dựa vào tính chẵn lẻ của tổng các điểm đen trong khối.
    """
    
    def __init__(self, block_size_m, block_size_n):
        """
        Khởi tạo thuật toán thủy vân SW.
        
        Tham số:
            block_size_m (int): Chiều cao khối
            block_size_n (int): Chiều rộng khối
        """
        self.m = block_size_m
        self.n = block_size_n
            
    def embed(self, binary_image, message_bits):
        """
        Nhúng chuỗi bit vào ảnh nhị phân sử dụng thuật toán SW.
        
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
            if bit_index >= len(message_bits):
                break
                
            for j in range(blocks_x):
                if bit_index >= len(message_bits):
                    break
                
                # Trích xuất khối hiện tại B
                y_start = i * self.m
                x_start = j * self.n
                B = binary_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Lấy bit cần nhúng
                bit_to_embed = message_bits[bit_index]
                
                # Nhúng bit vào khối
                modified_block = self._embed_in_block(B, bit_to_embed)
                
                # Cập nhật khối trong ảnh đã nhúng
                watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n] = modified_block
                
                # Tăng số bit đã nhúng và chỉ số bit
                embedded_bits += 1
                bit_index += 1
                
        return watermarked_image, embedded_bits
    
    def _embed_in_block(self, B, bit):
        """
        Nhúng 1 bit vào một khối B theo thuật toán SW
        
        Tham số:
            B (numpy.ndarray): Khối ảnh kích thước m x n
            bit (int): Bit cần nhúng (0 hoặc 1)
            
        Trả về:
            numpy.ndarray: Khối đã được sửa đổi để nhúng bit
        """
        # Tạo bản sao của khối để làm việc
        B_modified = np.copy(B)
        
        # Tính tổng các điểm đen trong khối
        sum_B = np.sum(B)
        
        # Tính t = SUM[B] mod 2
        t = sum_B % 2
        
        # So sánh tính chẵn lẻ giữa t và bit
        if (t == 0 and bit == 0) or (t == 1 and bit == 1):
            # Cùng tính chẵn lẻ, không cần thay đổi
            return B_modified
        
        # Khác tính chẵn lẻ, cần đảo 1 bit trong B
        # Đếm số điểm đen và điểm trắng
        black_points = sum_B
        white_points = B.size - black_points
        
        # Chính sách đảo bit theo yêu cầu
        if abs(black_points - white_points) <= 5:
            # Số điểm đen và trắng xấp xỉ nhau, chọn ngẫu nhiên 1 bit để đảo
            positions = []
            for y in range(self.m):
                for x in range(self.n):
                    positions.append((y, x))
            
            # Chọn ngẫu nhiên một vị trí
            if positions:
                y, x = random.choice(positions)
                B_modified[y, x] = 1 - B_modified[y, x]  # Đảo bit
                
        elif black_points > white_points:
            # Nhiều điểm đen hơn, sửa điểm đen thành điểm trắng
            black_positions = []
            for y in range(self.m):
                for x in range(self.n):
                    if B[y, x] == 1:  # Điểm đen
                        black_positions.append((y, x))
            
            # Chọn một điểm đen để đảo thành trắng
            if black_positions:
                y, x = self._select_best_point_to_flip(B, black_positions, 1, 0)
                B_modified[y, x] = 0  # Đổi từ đen sang trắng
                
        else:
            # Nhiều điểm trắng hơn, sửa điểm trắng thành điểm đen
            white_positions = []
            for y in range(self.m):
                for x in range(self.n):
                    if B[y, x] == 0:  # Điểm trắng
                        white_positions.append((y, x))
            
            # Chọn một điểm trắng để đảo thành đen
            if white_positions:
                y, x = self._select_best_point_to_flip(B, white_positions, 0, 1)
                B_modified[y, x] = 1  # Đổi từ trắng sang đen
        
        return B_modified
    
    def _select_best_point_to_flip(self, block, positions, from_val, to_val):
        """
        Chọn điểm tốt nhất để đảo giá trị dựa trên tính trơn và liên kết
        
        Tham số:
            block (numpy.ndarray): Khối ảnh
            positions (list): Danh sách các vị trí có thể đảo
            from_val (int): Giá trị hiện tại
            to_val (int): Giá trị sau khi đảo
            
        Trả về:
            tuple: Vị trí (y, x) của điểm tốt nhất để đảo
        """
        if not positions:
            return (0, 0)
            
        # Nếu chỉ có 1 vị trí, trả về vị trí đó
        if len(positions) == 1:
            return positions[0]
            
        # Đánh giá tác động của việc đảo mỗi điểm
        best_score = float('inf')
        best_position = positions[0]
        
        block_height, block_width = block.shape
        
        for y, x in positions:
            # Tạo bản sao của khối để thử đảo
            test_block = np.copy(block)
            test_block[y, x] = to_val
            
            # Tính điểm dựa trên sự mất mát tính trơn
            score = 0
            
            # Kiểm tra 8 điểm xung quanh (cửa sổ 3x3)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if (dy == 0 and dx == 0) or ny < 0 or nx < 0 or ny >= block_height or nx >= block_width:
                        continue
                    
                    # Cộng điểm cho sự khác biệt với các điểm lân cận
                    if test_block[y, x] != test_block[ny, nx]:
                        score += 1
            
            # Nếu điểm thấp hơn, đây là vị trí tốt hơn để đảo
            if score < best_score:
                best_score = score
                best_position = (y, x)
        
        return best_position
    
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
                # Trích xuất khối hiện tại B'
                y_start = i * self.m
                x_start = j * self.n
                B_prime = watermarked_image[y_start:y_start+self.m, x_start:x_start+self.n]
                
                # Tính tổng các điểm đen trong khối
                sum_B_prime = np.sum(B_prime)
                
                # Tính bit dựa trên tính chẵn lẻ của tổng
                extracted_bit = sum_B_prime % 2
                
                extracted_bits.append(extracted_bit)
        
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

def apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", output_path="SW/ket_qua_thuy_van.png", block_size_m=8, block_size_n=8):
    """
    Áp dụng thuật toán thủy vân SW cho ảnh cover và watermark có sẵn
    
    Tham số:
        cover_path (str): Đường dẫn đến file ảnh gốc
        watermark_path (str): Đường dẫn đến file ảnh thủy vân
        output_path (str): Đường dẫn để lưu ảnh kết quả
        block_size_m (int): Kích thước khối theo chiều cao
        block_size_n (int): Kích thước khối theo chiều rộng
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
        
        # Ước tính số bit có thể nhúng
        est_embedding_capacity = num_blocks
        
        print(f"Số khối ảnh: {num_blocks} (kích thước {block_size_m}x{block_size_n})")
        print(f"Ước tính dung lượng có thể nhúng: {est_embedding_capacity} bits")
        
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
    
    # Khởi tạo thuật toán thủy vân SW
    sw = SWWatermark(block_size_m, block_size_n)
    
    # Đo thời gian nhúng thủy vân
    start_time_embed = time.time()
    
    # Nhúng thủy vân
    print("Đang nhúng thủy vân...")
    watermarked_image, embedded_bits = sw.embed(binary_cover, watermark_bits)
    
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
    diff_path = "SW/pixel_thay_doi.png"
    save_binary_image(diff_path, diff_image)
    print(f"Đã lưu ảnh hiển thị các pixel thay đổi tại: {diff_path}")
    
    print("\n--- TRÍCH XUẤT THỦY VÂN ---")
    
    # Đo thời gian trích xuất
    start_time_extract = time.time()
    
    # Trích xuất thủy vân
    extracted_bits = sw.extract(watermarked_image)
    extracted_bits = extracted_bits[:embedded_bits]
    
    # Kết thúc đo thời gian trích xuất
    extract_time = time.time() - start_time_extract
    print(f"Thời gian trích xuất thủy vân: {extract_time:.3f} giây")
    
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
            ext_width = min(wm_width, int(math.sqrt(len(extracted_bits) * wm_width / wm_height)))
            ext_height = min(wm_height, len(extracted_bits) // ext_width + (1 if len(extracted_bits) % ext_width > 0 else 0))
            
            reshaped_bits = np.zeros((ext_height, ext_width), dtype=np.uint8)
            for i in range(min(len(extracted_bits), ext_height * ext_width)):
                row = i // ext_width
                col = i % ext_width
                if row < ext_height and col < ext_width:
                    reshaped_bits[row, col] = extracted_bits[i]
            
            # Lưu ảnh trích xuất
            extracted_path = "SW/thuy_van_trich_xuat.png"
            save_binary_image(extracted_path, reshaped_bits)
            print(f"Đã lưu ảnh thủy vân trích xuất tại: {extracted_path}")
        except Exception as e:
            print(f"Không thể lưu ảnh thủy vân trích xuất: {e}")
    
    # Tạo file báo cáo kết quả
    try:
        report_path = "SW/bao_cao_ket_qua.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("BÁO CÁO KẾT QUẢ THỦY VÂN SW\n")
            f.write("============================\n\n")
            f.write(f"Ảnh gốc: {cover_path} - {width}x{height} pixels\n")
            f.write(f"Ảnh thủy vân: {watermark_path} - {wm_width}x{wm_height} pixels\n\n")
            
            f.write("THÔNG SỐ THUẬT TOÁN\n")
            f.write(f"- Kích thước khối: {block_size_m}x{block_size_n}\n")
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
    print("Thuật toán thủy vân SW (Simple Watermarking) cho ảnh nhị phân")
    print("Đang xử lý ảnh cover.png và watermark.jpg...")
    
    # Gọi hàm áp dụng thủy vân
    apply_watermark_to_files(cover_path="cover.png", watermark_path="watermark.jpg", 
                           output_path="SW/ket_qua_thuy_van.png", 
                           block_size_m=8, block_size_n=8) 