import cv2
import numpy as np
import time
import os
import re
import base64
import pickle

class LSBWatermark:
    def __init__(self, secret_key=12345):
        """
        Khởi tạo đối tượng LSB Watermark
        
        Args:
            secret_key: Khóa bí mật dùng để tạo mẫu nhúng
        """
        self.secret_key = secret_key
        # Khởi tạo bộ tạo số ngẫu nhiên với khóa bí mật
        np.random.seed(secret_key)
    
    def generate_embedding_pattern(self, width, height, total_bits):
        """
        Tạo mẫu nhúng dựa trên khóa bí mật
        
        Args:
            width, height: Kích thước ảnh
            total_bits: Tổng số bit cần nhúng
            
        Returns:
            Danh sách các vị trí (x, y) để nhúng dữ liệu
        """
        # Khởi tạo lại seed để đảm bảo tính nhất quán
        np.random.seed(self.secret_key)
        
        # Tạo danh sách tất cả các vị trí pixel
        all_positions = [(x, y) for y in range(height) for x in range(width)]
        
        # Xáo trộn danh sách vị trí dựa trên khóa bí mật
        np.random.shuffle(all_positions)
        
        # Lấy số lượng vị trí cần thiết + thêm một chút để dự phòng
        # +1 cho vị trí lưu cờ
        required_positions = all_positions[:total_bits + 100]
        
        return required_positions
    
    def text_to_bits(self, text):
        """Chuyển đổi văn bản thành chuỗi bit"""
        bits = []
        # Chuyển đổi từng ký tự thành 8 bit
        for char in text:
            # Chuyển đổi ký tự thành giá trị ASCII và sau đó thành nhị phân 8 bit
            binary = format(ord(char), '08b')
            # Thêm từng bit vào danh sách
            for bit in binary:
                bits.append(int(bit))
        
        # Thêm EOF marker (0xFF trong nhị phân)
        for bit in format(255, '08b'):
            bits.append(int(bit))
            
        return bits
    
    def bits_to_text(self, bits):
        """Chuyển đổi chuỗi bit thành văn bản"""
        text = ""
        # Xử lý 8 bit một lần
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) < 8:  # Xử lý byte không đầy đủ ở cuối
                break
            
            # Chuyển đổi byte thành số thập phân
            byte_value = int(''.join(map(str, byte)), 2)
            
            # Kiểm tra EOF marker (0xFF)
            if byte_value == 255:
                break
                
            # Chuyển đổi số thập phân thành ký tự và thêm vào văn bản
            text += chr(byte_value)
            
        return text
    
    def count_bits(self, bits):
        """Đếm số bit 0 và 1 trong chuỗi bit"""
        zeros = bits.count(0)
        ones = bits.count(1)
        return zeros, ones
    
    def embed(self, cover_img, watermark_data, is_image=False):
        """
        Nhúng thủy vân vào ảnh sử dụng kỹ thuật LSB
        
        Args:
            cover_img: Ảnh gốc (NumPy array)
            watermark_data: Dữ liệu thủy vân (văn bản hoặc ảnh đã mã hóa)
            is_image: True nếu watermark_data là dữ liệu ảnh đã mã hóa
            
        Returns:
            stego_img: Ảnh đã nhúng thủy vân
            stats: Thống kê về quá trình nhúng
        """
        # Tạo bản sao của ảnh gốc để nhúng
        stego_img = cover_img.copy()
        height, width, _ = stego_img.shape
        
        # Chuẩn bị dữ liệu để nhúng
        if is_image:
            # Thêm tiền tố để đánh dấu dữ liệu là ảnh
            watermark_text = "IMG:" + watermark_data
        else:
            watermark_text = "TXT:" + watermark_data
        
        # Chuyển đổi văn bản thành chuỗi bit
        watermark_bits = self.text_to_bits(watermark_text)
        total_bits = len(watermark_bits)
        
        # Kiểm tra xem ảnh có đủ pixel để nhúng toàn bộ thủy vân
        if height * width < total_bits:
            raise ValueError("Ảnh quá nhỏ để nhúng toàn bộ thủy vân.")
        
        # Đếm số bit 0 và 1 trong chuỗi bit thủy vân
        i0, i1 = self.count_bits(watermark_bits)
        
        # Tạo mẫu nhúng dựa trên khóa bí mật
        embedding_positions = self.generate_embedding_pattern(width, height, total_bits)
        
        # Đọc LSB từ các vị trí sẽ nhúng
        blue_channel_lsb = []
        for x, y in embedding_positions[:total_bits]:
            # Lấy LSB từ kênh xanh da trời
            lsb = stego_img[y, x, 0] & 1  # Blue ở vị trí 0 trong OpenCV (BGR)
            blue_channel_lsb.append(lsb)
        
        c0 = blue_channel_lsb.count(0)
        c1 = blue_channel_lsb.count(1)
        
        # Xác định giá trị cờ dựa trên số lượng bit
        if ((c0 > c1 and i0 > i1) or (c1 > c0 and i1 > i0)):
            flag = 0  # Không cần đảo bit
        else:
            flag = 1  # Cần đảo bit
        
        # Vị trí đầu tiên dùng để lưu cờ
        flag_pos = embedding_positions[0]
        # Lưu giá trị cờ vào bit thứ hai ít quan trọng nhất của pixel này
        stego_img[flag_pos[1], flag_pos[0], 0] = (stego_img[flag_pos[1], flag_pos[0], 0] & 0xFD) | (flag << 1)
        
        # Đếm số pixel đã sửa đổi
        modified_pixels = 0
        
        # Nhúng các bit thủy vân - bắt đầu từ vị trí thứ 2 (sau vị trí lưu cờ)
        for i, bit in enumerate(watermark_bits):
            pos = embedding_positions[i + 1]  # +1 vì vị trí đầu tiên là để lưu cờ
            x, y = pos
            
            # Lấy bit hiện tại để nhúng
            current_bit = bit
            
            # Áp dụng cờ (đảo bit nếu cờ là 1)
            if flag == 1:
                current_bit = 1 - current_bit
            
            # Lấy giá trị LSB hiện tại của pixel
            current_lsb = stego_img[y, x, 0] & 1
            
            # Chỉ sửa đổi nếu LSB hiện tại khác với bit cần nhúng
            if current_lsb != current_bit:
                # Sửa đổi LSB của kênh xanh da trời
                stego_img[y, x, 0] = (stego_img[y, x, 0] & 0xFE) | current_bit
                modified_pixels += 1
        
        # Tính tổng số pixel trong ảnh
        total_pixels = height * width
        
        # Trả về ảnh đã nhúng thủy vân và thống kê
        return stego_img, {
            "total_bits": total_bits,
            "modified_pixels": modified_pixels,
            "total_pixels": total_pixels,
            "original_bits": {"zeros": i0, "ones": i1},
            "cover_lsb": {"zeros": c0, "ones": c1},
            "flag": flag,
            "is_image": is_image
        }
    
    def extract(self, stego_img):
        """
        Trích xuất thủy vân từ ảnh đã nhúng
        
        Args:
            stego_img: Ảnh đã nhúng thủy vân
            
        Returns:
            extracted_data: Dữ liệu thủy vân đã trích xuất
            is_image: True nếu dữ liệu trích xuất là ảnh
        """
        height, width, _ = stego_img.shape
        
        # Tạo mẫu nhúng dựa trên khóa bí mật - cần tạo lại giống như lúc nhúng
        total_positions_needed = height * width  # Tạo đủ vị trí để đảm bảo tìm được đầy đủ dữ liệu
        embedding_positions = self.generate_embedding_pattern(width, height, total_positions_needed)
        
        # Vị trí đầu tiên chứa cờ
        flag_pos = embedding_positions[0]
        # Đọc giá trị cờ từ bit thứ hai ít quan trọng nhất của pixel
        flag = (stego_img[flag_pos[1], flag_pos[0], 0] >> 1) & 1
        
        # Trích xuất các bit từ LSB của kênh xanh da trời theo mẫu nhúng
        extracted_bits = []
        
        # Khởi tạo biến để lưu trữ chuỗi 8 bit
        current_byte = []
        text_ended = False
        
        # Bắt đầu từ vị trí thứ 2 (sau vị trí lưu cờ)
        position_index = 1
        
        while position_index < len(embedding_positions) and not text_ended:
            pos = embedding_positions[position_index]
            x, y = pos
            
            # Trích xuất LSB từ kênh xanh da trời
            bit = stego_img[y, x, 0] & 1
            
            # Áp dụng cờ (đảo bit nếu cờ là 1)
            if flag == 1:
                bit = 1 - bit
                
            current_byte.append(bit)
            
            # Kiểm tra xem đã có đủ 8 bit chưa
            if len(current_byte) == 8:
                # Chuyển byte thành số thập phân
                byte_value = int(''.join(map(str, current_byte)), 2)
                
                # Kiểm tra EOF marker (0xFF)
                if byte_value == 255:
                    text_ended = True
                else:
                    # Thêm byte này vào chuỗi bit đã trích xuất
                    extracted_bits.extend(current_byte)
                
                current_byte = []
            
            position_index += 1
        
        # Chuyển đổi chuỗi bit đã trích xuất thành văn bản
        extracted_text = self.bits_to_text(extracted_bits)
        
        # Kiểm tra xem dữ liệu có phải là ảnh hay không
        if extracted_text.startswith("IMG:"):
            # Dữ liệu ảnh
            return extracted_text[4:], True
        elif extracted_text.startswith("TXT:"):
            # Dữ liệu văn bản
            return extracted_text[4:], False
        else:
            # Định dạng không rõ, coi như văn bản
            return extracted_text, False

def encode_image_to_base64(image_path):
    """Mã hóa ảnh thành chuỗi base64"""
    try:
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Không thể đọc ảnh thủy vân"
        
        # Resize ảnh nếu quá lớn
        max_size = 128  # Tăng kích thước tối đa để giữ chất lượng tốt hơn
        h, w = img.shape[:2]
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)  # Sử dụng INTER_AREA để giảm artifacts
        
        # Chuyển đổi ảnh thành mảng byte với chất lượng cao hơn
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]  # Nén PNG với chất lượng tối đa
        _, buffer = cv2.imencode('.png', img, encode_param)
        
        # Mã hóa base64
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        # Lưu thông tin về kích thước ảnh
        h, w = img.shape[:2]
        encoded_data = f"{w},{h},{base64_str}"
        
        return encoded_data
    except Exception as e:
        return f"Error: {str(e)}"

def decode_base64_to_image(encoded_data):
    """Giải mã chuỗi base64 thành ảnh"""
    try:
        # Tách kích thước và dữ liệu
        parts = encoded_data.split(',', 2)
        if len(parts) != 3:
            raise ValueError("Định dạng dữ liệu không hợp lệ")
        
        width = int(parts[0])
        height = int(parts[1])
        base64_str = parts[2]
        
        # Giải mã base64
        img_data = base64.b64decode(base64_str)
        
        # Chuyển đổi từ mảng byte thành ảnh
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Không thể giải mã dữ liệu ảnh")
        
        # Resize lại kích thước gốc nếu cần với chất lượng tốt hơn
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)  # Sử dụng INTER_CUBIC để tăng chất lượng
        
        return img
    except Exception as e:
        # Nếu có lỗi, tạo ảnh với thông báo lỗi
        error_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(error_img, f"Lỗi giải mã: {str(e)}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return error_img

def apply_watermark_to_files(cover_path, watermark_path=None, watermark_text=None, output_path=None, secret_key=12345, block_size_m=8, block_size_n=8):
    """
    Áp dụng thủy vân LSB từ file ảnh gốc và lưu kết quả
    
    Args:
        cover_path: Đường dẫn đến ảnh gốc
        watermark_path: Đường dẫn đến ảnh thủy vân (nếu dùng ảnh)
        watermark_text: Văn bản cần nhúng làm thủy vân (ưu tiên dùng nếu cung cấp)
        output_path: Đường dẫn để lưu ảnh đã nhúng thủy vân
        secret_key: Khóa bí mật dùng để tạo mẫu nhúng
        block_size_m, block_size_n: Tham số giữ nguyên để tương thích với giao diện (không sử dụng)
        
    Returns:
        dict: Thống kê về quá trình nhúng và trích xuất
    """
    # Tạo thư mục nếu cần
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Khởi tạo đối tượng LSB Watermark
    lsb = LSBWatermark(secret_key)
    
    # Đọc ảnh gốc
    cover_img = cv2.imread(cover_path)
    if cover_img is None:
        raise ValueError(f"Không thể đọc ảnh gốc: {cover_path}")
    
    # Xác định loại thủy vân và dữ liệu
    is_image_watermark = False
    watermark_data = None
    original_watermark_path = None
    
    if watermark_text is None and watermark_path is not None:
        # Thử đọc như ảnh
        if os.path.exists(watermark_path) and watermark_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Lưu đường dẫn gốc
            original_watermark_path = watermark_path
            # Mã hóa ảnh thủy vân thành chuỗi base64
            watermark_data = encode_image_to_base64(watermark_path)
            if not watermark_data.startswith("Error:"):
                is_image_watermark = True
                print(f"Nhúng ảnh thủy vân: {watermark_path}")
            else:
                # Nếu không đọc được ảnh, dùng đường dẫn làm văn bản
                watermark_data = f"Watermark from {os.path.basename(watermark_path)}"
        else:
            # Thử đọc như văn bản
            try:
                with open(watermark_path, 'r', encoding='utf-8') as f:
                    watermark_data = f.read()
            except:
                watermark_data = f"Watermark from {os.path.basename(watermark_path)}"
    elif watermark_text is not None:
        watermark_data = watermark_text
    else:
        watermark_data = "Default watermark text"
    
    # Đo thời gian nhúng
    start_time = time.time()
    
    # Nhúng thủy vân
    stego_img, stats = lsb.embed(cover_img, watermark_data, is_image=is_image_watermark)
    
    # Ghi lại thời gian nhúng
    embed_time = time.time() - start_time
    
    # Lưu ảnh đã nhúng thủy vân
    cv2.imwrite(output_path, stego_img)
    
    # Tính PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((cover_img.astype(np.float64) - stego_img.astype(np.float64)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255.0 ** 2) / mse)
    
    # Đo thời gian trích xuất
    start_time = time.time()
    
    # Trích xuất thủy vân
    extracted_data, is_extracted_image = lsb.extract(stego_img)
    
    # Ghi lại thời gian trích xuất
    extract_time = time.time() - start_time
    
    # Tạo ảnh thủy vân trích xuất
    extracted_watermark_path = os.path.join(os.path.dirname(output_path), "thuy_van_trich_xuat.png")
    
    if is_extracted_image:
        # Giải mã dữ liệu ảnh
        extracted_img = decode_base64_to_image(extracted_data)
        
        # Lưu ảnh với chất lượng cao
        cv2.imwrite(extracted_watermark_path, extracted_img, 
                   [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Chất lượng PNG tối đa
        
        # Nếu có ảnh gốc, tính toán độ tương đồng
        similarity = "N/A"
        if original_watermark_path and os.path.exists(original_watermark_path):
            try:
                original_img = cv2.imread(original_watermark_path)
                # Resize về cùng kích thước để so sánh
                if original_img.shape != extracted_img.shape:
                    original_img = cv2.resize(original_img, 
                                            (extracted_img.shape[1], extracted_img.shape[0]),
                                            interpolation=cv2.INTER_AREA)
                # Tính độ tương đồng cấu trúc
                if 'compareSSIM' in dir(cv2):  # OpenCV 3.x
                    similarity = cv2.compareSSIM(original_img, extracted_img)
                else:  # OpenCV 4.x có thể có API khác
                    from skimage.metrics import structural_similarity as ssim
                    # Chuyển sang grayscale để so sánh
                    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                    extracted_gray = cv2.cvtColor(extracted_img, cv2.COLOR_BGR2GRAY)
                    similarity = ssim(orig_gray, extracted_gray)
                similarity = f"{similarity:.2f}"
            except Exception as e:
                print(f"Không thể tính độ tương đồng ảnh: {e}")
        
        extracted_display = f"Ảnh (đã lưu tại {extracted_watermark_path}, độ tương đồng: {similarity})"
    else:
        # Tạo ảnh hiển thị văn bản trích xuất
        text_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        
        # Chia văn bản thành các dòng
        text_lines = []
        text_so_far = ""
        for word in extracted_data.split():
            test_text = text_so_far + " " + word if text_so_far else word
            text_size = cv2.getTextSize(test_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            if text_size[0] > 580:
                text_lines.append(text_so_far)
                text_so_far = word
            else:
                text_so_far = test_text
        if text_so_far:
            text_lines.append(text_so_far)
        
        # Vẽ văn bản
        for i, line in enumerate(text_lines):
            cv2.putText(text_img, line, (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imwrite(extracted_watermark_path, text_img)
        extracted_display = extracted_data
    
    # Tính độ chính xác (chỉ áp dụng cho văn bản)
    accuracy = 100.0
    error_bits = 0
    
    if not is_image_watermark and not is_extracted_image:
        # Đối với văn bản, so sánh bit-by-bit
        original_bits = lsb.text_to_bits("TXT:" + watermark_data)[:len(watermark_data)*8]
        extracted_bits = lsb.text_to_bits("TXT:" + extracted_data)[:len(extracted_data)*8]
        
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len > 0:
            correct_bits = sum(1 for i in range(min_len) if original_bits[i] == extracted_bits[i])
            accuracy = (correct_bits / min_len * 100)
            error_bits = min_len - correct_bits
    
    # Tạo báo cáo
    watermark_display = "Ảnh thủy vân" if is_image_watermark else watermark_data[:50] + "..." if len(watermark_data) > 50 else watermark_data
    
    report = f"""
Báo cáo kết quả nhúng thủy vân LSB
===================================

Thông tin đầu vào:
- Ảnh gốc: {cover_path}
- Kích thước ảnh: {cover_img.shape[1]}x{cover_img.shape[0]} pixels
- Thủy vân: {watermark_display}
- Loại thủy vân: {"Ảnh" if is_image_watermark else "Văn bản"}
- Khóa bí mật: {secret_key}

Thống kê quá trình nhúng:
- Cờ đảo bit: {stats['flag']}
- Tổng số bit thủy vân: {stats['total_bits']}
- Phân bố bit trong thông điệp gốc: {stats['original_bits']['zeros']} bit 0, {stats['original_bits']['ones']} bit 1
- Phân bố LSB trong ảnh gốc: {stats['cover_lsb']['zeros']} bit 0, {stats['cover_lsb']['ones']} bit 1

Kết quả:
- Ảnh đã nhúng thủy vân: {output_path}
- PSNR: {psnr:.2f} dB
- Số pixel đã sửa đổi: {stats['modified_pixels']}/{stats['total_pixels']} ({stats['modified_pixels']/stats['total_pixels']*100:.4f}%)

Trích xuất thủy vân:
- Thủy vân trích xuất: {extracted_display}
- Loại thủy vân trích xuất: {"Ảnh" if is_extracted_image else "Văn bản"}
- Độ chính xác trích xuất: {accuracy:.2f}% ({error_bits} bit lỗi)

Hiệu suất:
- Thời gian nhúng thủy vân: {embed_time:.6f} giây
- Thời gian trích xuất thủy vân: {extract_time:.6f} giây

===================================
"""
    
    # Lưu báo cáo
    report_path = os.path.join(os.path.dirname(output_path), "bao_cao_ket_qua.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # In báo cáo
    print(report)
    
    # Trả về thống kê để hiển thị trong giao diện
    return {
        "psnr": psnr,
        "embed_time": embed_time,
        "extract_time": extract_time,
        "modified_pixels": stats['modified_pixels'],
        "total_pixels": stats['total_pixels'],
        "accuracy": accuracy,
        "error_bits": error_bits,
        "is_image": is_image_watermark
    }

# Test module nếu chạy trực tiếp
if __name__ == "__main__":
    cover_image = "original.png"  # Đường dẫn đến ảnh gốc
    watermark = "Đây là thông điệp thủy vân bí mật!"  # Văn bản cần nhúng
    output_dir = "LSB"
    
    # Đảm bảo thư mục tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    stego_image = os.path.join(output_dir, "ket_qua_thuy_van.png")  # Đường dẫn để lưu ảnh đã nhúng
    
    # Nhúng thủy vân
    result = apply_watermark_to_files(cover_image, watermark_text=watermark, output_path=stego_image)
    print(f"PSNR: {result['psnr']:.2f} dB")
    print(f"Độ chính xác trích xuất: {result['accuracy']:.2f}%")
