import cv2
import numpy as np
import time
import os
import pywt
from skimage.metrics import structural_similarity as ssim

class DWTWatermark:
    def __init__(self, decomposition_level=2, quantization_step=35):
        """
        Khởi tạo đối tượng DWT Watermark
        
        Args:
            decomposition_level: Số mức phân giải wavelet
            quantization_step: Kích thước bước lượng tử hóa Q
        """
        self.decomposition_level = decomposition_level
        self.quantization_step = quantization_step
        self.wavelet_name = 'haar'  # Sử dụng wavelet Haar cho đơn giản
    
    def embed(self, cover_img, watermark_img):
        """
        Nhúng thủy vân vào ảnh sử dụng kỹ thuật DWT
        
        Args:
            cover_img: Ảnh gốc (NumPy array)
            watermark_img: Ảnh thủy vân nhị phân (NumPy array)
            
        Returns:
            stego_img: Ảnh đã nhúng thủy vân
            stats: Thống kê về quá trình nhúng
        """
        # Tạo bản sao của ảnh gốc để nhúng
        stego_img = cover_img.copy()
        
        # Chuyển sang ảnh grayscale nếu cần
        if len(stego_img.shape) > 2:
            gray_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = stego_img.copy()
        
        # Chuẩn bị ảnh thủy vân nhị phân
        if len(watermark_img.shape) > 2:
            watermark = cv2.cvtColor(watermark_img, cv2.COLOR_BGR2GRAY)
        else:
            watermark = watermark_img.copy()
        
        # Ngưỡng hóa thủy vân để đảm bảo nhị phân (0, 1)
        _, watermark_binary = cv2.threshold(watermark, 127, 1, cv2.THRESH_BINARY)
        
        # Lấy kích thước thủy vân
        watermark_height, watermark_width = watermark_binary.shape
        
        # Biến đổi wavelet n mức
        coeffs = pywt.wavedec2(gray_img, self.wavelet_name, level=self.decomposition_level)
        
        # Lấy dải thông LLn (xấp xỉ băng tần thấp ở mức cao nhất)
        ll_band = coeffs[0].copy()
        
        # Kiểm tra kích thước của băng tần LL
        ll_height, ll_width = ll_band.shape
        
        # Tính số khối có thể có
        blocks_per_row = ll_width // watermark_width
        blocks_per_col = ll_height // watermark_height
        num_blocks = blocks_per_row * blocks_per_col
        
        if num_blocks == 0:
            raise ValueError(f"Kích thước ảnh thủy vân ({watermark_width}x{watermark_height}) quá lớn so với dải thông LL{self.decomposition_level} ({ll_width}x{ll_height})")
        
        print(f"Thông tin nhúng: Dải thông LL{self.decomposition_level} kích thước {ll_width}x{ll_height}")
        print(f"Số khối có thể nhúng: {num_blocks} ({blocks_per_row}x{blocks_per_col})")
        
        modified_pixels = 0
        total_pixels = ll_height * ll_width
        
        # Duyệt qua từng khối và nhúng thủy vân vào mỗi khối
        for block_idx in range(num_blocks):
            block_row = block_idx // blocks_per_row
            block_col = block_idx % blocks_per_row
            
            # Vị trí bắt đầu của khối
            start_row = block_row * watermark_height
            start_col = block_col * watermark_width
            
            # Nhúng thủy vân vào khối hiện tại
            for i in range(watermark_height):
                for j in range(watermark_width):
                    row_idx = start_row + i
                    col_idx = start_col + j
                    
                    if row_idx < ll_height and col_idx < ll_width:
                        # Lượng tử hóa hệ số
                        original_value = ll_band[row_idx, col_idx]
                        m = original_value // self.quantization_step
                        
                        # Áp dụng công thức nhúng thủy vân
                        if watermark_binary[i, j] == 0:
                            new_value = m * self.quantization_step
                        else:
                            new_value = (m * self.quantization_step) + (self.quantization_step // 2)
                        
                        # Chỉ đếm nếu pixel được thay đổi
                        if original_value != new_value:
                            modified_pixels += 1
                            
                        # Cập nhật giá trị
                        ll_band[row_idx, col_idx] = new_value
        
        # Thay thế băng tần LL trong coeffs
        coeffs[0] = ll_band
        
        # Thực hiện biến đổi ngược để lấy ảnh chứa thủy vân
        watermarked_img = pywt.waverec2(coeffs, self.wavelet_name)
        
        # Chuẩn hóa giá trị để phù hợp với định dạng uint8
        watermarked_img = np.clip(watermarked_img, 0, 255).astype(np.uint8)
        
        # Nếu ảnh gốc là ảnh màu, cần thay đổi kênh độ sáng
        if len(stego_img.shape) > 2:
            # Chuyển sang không gian màu YUV (YCrCb)
            stego_yuv = cv2.cvtColor(stego_img, cv2.COLOR_BGR2YUV)
            # Thay thế kênh Y (độ sáng) bằng ảnh đã thủy vân
            stego_yuv[:, :, 0] = watermarked_img
            # Chuyển lại sang RGB
            stego_img = cv2.cvtColor(stego_yuv, cv2.COLOR_YUV2BGR)
        else:
            stego_img = watermarked_img
        
        # Tính PSNR
        mse = np.mean((cover_img.astype(np.float64) - stego_img.astype(np.float64)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10((255.0 ** 2) / mse)
        
        # Trả về ảnh đã nhúng thủy vân và thống kê
        return stego_img, {
            "modified_pixels": modified_pixels,
            "total_pixels": total_pixels,
            "decomposition_level": self.decomposition_level,
            "quantization_step": self.quantization_step,
            "num_blocks": num_blocks,
            "psnr": psnr
        }
    
    def extract(self, stego_img, watermark_size):
        """
        Trích xuất thủy vân từ ảnh đã nhúng
        
        Args:
            stego_img: Ảnh đã nhúng thủy vân
            watermark_size: Kích thước của ảnh thủy vân (width, height)
            
        Returns:
            extracted_watermark: Thủy vân đã trích xuất (ảnh nhị phân)
        """
        # Chuyển sang ảnh grayscale nếu cần
        if len(stego_img.shape) > 2:
            gray_img = cv2.cvtColor(stego_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = stego_img.copy()
        
        # Kích thước thủy vân
        watermark_width, watermark_height = watermark_size
        
        # Biến đổi wavelet n mức
        coeffs = pywt.wavedec2(gray_img, self.wavelet_name, level=self.decomposition_level)
        
        # Lấy dải thông LLn
        ll_band = coeffs[0]
        
        # Kiểm tra kích thước của băng tần LL
        ll_height, ll_width = ll_band.shape
        
        # Tính số khối có thể có
        blocks_per_row = ll_width // watermark_width
        blocks_per_col = ll_height // watermark_height
        num_blocks = blocks_per_row * blocks_per_col
        
        if num_blocks == 0:
            raise ValueError(f"Kích thước ảnh thủy vân ({watermark_width}x{watermark_height}) quá lớn so với dải thông LL{self.decomposition_level} ({ll_width}x{ll_height})")
        
        # Tạo mảng để lưu trữ các thủy vân được trích xuất từ mỗi khối
        all_watermarks = np.zeros((num_blocks, watermark_height, watermark_width), dtype=np.uint8)
        
        # Duyệt qua từng khối và trích xuất thủy vân
        for block_idx in range(num_blocks):
            block_row = block_idx // blocks_per_row
            block_col = block_idx % blocks_per_row
            
            # Vị trí bắt đầu của khối
            start_row = block_row * watermark_height
            start_col = block_col * watermark_width
            
            # Khởi tạo thủy vân cho khối hiện tại
            watermark_block = np.zeros((watermark_height, watermark_width), dtype=np.uint8)
            
            # Trích xuất thủy vân từ khối hiện tại
            for i in range(watermark_height):
                for j in range(watermark_width):
                    row_idx = start_row + i
                    col_idx = start_col + j
                    
                    if row_idx < ll_height and col_idx < ll_width:
                        # Lấy giá trị hệ số
                        coeff_value = ll_band[row_idx, col_idx]
                        
                        # Tính phần dư theo bước lượng tử
                        remainder = coeff_value % self.quantization_step
                        
                        # Xác định bit thủy vân: 0 nếu phần dư gần 0, 1 nếu gần Q/2
                        if remainder < self.quantization_step / 4 or remainder > 3 * self.quantization_step / 4:
                            watermark_bit = 0
                        else:
                            watermark_bit = 1
                        
                        watermark_block[i, j] = watermark_bit
            
            # Lưu thủy vân trích xuất từ khối này
            all_watermarks[block_idx] = watermark_block
        
        # Kết hợp tất cả các thủy vân trích xuất
        combined_watermark = np.zeros((watermark_height, watermark_width), dtype=np.float32)
        
        # Tổng hợp các thủy vân trích xuất từ mỗi khối
        for block_idx in range(num_blocks):
            combined_watermark += all_watermarks[block_idx]
        
        # Ngưỡng hóa để lấy thủy vân cuối cùng
        # Bit được coi là 1 nếu hơn 50% các khối cho bit là 1
        combined_watermark = (combined_watermark > (num_blocks / 2)).astype(np.uint8) * 255
        
        return combined_watermark


def apply_watermark_to_files(cover_path, watermark_path=None, watermark_text=None, output_path=None, 
                            decomposition_level=2, quantization_step=35, block_size_m=8, block_size_n=8):
    """
    Áp dụng thủy vân DWT từ file ảnh gốc và lưu kết quả
    
    Args:
        cover_path: Đường dẫn đến ảnh gốc
        watermark_path: Đường dẫn đến ảnh thủy vân
        watermark_text: Văn bản cần nhúng làm thủy vân (không sử dụng trong DWT)
        output_path: Đường dẫn để lưu ảnh đã nhúng thủy vân
        decomposition_level: Số mức phân giải wavelet
        quantization_step: Kích thước bước lượng tử hóa Q
        block_size_m, block_size_n: Kích thước khối (không dùng trong phương pháp này)
        
    Returns:
        dict: Thống kê về quá trình nhúng và trích xuất
    """
    # Tạo thư mục nếu cần
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tạo đối tượng thủy vân DWT
    dwt = DWTWatermark(decomposition_level=decomposition_level, quantization_step=quantization_step)
    
    # Đọc ảnh gốc
    cover_img = cv2.imread(cover_path)
    if cover_img is None:
        raise ValueError(f"Không thể đọc ảnh gốc: {cover_path}")
    
    # Đọc ảnh thủy vân
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise ValueError(f"Không thể đọc ảnh thủy vân: {watermark_path}")
    
    # Nếu ảnh không phải nhị phân, chuyển sang nhị phân
    _, watermark_binary = cv2.threshold(watermark_img, 127, 255, cv2.THRESH_BINARY)
    
    print(f"Ảnh gốc: {cover_path}, kích thước: {cover_img.shape[1]}x{cover_img.shape[0]}")
    print(f"Thủy vân: {watermark_path}, kích thước: {watermark_binary.shape[1]}x{watermark_binary.shape[0]}")
    
    # Đo thời gian nhúng
    start_time = time.time()
    
    # Nhúng thủy vân
    stego_img, stats = dwt.embed(cover_img, watermark_binary)
    
    # Ghi lại thời gian nhúng
    embed_time = time.time() - start_time
    
    # Lưu ảnh đã nhúng thủy vân
    cv2.imwrite(output_path, stego_img)
    
    # Đo thời gian trích xuất
    start_time = time.time()
    
    # Trích xuất thủy vân
    extracted_watermark = dwt.extract(stego_img, (watermark_binary.shape[1], watermark_binary.shape[0]))
    
    # Ghi lại thời gian trích xuất
    extract_time = time.time() - start_time
    
    # Lưu ảnh thủy vân trích xuất
    extracted_path = os.path.join(os.path.dirname(output_path), "thuy_van_trich_xuat.png")
    cv2.imwrite(extracted_path, extracted_watermark)
    
    # Tính độ chính xác (tỷ lệ pixel khớp giữa thủy vân gốc và thủy vân trích xuất)
    normalized_original = watermark_binary // 255
    normalized_extracted = extracted_watermark // 255
    
    try:
        similarity = ssim(normalized_original, normalized_extracted)
    except Exception:
        # Nếu kích thước không khớp, cố gắng resize
        normalized_extracted_resized = cv2.resize(normalized_extracted, 
                                                (normalized_original.shape[1], normalized_original.shape[0]))
        similarity = ssim(normalized_original, normalized_extracted_resized)
    
    accuracy = similarity * 100
    
    # Tạo báo cáo
    report = f"""
Báo cáo kết quả nhúng thủy vân DWT
===================================

Thông tin đầu vào:
- Ảnh gốc: {cover_path}
- Kích thước ảnh: {cover_img.shape[1]}x{cover_img.shape[0]} pixels
- Thủy vân: {watermark_path}
- Kích thước thủy vân: {watermark_binary.shape[1]}x{watermark_binary.shape[0]} pixels

Tham số:
- Mức phân giải Wavelet: {decomposition_level}
- Kích thước bước lượng tử (Q): {quantization_step}
- Số khối nhúng: {stats['num_blocks']}

Kết quả:
- Ảnh đã nhúng thủy vân: {output_path}
- PSNR: {stats['psnr']:.2f} dB
- Số pixel đã sửa đổi: {stats['modified_pixels']}/{stats['total_pixels']} ({stats['modified_pixels']/stats['total_pixels']*100:.4f}%)
- Thủy vân trích xuất: {extracted_path}
- Độ tương đồng cấu trúc (SSIM): {similarity:.4f}
- Độ chính xác trích xuất: {accuracy:.2f}%

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
        "psnr": stats['psnr'],
        "embed_time": embed_time,
        "extract_time": extract_time,
        "modified_pixels": stats['modified_pixels'],
        "total_pixels": stats['total_pixels'],
        "accuracy": accuracy,
        "num_blocks": stats['num_blocks']
    }
