import cv2
import numpy as np
import time
import os
import math
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct, idct

class DCTWatermark:
    def __init__(self, block_size=8, k_factor=20):
        """
        Khởi tạo đối tượng DCT Watermark
        
        Args:
            block_size: Kích thước khối (mặc định là 8x8)
            k_factor: Hệ số k - quyết định độ mạnh của thủy vân (càng lớn càng bền vững nhưng ảnh chất lượng giảm)
        """
        self.block_size = block_size
        self.k_factor = k_factor
        
        # Vị trí hai hệ số tần số giữa trong khối DCT để nhúng bit (mặc định)
        self.coef1_pos = (4, 3)  # Vị trí hệ số thứ nhất u,v
        self.coef2_pos = (3, 4)  # Vị trí hệ số thứ hai p,q
        
    def apply_dct_to_block(self, block):
        """Áp dụng biến đổi DCT cho một khối ảnh"""
        return dct(dct(block.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')
        
    def apply_idct_to_block(self, dct_block):
        """Áp dụng biến đổi ngược IDCT cho một khối hệ số DCT"""
        return idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    def embed(self, cover_img, watermark_bits):
        """
        Nhúng thủy vân vào ảnh sử dụng phương pháp DCT
        
        Args:
            cover_img: Ảnh gốc (grayscale)
            watermark_bits: Dãy bit cần nhúng
            
        Returns:
            watermarked_img: Ảnh đã nhúng thủy vân
            embedded_blocks: Số khối đã nhúng thủy vân
        """
        # Tạo bản sao của ảnh gốc
        watermarked_img = cover_img.copy()
        height, width = cover_img.shape[:2]
        
        # Đảm bảo ảnh có kích thước là bội của block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Tính kích thước mới
            new_height = (height // self.block_size) * self.block_size
            new_width = (width // self.block_size) * self.block_size
            
            # Cắt ảnh
            watermarked_img = watermarked_img[:new_height, :new_width]
            height, width = new_height, new_width
            
        # Tính số khối
        blocks_y = height // self.block_size
        blocks_x = width // self.block_size
        total_blocks = blocks_y * blocks_x
        
        print(f"Kích thước ảnh: {width}x{height}")
        print(f"Số khối {self.block_size}x{self.block_size}: {total_blocks} ({blocks_x}x{blocks_y})")
        
        # Kiểm tra xem có đủ khối để nhúng toàn bộ thủy vân
        if len(watermark_bits) > total_blocks:
            print(f"Cảnh báo: Chuỗi thủy vân ({len(watermark_bits)} bits) dài hơn số khối có sẵn ({total_blocks})")
            print(f"Chỉ nhúng {total_blocks} bits đầu tiên")
            watermark_bits = watermark_bits[:total_blocks]
        
        # Biến đếm số khối đã nhúng
        embedded_blocks = 0
        modified_pixels = 0
        
        # Duyệt qua từng khối và nhúng thủy vân
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                # Kiểm tra nếu đã nhúng hết bit thủy vân
                if embedded_blocks >= len(watermark_bits):
                    break
                    
                # Lấy bit thủy vân hiện tại
                bit = watermark_bits[embedded_blocks]
                
                # Lấy khối hiện tại
                block = watermarked_img[y:y+self.block_size, x:x+self.block_size]
                
                # Nếu ảnh là ảnh màu, chỉ xử lý kênh độ sáng (Y)
                if len(block.shape) > 2:
                    # Chuyển từ BGR sang YCrCb
                    block_ycrcb = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
                    # Lấy kênh Y
                    block_y = block_ycrcb[:, :, 0]
                    
                    # Áp dụng DCT cho kênh Y
                    dct_block = self.apply_dct_to_block(block_y)
                    
                    # Nhúng bit vào miền tần số giữa
                    original_dct_block = dct_block.copy()
                    dct_block, modified = self._embed_bit_in_block(dct_block, bit)
                    
                    # Nếu khối bị thay đổi
                    if modified:
                        # Áp dụng IDCT
                        idct_block = self.apply_idct_to_block(dct_block)
                        
                        # Đếm số pixel thay đổi
                        for i in range(self.block_size):
                            for j in range(self.block_size):
                                if abs(block_y[i, j] - idct_block[i, j]) > 0.5:
                                    modified_pixels += 1
                        
                        # Cập nhật kênh Y
                        block_ycrcb[:, :, 0] = np.clip(idct_block, 0, 255).astype(np.uint8)
                        
                        # Chuyển lại sang BGR
                        watermarked_block = cv2.cvtColor(block_ycrcb, cv2.COLOR_YCrCb2BGR)
                        
                        # Cập nhật khối trong ảnh
                        watermarked_img[y:y+self.block_size, x:x+self.block_size] = watermarked_block
                        
                        # Tăng số khối đã nhúng
                        embedded_blocks += 1
                    
                else:
                    # Áp dụng DCT
                    dct_block = self.apply_dct_to_block(block)
                    
                    # Nhúng bit vào miền tần số giữa
                    original_dct_block = dct_block.copy()
                    dct_block, modified = self._embed_bit_in_block(dct_block, bit)
                    
                    # Nếu khối bị thay đổi
                    if modified:
                        # Áp dụng IDCT
                        idct_block = self.apply_idct_to_block(dct_block)
                        
                        # Đếm số pixel thay đổi
                        for i in range(self.block_size):
                            for j in range(self.block_size):
                                if abs(block[i, j] - idct_block[i, j]) > 0.5:
                                    modified_pixels += 1
                        
                        # Cập nhật khối trong ảnh
                        watermarked_img[y:y+self.block_size, x:x+self.block_size] = np.clip(idct_block, 0, 255).astype(np.uint8)
                        
                        # Tăng số khối đã nhúng
                        embedded_blocks += 1
        
        print(f"Đã nhúng {embedded_blocks}/{len(watermark_bits)} bits")
        print(f"Số pixel đã thay đổi: {modified_pixels}/{height*width} ({modified_pixels/(height*width)*100:.4f}%)")
        
        return watermarked_img, embedded_blocks, modified_pixels
    
    def _embed_bit_in_block(self, dct_block, bit):
        """
        Nhúng một bit vào khối DCT bằng cách thay đổi hai hệ số
        
        Args:
            dct_block: Khối hệ số DCT
            bit: Bit cần nhúng (0 hoặc 1)
            
        Returns:
            dct_block: Khối hệ số DCT sau khi nhúng
            modified: True nếu khối bị thay đổi, False nếu không
        """
        # Lấy hai hệ số ở vị trí đã chọn
        u, v = self.coef1_pos
        p, q = self.coef2_pos
        
        coef1 = dct_block[u, v]
        coef2 = dct_block[p, q]
        
        modified = False
        
        # Theo thuật toán DCT2 đã mô tả
        if bit == 0:
            # Nếu bit là 0, đảm bảo coef1 < coef2
            if coef1 < coef2:
                # Đã đúng, không cần thay đổi
                pass
            else:
                # Đổi chỗ hai hệ số
                dct_block[u, v] = coef2
                dct_block[p, q] = coef1
                modified = True
        else:  # bit == 1
            # Nếu bit là 1, đảm bảo coef1 > coef2
            if coef1 > coef2:
                # Đã đúng, kiểm tra khoảng cách
                if coef1 - coef2 < self.k_factor:
                    # Tăng khoảng cách
                    dct_block[u, v] += self.k_factor / 2
                    dct_block[p, q] -= self.k_factor / 2
                    modified = True
            else:
                # Cần đổi chỗ và có thể tăng khoảng cách
                if coef2 - coef1 < self.k_factor:
                    # Tăng khoảng cách đồng thời đổi chỗ
                    dct_block[u, v] = coef2 + self.k_factor / 2
                    dct_block[p, q] = coef1 - self.k_factor / 2
                else:
                    # Chỉ cần đổi chỗ
                    dct_block[u, v] = coef2
                    dct_block[p, q] = coef1
                modified = True
                
        return dct_block, modified
    
    def extract(self, watermarked_img, num_bits):
        """
        Trích xuất thủy vân từ ảnh đã nhúng
        
        Args:
            watermarked_img: Ảnh đã nhúng thủy vân
            num_bits: Số bit cần trích xuất
            
        Returns:
            watermark_bits: Dãy bit đã trích xuất
        """
        height, width = watermarked_img.shape[:2]
        
        # Đảm bảo ảnh có kích thước là bội của block_size
        if height % self.block_size != 0 or width % self.block_size != 0:
            # Tính kích thước mới
            new_height = (height // self.block_size) * self.block_size
            new_width = (width // self.block_size) * self.block_size
            
            # Cắt ảnh
            watermarked_img = watermarked_img[:new_height, :new_width]
            height, width = new_height, new_width
        
        # Khởi tạo mảng để lưu các bit đã trích xuất
        extracted_bits = []
        
        # Duyệt qua từng khối và trích xuất thủy vân
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                # Kiểm tra nếu đã trích xuất đủ bit
                if len(extracted_bits) >= num_bits:
                    break
                    
                # Lấy khối hiện tại
                block = watermarked_img[y:y+self.block_size, x:x+self.block_size]
                
                # Nếu ảnh là ảnh màu, chỉ xử lý kênh độ sáng (Y)
                if len(block.shape) > 2:
                    # Chuyển từ BGR sang YCrCb
                    block_ycrcb = cv2.cvtColor(block, cv2.COLOR_BGR2YCrCb)
                    # Lấy kênh Y
                    block_y = block_ycrcb[:, :, 0]
                    
                    # Áp dụng DCT
                    dct_block = self.apply_dct_to_block(block_y)
                    
                    # Trích xuất bit
                    bit = self._extract_bit_from_block(dct_block)
                    extracted_bits.append(bit)
                    
                else:
                    # Áp dụng DCT
                    dct_block = self.apply_dct_to_block(block)
                    
                    # Trích xuất bit
                    bit = self._extract_bit_from_block(dct_block)
                    extracted_bits.append(bit)
        
        return np.array(extracted_bits)
    
    def _extract_bit_from_block(self, dct_block):
        """
        Trích xuất một bit từ khối DCT
        
        Args:
            dct_block: Khối hệ số DCT
            
        Returns:
            bit: Bit đã trích xuất (0 hoặc 1)
        """
        # Lấy hai hệ số ở vị trí đã chọn
        u, v = self.coef1_pos
        p, q = self.coef2_pos
        
        coef1 = dct_block[u, v]
        coef2 = dct_block[p, q]
        
        # So sánh hai hệ số để xác định bit
        if coef1 < coef2:
            return 0
        else:
            return 1

def apply_watermark_to_files(cover_path, watermark_path=None, watermark_text=None, output_path=None, 
                           block_size=8, k_factor=20, coef1_pos=(4, 3), coef2_pos=(3, 4)):
    """
    Áp dụng thủy vân DCT cho ảnh từ file
    
    Args:
        cover_path: Đường dẫn đến ảnh gốc
        watermark_path: Đường dẫn đến ảnh thủy vân
        watermark_text: Văn bản cần nhúng làm thủy vân (không sử dụng)
        output_path: Đường dẫn để lưu ảnh đã nhúng thủy vân
        block_size: Kích thước khối (mặc định là 8)
        k_factor: Hệ số k quyết định độ mạnh của thủy vân
        coef1_pos: Vị trí hệ số thứ nhất trong khối DCT
        coef2_pos: Vị trí hệ số thứ hai trong khối DCT
    
    Returns:
        dict: Thống kê về quá trình nhúng và trích xuất
    """
    start_time = time.time()
    
    # Tạo thư mục nếu cần
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Đọc ảnh gốc
    print(f"\n--- ĐỌC ẢNH GỐC ---")
    cover_img = cv2.imread(cover_path)
    if cover_img is None:
        raise ValueError(f"Không thể đọc ảnh gốc: {cover_path}")
    
    # Hiển thị thông tin ảnh gốc
    height, width = cover_img.shape[:2]
    print(f"Kích thước ảnh gốc: {width}x{height} pixels")
    
    # Chuyển ảnh sang grayscale nếu cần
    if len(cover_img.shape) > 2:
        print("Ảnh gốc là ảnh màu RGB")
    else:
        print("Ảnh gốc là ảnh grayscale")

    # Đọc ảnh thủy vân
    print(f"\n--- ĐỌC ẢNH THỦY VÂN ---")
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise ValueError(f"Không thể đọc ảnh thủy vân: {watermark_path}")
    
    # Chuyển thành ảnh nhị phân
    _, watermark_binary = cv2.threshold(watermark_img, 127, 1, cv2.THRESH_BINARY)
    
    # Hiển thị thông tin ảnh thủy vân
    watermark_height, watermark_width = watermark_binary.shape
    print(f"Kích thước ảnh thủy vân: {watermark_width}x{watermark_height} pixels")
    
    # Chuyển ảnh thủy vân thành dãy bit
    watermark_bits = watermark_binary.flatten()
    
    print(f"Số bit cần nhúng: {len(watermark_bits)}")

    # Khởi tạo đối tượng DCT Watermark
    dct = DCTWatermark(block_size=block_size, k_factor=k_factor)
    dct.coef1_pos = coef1_pos
    dct.coef2_pos = coef2_pos
    
    # Hiển thị thông số
    print(f"\n--- THÔNG SỐ THUẬT TOÁN ---")
    print(f"Kích thước khối: {block_size}x{block_size}")
    print(f"Hệ số k: {k_factor}")
    print(f"Vị trí hệ số 1: {coef1_pos}")
    print(f"Vị trí hệ số 2: {coef2_pos}")
    
    # Nhúng thủy vân
    print(f"\n--- NHÚNG THỦY VÂN ---")
    
    embed_start = time.time()
    watermarked_img, embedded_blocks, modified_pixels = dct.embed(cover_img, watermark_bits)
    embed_time = time.time() - embed_start
    
    print(f"Thời gian nhúng: {embed_time:.3f} giây")
    
    # Lưu ảnh đã nhúng thủy vân
    cv2.imwrite(output_path, watermarked_img)
    
    # Tính PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((cover_img.astype(float) - watermarked_img.astype(float)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10((255.0 ** 2) / mse)
    
    print(f"PSNR: {psnr:.2f} dB")
    
    # Trích xuất thủy vân
    print(f"\n--- TRÍCH XUẤT THỦY VÂN ---")
    
    extract_start = time.time()
    extracted_bits = dct.extract(watermarked_img, embedded_blocks)
    extract_time = time.time() - extract_start
    
    print(f"Thời gian trích xuất: {extract_time:.3f} giây")
    
    # So sánh với thủy vân gốc
    original_bits = watermark_bits[:embedded_blocks]
    bit_errors = np.sum(original_bits != extracted_bits)
    accuracy = 100 * (1 - bit_errors / len(extracted_bits)) if len(extracted_bits) > 0 else 0
    
    print(f"Số bit lỗi: {bit_errors}/{len(extracted_bits)}")
    print(f"Độ chính xác: {accuracy:.2f}%")
    
    # Tạo ảnh nhị phân từ các bit đã trích xuất
    # Cố gắng khôi phục kích thước ban đầu hoặc gần nhất có thể
    extracted_width = min(watermark_width, int(math.sqrt(len(extracted_bits) * watermark_width / watermark_height)))
    extracted_height = min(watermark_height, len(extracted_bits) // extracted_width + (1 if len(extracted_bits) % extracted_width > 0 else 0))
    
    extracted_watermark = np.zeros((extracted_height, extracted_width), dtype=np.uint8)
    
    # Điền các bit vào ảnh trích xuất
    for i in range(min(len(extracted_bits), extracted_height * extracted_width)):
        row = i // extracted_width
        col = i % extracted_width
        if row < extracted_height and col < extracted_width:
            extracted_watermark[row, col] = extracted_bits[i] * 255  # Chuyển từ 0/1 sang 0/255
    
    # Lưu ảnh thủy vân trích xuất
    extracted_path = os.path.join(os.path.dirname(output_path), "thuy_van_trich_xuat.png")
    cv2.imwrite(extracted_path, extracted_watermark)
    
    total_time = time.time() - start_time
    
    # Tạo báo cáo
    report = f"""BÁO CÁO KẾT QUẢ THỦY VÂN DCT
===============================

THÔNG TIN ĐẦU VÀO
-----------------
- Ảnh gốc: {cover_path} - {width}x{height} pixels
- Thủy vân: {watermark_path} - {watermark_width}x{watermark_height} pixels
- Số bit cần nhúng: {len(watermark_bits)}

THÔNG SỐ THUẬT TOÁN
-----------------
- Kích thước khối: {block_size}x{block_size}
- Hệ số k: {k_factor}
- Vị trí hệ số 1: {coef1_pos}
- Vị trí hệ số 2: {coef2_pos}

KẾT QUẢ
-------
- Ảnh thủy vân: {output_path}
- Ảnh thủy vân trích xuất: {extracted_path}
- Số khối đã nhúng: {embedded_blocks}/{len(watermark_bits) // (block_size*block_size) + 1}
- Số bit đã nhúng: {embedded_blocks}/{len(watermark_bits)}
- Số pixel đã sửa đổi: {modified_pixels}/{height*width} ({modified_pixels/(height*width)*100:.4f}%)
- PSNR: {psnr:.2f} dB
- Độ chính xác trích xuất: {accuracy:.2f}% ({bit_errors} bit lỗi)

HIỆU SUẤT
--------
- Thời gian nhúng: {embed_time:.3f} giây
- Thời gian trích xuất: {extract_time:.3f} giây
- Tổng thời gian: {total_time:.3f} giây

===============================
"""
    
    # Lưu báo cáo
    report_path = os.path.join(os.path.dirname(output_path), "bao_cao_ket_qua.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n--- Đã lưu báo cáo tại: {report_path} ---")
    
    # Trả về thống kê để hiển thị trong giao diện
    return {
        "psnr": psnr,
        "embed_time": embed_time,
        "extract_time": extract_time,
        "modified_pixels": modified_pixels,
        "total_pixels": height * width,
        "accuracy": accuracy,
        "embedded_blocks": embedded_blocks
    }

if __name__ == "__main__":
    print("Thuật toán thủy vân DCT (Discrete Cosine Transform)")
    print("Thực hiện nhúng thủy vân vào ảnh sử dụng phép biến đổi cosin rời rạc")
    
    # Kiểm tra tham số đầu vào
    cover_path = "cover.png"
    watermark_path = "watermark.jpg"
    output_path = "DCT/ket_qua_thuy_van.png"
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Gọi hàm áp dụng thủy vân
    apply_watermark_to_files(
        cover_path=cover_path,
        watermark_path=watermark_path,
        output_path=output_path,
        block_size=8,
        k_factor=20
    )
