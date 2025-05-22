import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale
from PIL import Image, ImageTk
import cv2
import numpy as np
from io import StringIO
import re
import threading
import time
import matplotlib.pyplot as plt

# Import phương pháp
from pct_watermark import apply_watermark_to_files as apply_pct_watermark
from wu_lee_watermark import apply_watermark_to_files as apply_wulee_watermark
from sw_watermark import apply_watermark_to_files as apply_sw_watermark
from lsb_watermark import apply_watermark_to_files as apply_lsb_watermark
from dwt_watermark import apply_watermark_to_files as apply_dwt_watermark
from dct_watermark import apply_watermark_to_files as apply_dct_watermark

class WatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Thủy vân Ảnh")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")
        
        # Create necessary directories
        if not os.path.exists("PCT"):
            os.makedirs("PCT")
        if not os.path.exists("WU_LEE"):
            os.makedirs("WU_LEE")
        if not os.path.exists("SW"):
            os.makedirs("SW")
        if not os.path.exists("LSB"):
            os.makedirs("LSB")
        if not os.path.exists("DWT"):
            os.makedirs("DWT")
        if not os.path.exists("DCT"):
            os.makedirs("DCT")
        
        # Variables
        self.algorithm = tk.StringVar(value="PCT")
        self.cover_path = tk.StringVar()
        self.watermark_path = tk.StringVar()
        
        # PCT parameters
        self.pct_block_size_m = tk.IntVar(value=8)
        self.pct_block_size_n = tk.IntVar(value=8)
        self.pct_r_value = tk.IntVar(value=3)
        
        # Wu-Lee parameters
        self.secret_key = tk.IntVar(value=77337)
        self.wulee_block_size_m = tk.IntVar(value=8)
        self.wulee_block_size_n = tk.IntVar(value=8)
        self.alpha = tk.DoubleVar(value=5.0)
        
        # SW parameters
        self.sw_block_size_m = tk.IntVar(value=8)
        self.sw_block_size_n = tk.IntVar(value=8)
        
        # LSB parameters
        self.lsb_block_size_m = tk.IntVar(value=8)
        self.lsb_block_size_n = tk.IntVar(value=8)
        
        # DWT parameters
        self.dwt_decomposition_level = tk.IntVar(value=2)
        self.dwt_quantization_step = tk.IntVar(value=35)
        
        # DCT parameters
        self.dct_block_size = tk.IntVar(value=8)
        self.dct_k_factor = tk.IntVar(value=20)
        
        # Image variables
        self.cover_img = None
        self.watermark_img = None
        self.watermarked_img = None
        self.extracted_watermark = None
        
        # Create main layout
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Điều khiển", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Algorithm selection
        ttk.Label(control_frame, text="Chọn thuật toán:").grid(row=0, column=0, sticky=tk.W, pady=5)
        algorithms = ["PCT", "Wu-Lee", "SW", "LSB", "DWT", "DCT"]
        algorithm_dropdown = ttk.Combobox(control_frame, textvariable=self.algorithm, values=algorithms, state="readonly", width=28)
        algorithm_dropdown.grid(row=0, column=1, columnspan=2, sticky=tk.W, pady=5)
        algorithm_dropdown.bind("<<ComboboxSelected>>", self.on_algorithm_change)
        
        # Image selection
        ttk.Label(control_frame, text="Ảnh gốc:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.cover_path, width=30).grid(row=1, column=1, pady=5)
        ttk.Button(control_frame, text="Chọn", command=self.browse_cover).grid(row=1, column=2, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Ảnh thủy vân:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(control_frame, textvariable=self.watermark_path, width=30).grid(row=2, column=1, pady=5)
        ttk.Button(control_frame, text="Chọn", command=self.browse_watermark).grid(row=2, column=2, padx=5, pady=5)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(control_frame, text="Tham số", padding=10)
        self.params_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Create parameters based on algorithm
        self.create_pct_params()
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=4, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        ttk.Button(action_frame, text="Nhúng thủy vân", command=self.embed_watermark).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Trích xuất thủy vân", command=self.extract_watermark).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Lưu kết quả", command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        # Kết quả đánh giá
        result_frame = ttk.LabelFrame(control_frame, text="Kết quả")
        result_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        # Các biến lưu kết quả
        self.embed_time_var = tk.StringVar(value="-")
        self.extract_time_var = tk.StringVar(value="-")
        self.psnr_var = tk.StringVar(value="-")
        self.modified_pixels_var = tk.StringVar(value="-")
        self.accuracy_var = tk.StringVar(value="-")
        
        # Hiển thị kết quả
        ttk.Label(result_frame, text="Thời gian nhúng:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(result_frame, textvariable=self.embed_time_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(result_frame, text="Thời gian trích xuất:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(result_frame, textvariable=self.extract_time_var).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(result_frame, text="PSNR:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(result_frame, textvariable=self.psnr_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(result_frame, text="Pixel đã sửa đổi:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Label(result_frame, textvariable=self.modified_pixels_var).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(result_frame, text="Độ chính xác:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Label(result_frame, textvariable=self.accuracy_var).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # Right panel for image display
        display_frame = ttk.LabelFrame(main_frame, text="Hiển thị ảnh", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different views
        self.tabs = ttk.Notebook(display_frame)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Images
        self.images_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.images_tab, text="Hình ảnh")
        
        # Tab 2: Report
        self.report_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.report_tab, text="Báo cáo kết quả")
        
        # Configure image frames in the Images tab
        self.setup_image_frames()
        
        # Configure report tab
        self.report_frame = ttk.Frame(self.report_tab)
        self.report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.report_text = tk.Text(self.report_frame, wrap=tk.WORD)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.report_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.config(yscrollcommand=scrollbar.set)
        self.report_text.insert(tk.END, "Báo cáo kết quả sẽ hiển thị ở đây sau khi nhúng và trích xuất thủy vân.")
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=2)
        
    def setup_image_frames(self):
        # Image frames in 2x2 grid
        self.images_tab.grid_rowconfigure(0, weight=1)
        self.images_tab.grid_rowconfigure(1, weight=1)
        self.images_tab.grid_columnconfigure(0, weight=1)
        self.images_tab.grid_columnconfigure(1, weight=1)
        
        # Cover image frame
        self.cover_frame = ttk.LabelFrame(self.images_tab, text="Ảnh gốc")
        self.cover_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.cover_canvas = tk.Canvas(self.cover_frame, width=350, height=250, bg="#eeeeee")
        self.cover_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Watermark image frame
        self.watermark_frame = ttk.LabelFrame(self.images_tab, text="Thủy vân")
        self.watermark_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.watermark_canvas = tk.Canvas(self.watermark_frame, width=350, height=250, bg="#eeeeee")
        self.watermark_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Watermarked image frame
        self.watermarked_frame = ttk.LabelFrame(self.images_tab, text="Ảnh đã nhúng thủy vân")
        self.watermarked_frame.grid(row=1, column=0, padx=5, pady=5, sticky=tk.NSEW)
        self.watermarked_canvas = tk.Canvas(self.watermarked_frame, width=350, height=250, bg="#eeeeee")
        self.watermarked_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Extracted watermark frame
        self.extracted_frame = ttk.LabelFrame(self.images_tab, text="Thủy vân trích xuất")
        self.extracted_frame.grid(row=1, column=1, padx=5, pady=5, sticky=tk.NSEW)
        self.extracted_canvas = tk.Canvas(self.extracted_frame, width=350, height=250, bg="#eeeeee")
        self.extracted_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_pct_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Add PCT parameters
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=0, column=0, sticky=tk.W, pady=5)
        block_size_frame = ttk.Frame(self.params_frame)
        block_size_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.pct_block_size_m, width=3).pack(side=tk.LEFT)
        ttk.Label(block_size_frame, text="×").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.pct_block_size_n, width=3).pack(side=tk.LEFT)
        
        ttk.Label(self.params_frame, text="Bits per Block (r):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=1, to=10, textvariable=self.pct_r_value, width=3).grid(row=1, column=1, sticky=tk.W, pady=5)
        
    def create_wulee_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Add Wu-Lee parameters
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=0, column=0, sticky=tk.W, pady=5)
        block_size_frame = ttk.Frame(self.params_frame)
        block_size_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.wulee_block_size_m, width=3).pack(side=tk.LEFT)
        ttk.Label(block_size_frame, text="×").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.wulee_block_size_n, width=3).pack(side=tk.LEFT)
        
        ttk.Label(self.params_frame, text="Khóa bí mật:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=1, to=999999, textvariable=self.secret_key, width=8).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Alpha:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Scale(self.params_frame, from_=1.0, to=10.0, variable=self.alpha, orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, sticky=tk.W, pady=5)
        
    def create_sw_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Add SW parameters (Simple Watermarking)
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=0, column=0, sticky=tk.W, pady=5)
        block_size_frame = ttk.Frame(self.params_frame)
        block_size_frame.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.sw_block_size_m, width=3).pack(side=tk.LEFT)
        ttk.Label(block_size_frame, text="×").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(block_size_frame, from_=2, to=64, textvariable=self.sw_block_size_n, width=3).pack(side=tk.LEFT)
    
    def create_lsb_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        ttk.Label(self.params_frame, text="Khóa bí mật:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=1, to=999999, textvariable=self.secret_key, width=8).grid(row=0, column=1, sticky=tk.W, pady=5)

    def create_dwt_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        ttk.Label(self.params_frame, text="Mức phân giải:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=1, to=4, textvariable=self.dwt_decomposition_level, width=3).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Bước lượng tử (Q):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=5, to=100, textvariable=self.dwt_quantization_step, width=3).grid(row=1, column=1, sticky=tk.W, pady=5)

    def create_dct_params(self):
        # Clear current parameters
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        ttk.Label(self.params_frame, text="Kích thước khối:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=8, to=16, textvariable=self.dct_block_size, width=3).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(self.params_frame, text="Hệ số k:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(self.params_frame, from_=5, to=50, textvariable=self.dct_k_factor, width=3).grid(row=1, column=1, sticky=tk.W, pady=5)
        
    def on_algorithm_change(self, event=None):
        algorithm = self.algorithm.get()
        if algorithm == "PCT":
            self.create_pct_params()
        elif algorithm == "Wu-Lee":
            self.create_wulee_params()
        elif algorithm == "SW":
            self.create_sw_params()
        elif algorithm == "LSB":
            self.create_lsb_params()
        elif algorithm == "DWT":
            self.create_dwt_params()
        elif algorithm == "DCT":
            self.create_dct_params()
            
    def browse_cover(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Tệp ảnh", "*.png;*.jpg;*.jpeg;*.bmp"), ("Tất cả tệp", "*.*")])
        if file_path:
            self.cover_path.set(file_path)
            self.load_cover_image()
    
    def browse_watermark(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Tệp ảnh", "*.png;*.jpg;*.jpeg;*.bmp"), ("Tất cả tệp", "*.*")])
        if file_path:
            self.watermark_path.set(file_path)
            self.load_watermark_image()
    
    def load_cover_image(self):
        try:
            self.status_var.set("Đang tải ảnh gốc...")
            self.root.update_idletasks()
            
            # Load image with OpenCV
            self.cover_img = cv2.imread(self.cover_path.get())
            
            # Display the image
            self.display_image(self.cover_img, self.cover_canvas)
            
            self.status_var.set("Ảnh gốc đã tải thành công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh gốc: {str(e)}")
            self.status_var.set("Lỗi khi tải ảnh gốc")
    
    def load_watermark_image(self):
        try:
            self.status_var.set("Đang tải ảnh thủy vân...")
            self.root.update_idletasks()
            
            # Load image with OpenCV
            self.watermark_img = cv2.imread(self.watermark_path.get())
            
            # Display the image
            self.display_image(self.watermark_img, self.watermark_canvas)
            
            self.status_var.set("Ảnh thủy vân đã tải thành công")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh thủy vân: {str(e)}")
            self.status_var.set("Lỗi khi tải ảnh thủy vân")
    
    def display_image(self, img, canvas):
        if img is None:
            return
        
        # Convert from BGR to RGB for display
        if len(img.shape) == 3:
            display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Resize image to fit in canvas while preserving aspect ratio
        canvas_width = canvas.winfo_width() or 350
        canvas_height = canvas.winfo_height() or 250
        
        h, w = display_img.shape[:2]
        ratio = min(canvas_width / w, canvas_height / h)
        new_size = (int(w * ratio), int(h * ratio))
        
        display_img = cv2.resize(display_img, new_size)
        
        # Convert to PIL format
        pil_img = Image.fromarray(display_img)
        
        # Convert to PhotoImage
        tk_img = ImageTk.PhotoImage(pil_img)
        
        # Update canvas
        canvas.delete("all")
        canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=tk_img)
        canvas.image = tk_img  # Keep a reference to prevent garbage collection

    def clear_canvas(self, canvas):
        """Xóa nội dung của một canvas"""
        canvas.delete("all")
    
    def embed_watermark(self):
        if self.cover_img is None or self.watermark_img is None:
            messagebox.showerror("Lỗi", "Vui lòng tải cả ảnh gốc và ảnh thủy vân trước")
            return
        
        try:
            algorithm = self.algorithm.get()
            self.status_var.set(f"Đang nhúng thủy vân với thuật toán {algorithm}...")
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            # Xóa ảnh trích xuất nếu có
            self.clear_canvas(self.extracted_canvas)
            
            # Prepare parameters
            cover_path = self.cover_path.get()
            watermark_path = self.watermark_path.get()
            
            # Ensure output directory exists
            if algorithm == "PCT":
                output_path = "PCT/ket_qua_thuy_van.png"
                os.makedirs("PCT", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply watermark
                result = apply_pct_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    block_size_m=self.pct_block_size_m.get(),
                    block_size_n=self.pct_block_size_n.get(),
                    r=self.pct_r_value.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "PCT/bao_cao_ket_qua.txt"
            
            elif algorithm == "Wu-Lee":
                output_path = "WU_LEE/ket_qua_thuy_van.png"
                os.makedirs("WU_LEE", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply Wu-Lee watermark
                result = apply_wulee_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    block_size_m=self.wulee_block_size_m.get(),
                    block_size_n=self.wulee_block_size_n.get(),
                    secret_key=self.secret_key.get(),
                    alpha=self.alpha.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "WU_LEE/bao_cao_ket_qua.txt"
            
            elif algorithm == "SW":
                output_path = "SW/ket_qua_thuy_van.png"
                os.makedirs("SW", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply SW watermark
                result = apply_sw_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    block_size_m=self.sw_block_size_m.get(),
                    block_size_n=self.sw_block_size_n.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "SW/bao_cao_ket_qua.txt"
            
            elif algorithm == "LSB":
                output_path = "LSB/ket_qua_thuy_van.png"
                os.makedirs("LSB", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply LSB watermark - thêm tham số khóa bí mật
                result = apply_lsb_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    secret_key=self.secret_key.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "LSB/bao_cao_ket_qua.txt"
            
            elif algorithm == "DWT":
                output_path = "DWT/ket_qua_thuy_van.png"
                os.makedirs("DWT", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply DWT watermark
                result = apply_dwt_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    decomposition_level=self.dwt_decomposition_level.get(),
                    quantization_step=self.dwt_quantization_step.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "DWT/bao_cao_ket_qua.txt"
            
            elif algorithm == "DCT":
                output_path = "DCT/ket_qua_thuy_van.png"
                os.makedirs("DCT", exist_ok=True)
                
                # Capture console output
                original_stdout = sys.stdout
                captured_output = StringIO()
                sys.stdout = captured_output
                
                self.progress_var.set(30)
                self.root.update_idletasks()
                
                # Apply DCT watermark
                result = apply_dct_watermark(
                    cover_path=cover_path,
                    watermark_path=watermark_path,
                    output_path=output_path,
                    block_size=self.dct_block_size.get(),
                    k_factor=self.dct_k_factor.get()
                )
                
                # Restore stdout
                sys.stdout = original_stdout
                
                # Load captured output
                log_text = captured_output.getvalue()
                
                # Update the report tab with the log
                self.report_text.delete(1.0, tk.END)
                self.report_text.insert(tk.END, log_text)
                
                # Update result values from dictionary
                if isinstance(result, dict):
                    self.psnr_var.set(f"{result.get('psnr', 0):.2f} dB")
                    self.embed_time_var.set(f"{result.get('embed_time', 0):.3f} giây")
                    self.extract_time_var.set(f"{result.get('extract_time', 0):.3f} giây")
                    
                    total_pixels = result.get('total_pixels', 0)
                    modified_pixels = result.get('modified_pixels', 0)
                    percent = (modified_pixels / total_pixels * 100) if total_pixels > 0 else 0
                    self.modified_pixels_var.set(f"{modified_pixels}/{total_pixels} ({percent:.4f}%)")
                    
                    self.accuracy_var.set(f"{result.get('accuracy', 0):.2f}%")
                
                # Đọc file báo cáo nếu có để cập nhật kết quả
                report_path = "DCT/bao_cao_ket_qua.txt"
            
            # Đọc và hiển thị báo cáo nếu tồn tại
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                    # Tìm và cập nhật các giá trị từ báo cáo
                    self.update_results_from_report(report_content)
            
            self.progress_var.set(70)
            self.root.update_idletasks()
            
            # Load watermarked image
            self.watermarked_img = cv2.imread(output_path)
            
            # Display watermarked image
            self.display_image(self.watermarked_img, self.watermarked_canvas)
            
            self.progress_var.set(100)
            self.status_var.set(f"Nhúng thủy vân thành công với thuật toán {algorithm}")
            
            # Switch to the Images tab
            self.tabs.select(0)
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể nhúng thủy vân: {str(e)}")
            self.status_var.set("Lỗi khi nhúng thủy vân")
        finally:
            self.progress_var.set(0)
  
    def update_results_from_report(self, report_text):
        """Cập nhật thông tin kết quả từ nội dung báo cáo"""
        try:
            # Tìm thời gian nhúng
            embed_time_match = re.search(r"Thời gian nhúng thủy vân: (\d+\.\d+) giây", report_text)
            if embed_time_match:
                self.embed_time_var.set(f"{float(embed_time_match.group(1)):.3f} giây")
                
            # Tìm thời gian trích xuất
            extract_time_match = re.search(r"Thời gian trích xuất thủy vân: (\d+\.\d+) giây", report_text)
            if extract_time_match:
                self.extract_time_var.set(f"{float(extract_time_match.group(1)):.3f} giây")
                
            # Tìm PSNR
            psnr_match = re.search(r"PSNR: (\d+\.\d+) dB", report_text)
            if psnr_match:
                self.psnr_var.set(f"{float(psnr_match.group(1)):.2f} dB")
                
            # Tìm số pixel đã sửa đổi
            pixel_match = re.search(r"Số pixel đã sửa đổi: (\d+)/(\d+) \((\d+\.\d+)%\)", report_text)
            if pixel_match:
                modified = pixel_match.group(1)
                total = pixel_match.group(2)
                percent = pixel_match.group(3)
                self.modified_pixels_var.set(f"{modified}/{total} ({percent}%)")
                
            # Tìm độ chính xác
            accuracy_match = re.search(r"Độ chính xác trích xuất: (\d+\.\d+)% \((\d+) bit lỗi\)", report_text)
            if accuracy_match:
                self.accuracy_var.set(f"{float(accuracy_match.group(1)):.2f}% ({accuracy_match.group(2)} bit lỗi)")
        except Exception as e:
            print(f"Lỗi khi cập nhật kết quả: {e}")
    
    def extract_watermark(self):
        algorithm = self.algorithm.get()
        
        output_folder = ""
        if algorithm == "PCT":
            output_folder = "PCT"
        elif algorithm == "Wu-Lee":
            output_folder = "WU_LEE"
        elif algorithm == "SW":
            output_folder = "SW"
        elif algorithm == "LSB":
            output_folder = "LSB"
        elif algorithm == "DWT":
            output_folder = "DWT"
        elif algorithm == "DCT":
            output_folder = "DCT"
            
        output_path = f"{output_folder}/ket_qua_thuy_van.png"
        
        # Check if there's a watermarked image
        if not os.path.exists(output_path):
            messagebox.showerror("Lỗi", "Không tìm thấy ảnh đã nhúng thủy vân. Vui lòng nhúng thủy vân trước.")
            return
        
        try:
            self.status_var.set("Đang trích xuất thủy vân...")
            self.progress_var.set(10)
            self.root.update_idletasks()
            
            # Check for extracted watermark
            extracted_path = f"{output_folder}/thuy_van_trich_xuat.png"
            if os.path.exists(extracted_path):
                self.progress_var.set(60)
                self.root.update_idletasks()
                
                # Load and display extracted watermark
                self.extracted_watermark = cv2.imread(extracted_path)
                self.display_image(self.extracted_watermark, self.extracted_canvas)
                
                # For DCT algorithm, also show original watermark if available
                if algorithm == "DCT":
                    original_watermark_path = f"{output_folder}/thuy_van_goc.png"
                    if os.path.exists(original_watermark_path):
                        original_watermark = cv2.imread(original_watermark_path)
                        if self.watermark_img is None:
                            self.watermark_img = original_watermark
                            self.display_image(self.watermark_img, self.watermark_canvas)
                
                # Load report file if it exists
                report_path = f"{output_folder}/bao_cao_ket_qua.txt"
                if os.path.exists(report_path):
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                        self.report_text.delete(1.0, tk.END)
                        self.report_text.insert(tk.END, report_content)
                        
                        # Cập nhật kết quả từ báo cáo
                        self.update_results_from_report(report_content)
                
                self.progress_var.set(100)
                self.status_var.set("Trích xuất thủy vân thành công")
            else:
                self.progress_var.set(0)
                messagebox.showinfo("Thông báo", "Không tìm thấy ảnh thủy vân trích xuất. Hãy thử nhúng thủy vân lại.")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi trích xuất: {str(e)}")
            self.status_var.set("Lỗi khi trích xuất thủy vân")
        finally:
            self.progress_var.set(0)
    
    def save_results(self):
        if self.watermarked_img is None:
            messagebox.showerror("Lỗi", "Vui lòng nhúng thủy vân trước")
            return
        
        try:
            # Create directory if it doesn't exist
            algorithm = self.algorithm.get()
            os.makedirs(algorithm, exist_ok=True)
            
            # Save watermarked image
            watermarked_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("Tệp PNG", "*.png"), ("Tất cả tệp", "*.*")],
                initialdir=algorithm,
                initialfile=f"{algorithm}_anh_da_thuy_van.png",
                title="Lưu ảnh đã nhúng thủy vân"
            )
            
            if watermarked_path:
                cv2.imwrite(watermarked_path, self.watermarked_img)
                self.status_var.set(f"Ảnh đã nhúng thủy vân được lưu tại {watermarked_path}")
            
            # Save extracted watermark if available
            if self.extracted_watermark is not None:
                extracted_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("Tệp PNG", "*.png"), ("Tất cả tệp", "*.*")],
                    initialdir=algorithm,
                    initialfile=f"{algorithm}_thuy_van_trich_xuat.png",
                    title="Lưu thủy vân đã trích xuất"
                )
                
                if extracted_path:
                    cv2.imwrite(extracted_path, self.extracted_watermark)
                    self.status_var.set(f"Thủy vân đã trích xuất được lưu tại {extracted_path}")
                
            # Save report if available
            report_path = "PCT/bao_cao_ket_qua.txt"
            if os.path.exists(report_path):
                report_save_path = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Tệp văn bản", "*.txt"), ("Tất cả tệp", "*.*")],
                    initialdir=algorithm,
                    initialfile=f"{algorithm}_bao_cao_ket_qua.txt",
                    title="Lưu báo cáo kết quả"
                )
                
                if report_save_path:
                    with open(report_path, 'r', encoding='utf-8') as src, open(report_save_path, 'w', encoding='utf-8') as dst:
                        dst.write(src.read())
                    self.status_var.set(f"Báo cáo kết quả đã được lưu tại {report_save_path}")
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu kết quả: {str(e)}")
            self.status_var.set("Lỗi khi lưu kết quả")

if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkApp(root)
    root.mainloop() 