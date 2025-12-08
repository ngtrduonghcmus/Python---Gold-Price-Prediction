import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from IPython.display import display

###########################################################
###---class DataPreprocessor---###
###########################################################

class DataPreprocessor:
    """
    DataPreprocessor
    ----------------
    Chịu trách nhiệm ETL cơ bản:
    1. Đọc dữ liệu (Read).
    2. Làm sạch cấu trúc và giá trị (Clean).
    3. Chuẩn hóa dữ liệu (Normalize).
    4. Xuất dữ liệu sạch (Export).
    """
    ###########################################################
    ###---0. KHỞI TẠO & CẤU HÌNH---###
    ###########################################################
    def __init__(self, filepath: str = None):
        """
        Khởi tạo đối tượng DataPreprocessor.
        Args:
            filepath (str): Đường dẫn file dữ liệu đầu vào.
        """
        self.filepath = filepath
        self._df = None
        self.scalers = {}  # Lưu scaler để có thể inverse_transform sau này nếu cần
        
        # Cấu hình Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)

    ###########################################################
    ###---PROPERTY — DataFrame---###
    ###########################################################

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, new_df):
        if new_df is not None and not isinstance(new_df, pd.DataFrame):
            raise ValueError("Dữ liệu gán vào phải là pandas DataFrame.")
        self._df = new_df

    def __repr__(self):
        rows = 0 if self.df is None else len(self.df)
        cols = 0 if self.df is None else len(self.df.columns)
        return f"<{self.__class__.__name__} source='{self.filepath}' shape=({rows}, {cols})>"

    ###########################################################
    ###---1. ĐỌC DỮ LIỆU---###
    ###########################################################

    def read_data(self):
        """Đọc dữ liệu từ file nguồn (CSV, Excel, JSON)."""
        if not self.filepath:
            raise ValueError("Filepath chưa được cung cấp.")
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Không tìm thấy file: {self.filepath}")

        ext = os.path.splitext(self.filepath)[1].lower()
        try:
            if ext == ".csv":
                self.df = pd.read_csv(self.filepath)
            elif ext in [".xlsx", ".xls"]:
                self.df = pd.read_excel(self.filepath)
            elif ext == ".json":
                self.df = pd.read_json(self.filepath)
            else:
                raise ValueError(f"Định dạng file '{ext}' không hỗ trợ.")

            self.logger.info(f"Đã đọc dữ liệu. Shape: {self.df.shape}")
            return self  # Trả về self để hỗ trợ method chaining
        except Exception as e:
            self.logger.error(f"Lỗi đọc file: {e}")
            raise
        
    ###########################################################
    ###---1.1. THÔNG TIN CƠ BẢN---###
    ###########################################################

    def basic_info(self):
        """
        Hiển thị thông tin tổng quan, thống kê mô tả và kiểm tra giá trị thiếu.
        """
        if self.df is None:
            print("DataFrame rỗng. Vui lòng gọi read_data() trước.")
            return

        print("\n--- INFO ---")
        self.df.info()

        print("\n--- DESCRIBE ---")
        display(self.df.describe(include="all").T)

        print("\n--- NULL COUNT ---")
        null_counts = self.df.isna().sum()
        if null_counts.sum() > 0:
            display(null_counts[null_counts > 0].to_frame(name="Missing Values"))
        else:
            print("Không có giá trị thiếu.")

    ###########################################################
    ###---2. XỬ LÝ CẤU TRÚC (STRUCTURE)---###
    ###########################################################

    def clean_structure(self):
        """
        Tổng hợp làm sạch cấu trúc:
        1. Xóa cột rác (Empty, Unnamed).
        2. Chuyển đổi định dạng số (xử lý dấu phẩy '1,234.56').
        """
        if self.df is None: return self

        # 1. Xóa cột rác
        junk_keywords = ["Empty", "Unnamed"]
        initial_cols = self.df.columns
        # Lọc các cột chứa từ khóa rác hoặc toàn bộ là NaN
        cols_to_drop = [c for c in initial_cols if any(k in c for k in junk_keywords)]
        all_na_cols = self.df.columns[self.df.isna().all()].tolist()
        cols_to_drop = list(set(cols_to_drop + all_na_cols))

        if cols_to_drop:
            self.df.drop(columns=cols_to_drop, inplace=True)
            self.logger.info(f"✓ Đã xóa {len(cols_to_drop)} cột rác.")

        # 2. Chuyển đổi String sang Float (Tối ưu hóa Vectorization)
        obj_cols = self.df.select_dtypes(include='object').columns
        converted_count = 0
        
        for col in obj_cols:
            if "date" in col.lower(): continue # Bỏ qua cột ngày
            try:
                # Thử chuyển đổi nhanh
                # Logic: Xóa dấu ',' rồi ép kiểu. 'coerce' sẽ biến lỗi thành NaN
                series_clean = self.df[col].astype(str).str.replace(',', '', regex=False)
                self.df[col] = pd.to_numeric(series_clean, errors='coerce')
                converted_count += 1
            except Exception:
                continue
                
        if converted_count > 0:
            self.logger.info(f"✓ Đã sửa định dạng số liệu cho {converted_count} cột.")
            
        return self

    ###########################################################
    ###---3. XỬ LÝ THỜI GIAN (DATETIME)---###
    ###########################################################

    def process_datetime(self, col: str = "Date"):
        """Chuẩn hóa cột thời gian, sắp xếp và set index."""
        if self.df is None: return self
        if col not in self.df.columns:
            self.logger.warning(f"Không tìm thấy cột thời gian '{col}'. Bỏ qua bước này.")
            return self

        # Chuyển đổi
        self.df[col] = pd.to_datetime(self.df[col], errors='coerce', dayfirst=False)
        
        # Xóa dòng không có ngày tháng
        self.df.dropna(subset=[col], inplace=True)
        
        # Sắp xếp và Set Index
        self.df.sort_values(col, inplace=True)
        self.df.set_index(col, inplace=True)
        
        # Xóa trùng lặp Index (nếu có, giữ giá trị cuối cùng)
        if self.df.index.duplicated().any():
            dup_count = self.df.index.duplicated().sum()
            self.df = self.df[~self.df.index.duplicated(keep='last')]
            self.logger.info(f"✓ Đã xóa {dup_count} mốc thời gian trùng lặp.")

        self.logger.info("✓ Đã chuẩn hóa DateTime.")
        return self
    
    ###########################################################
    ###---4. XỬ LÝ GIÁ TRỊ THIẾU & NGOẠI LAI---###
    ###########################################################

    def clean_values(self):
        """
        Xử lý giá trị nội tại:
        1. Điền giá trị thiếu (Interpolate cho Time Series).
        2. Xử lý ngoại lai (Capping - Giới hạn biên).
        """
        if self.df is None: return self

        # 1. Fill Missing Values (Ưu tiên Interpolate Time)
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        self.df[numeric_cols] = self.df[numeric_cols].interpolate(method='time').ffill().bfill()
        
        # Fill Categorical bằng Mode
        cat_cols = self.df.select_dtypes(exclude=np.number).columns
        for c in cat_cols:
            if self.df[c].isna().any():
                self.df[c] = self.df[c].fillna(self.df[c].mode()[0])

        # 2. Handle Outliers (Capping 3-Sigma)
        # Không xóa dòng để tránh đứt gãy chuỗi thời gian
        capped_count = 0
        for col in numeric_cols:
            if self.df[col].nunique() < 10: continue # Bỏ qua cột ít giá trị (như biến cờ 0/1)
            
            mean, std = self.df[col].mean(), self.df[col].std()
            upper = mean + 3 * std
            lower = mean - 3 * std
            
            # Nếu có giá trị vượt ngưỡng thì mới xử lý
            if (self.df[col] > upper).any() or (self.df[col] < lower).any():
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                capped_count += 1
                
        self.logger.info(f"✓ Đã điền giá trị thiếu và giới hạn ngoại lai ({capped_count} cột).")
        return self
    
    ###########################################################
    ###---5. CHUẨN HÓA DỮ LIỆU (NORMALIZATION)---###
    ###########################################################

    def normalize_data(self, method: str = 'minmax', exclude_cols: list = None):
        """
        Chuẩn hóa dữ liệu về cùng phạm vi (Scale).
        Args:
            method: 'minmax' (0-1) hoặc 'standard' (mean=0, std=1).
            exclude_cols: Danh sách các cột KHÔNG cần chuẩn hóa (ví dụ cột Target nếu muốn giữ nguyên giá).
        """
        if self.df is None: return self

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("Method chỉ hỗ trợ 'minmax' hoặc 'standard'.")

        # Xác định các cột cần scale (chỉ cột số)
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        if exclude_cols:
            numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        if len(numeric_cols) > 0:
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            self.scalers[method] = scaler # Lưu scaler
            self.logger.info(f"✓ Đã chuẩn hóa dữ liệu ({method}) cho {len(numeric_cols)} cột.")
        
        return self
    
    ###########################################################
    ###---6. PIPELINE & EXPORT---###
    ###########################################################

    def full_pipeline(self, output_path: str = "clean_data.csv"):
        """
        Chạy toàn bộ quy trình làm sạch và xuất file.
        """
        self.logger.info("=== BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ ===")
        
        (self.read_data()
             .clean_structure()
             .process_datetime(col="Date")
             .clean_values()
             .normalize_data(method='minmax') # Mặc định chuẩn hóa luôn
        )
        
        # Xuất file
        self.df.to_csv(output_path)
        self.logger.info(f"=== HOÀN TẤT. File đã lưu tại: {output_path} ===")
        return self.df

# =========================================================
# HƯỚNG DẪN SỬ DỤNG
# =========================================================
# preprocessor = DataPreprocessor('FINAL_USO_dirty.csv')
# df_clean = preprocessor.full_pipeline('clean_data.csv')