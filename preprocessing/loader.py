import pandas as pd
from utils.logger import get_preprocess_logger

class DataLoader:
    """
    Lớp hỗ trợ đọc dữ liệu, trích xuất các thông tin cơ bản và xuất dữ liệu.
    Các lớp khác (Preprocessor, Analyzer, Scaler, Visualizer) sẽ dùng self.df của lớp này.
    """

    def __init__(self, filepath=None):
        """
        Khởi tạo đối tượng xử lý dữ liệu.

        Tham số
        -------
        filepath : str hoặc None
            Đường dẫn đến file dữ liệu (CSV, Excel, JSON).
        """
        self.filepath = filepath
        self.df = None
        self.encoders = {}
        self.scalers = {}
        self.logger = get_preprocess_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO DATA LOADER <<<")

    def __repr__(self):
        """Trả về chuỗi mô tả class gồm đường dẫn file và số dòng của DataFrame."""
        return f"DataLoader(filepath={self.filepath}, rows={0 if self.df is None else len(self.df)})"

    # ************************************************
    # 1. ĐỌC DỮ LIỆU
    # ************************************************
    def read_data(self):
        """
        Đọc dữ liệu từ file CSV / Excel / JSON dựa trên phần mở rộng.

        Trả về
        -------
        DataFrame
            Dữ liệu đã được đọc thành công.
        """

        self.logger.info(f"*** ĐỌC DỮ LIỆU ***")
        if self.filepath is None:
            self.logger.error("Đường dẫn file chưa được cung cấp.")
            return None

        try:
            if self.filepath.endswith(".csv"):
                self.df = pd.read_csv(self.filepath)
            elif self.filepath.endswith(".xlsx"):
                self.df = pd.read_excel(self.filepath)
            elif self.filepath.endswith(".json"):
                self.df = pd.read_json(self.filepath)
            else:
                self.logger.error("Định dạng file không được hỗ trợ (chỉ chấp nhận .csv, .xlsx, .json).")
                raise ValueError("Định dạng file không được hỗ trợ (chỉ chấp nhận .csv, .xlsx, .json).")
            
            self.logger.info(f"Đọc dữ liệu thành công — Rows: {len(self.df)}, Cols: {len(self.df.columns)}")
        except FileNotFoundError:
            self.logger.error(f"Không tìm thấy file tại đường dẫn: {self.filepath}")
            self.df = None
        except Exception as e:
            self.logger.exception(f"Lỗi khi đọc file: {e}")
            self.df = None

        return self.df

    # ************************************************
    # 2. THÔNG TIN CƠ BẢN
    # ************************************************
    def get_basic_info(self):
        """
        In ra các thông tin cơ bản của DataFrame gồm info(), thống kê cột dạng chuỗi
        và số lượng giá trị thiếu. (describe() được chuyển sang StatisticsAnalyzer)
        """
        self.logger.info("*** THÔNG TIN CƠ BẢN CỦA DATAFRAME ***")
        if self.df is None:
            self.logger.warning("DataFrame trống – cần gọi read_data() trước.")
            return

        self.logger.debug(f"Số dòng: {len(self.df)}, Số cột: {len(self.df.columns)}")
        self.logger.debug(f"Thông tin DataFrame:\n{self.df.info()}")

        nan_counts = self.df.isna().sum()
        missing = nan_counts[nan_counts > 0]

        if len(missing) > 0:
            self.logger.warning(f"Giá trị thiếu:\n{missing}")
        else:
            self.logger.debug("Không có giá trị thiếu trong dữ liệu.")

    # ************************************************
    # 3. XUẤT DỮ LIỆU
    # ************************************************
    def export(self, path):
        """
        Xuất DataFrame ra file CSV.

        Tham số
        -------
        path : str
            Đường dẫn file CSV cần lưu.
        """
        self.logger.info("*** XUẤT DỮ LIỆU RA FILE CSV ***")
        if self.df is None:
            self.logger.error("DataFrame trống, không thể export.")
            return

        try:
            self.df.to_csv(path, index=False)
            self.logger.info(f"Đã xuất dữ liệu ra: {path}")
        except Exception as e:
            self.logger.error(f"Lỗi khi export file: {e}")
