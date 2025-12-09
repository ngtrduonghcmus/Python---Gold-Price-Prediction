import pandas as pd

class DataOverview:
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

    def __repr__(self):
        """Trả về chuỗi mô tả class gồm đường dẫn file và số dòng của DataFrame."""
        return f"DataOverview(filepath={self.filepath}, rows={0 if self.df is None else len(self.df)})"

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
        if self.filepath is None:
            print(" Lỗi: Đường dẫn file chưa được cung cấp.")
            return None

        try:
            if self.filepath.endswith(".csv"):
                self.df = pd.read_csv(self.filepath)
            elif self.filepath.endswith(".xlsx"):
                self.df = pd.read_excel(self.filepath)
            elif self.filepath.endswith(".json"):
                self.df = pd.read_json(self.filepath)
            else:
                raise ValueError("Định dạng file không được hỗ trợ (chỉ chấp nhận .csv, .xlsx, .json).")
        except FileNotFoundError:
             print(f" Lỗi: Không tìm thấy file tại đường dẫn: {self.filepath}")
             self.df = None
        except Exception as e:
            print(" Lỗi khi đọc file:", e)
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
        if self.df is None:
            print(" DataFrame trống. Hãy đọc dữ liệu trước.")
            return

        print("========== DATAFRAME INFO ==========")
        self.df.info()

        print("\n========== NULL COUNT ==========")
        nan_counts = self.df.isna().sum()
        print(nan_counts[nan_counts > 0])

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
        if self.df is None:
            print(" DataFrame trống. Không có gì để xuất.")
            return

        self.df.to_csv(path, index=False)
        print(" Đã lưu file:", path)