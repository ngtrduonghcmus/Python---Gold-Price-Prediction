import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
from pandas.api.types import is_numeric_dtype

class DataPreprocessor:
    """
    Lớp thực hiện làm sạch dữ liệu như điền các giá trị khuyết, chuẩn hoá định
    dạng kiểu dữ liệu, loại bỏ nhiễu và dữ liệu dư thừa.
    """
    def __init__(self, df: pd.DataFrame, encoders: dict):
        """
        Khởi tạo đối tượng với DataFrame và bộ mã hóa (cho LabelEncoder).

        Parameters
        ----------
        df : pandas.DataFrame
            Bộ dữ liệu cần tiền xử lý.
        encoders : dict
            Dict để lưu trữ các LabelEncoder đã fit.
        """
        self.df = df
        self.encoders = encoders

    # ************************************************
    # 1. FILL DATE NA (Phương pháp nội suy ngày tháng)
    # ************************************************
    def clean_date_column(self, column_name: str = 'Date') -> pd.DataFrame:
            """
            Xử lý cột ngày tháng: Chuyển đổi, nội suy, loại bỏ trùng lặp và sắp xếp.
            Đảm bảo mỗi ngày là duy nhất và xếp theo thứ tự tăng dần.
            """
            # 1. Chuyển đổi sang kiểu datetime, ép buộc lỗi thành NaT
            self.df[column_name] = pd.to_datetime(
                self.df[column_name],
                errors='coerce',
                format='%m/%d/%Y'
            )

            print(f"Bắt đầu nội suy các giá trị bị thiếu trong cột '{column_name}'...")

            # 2. Xử lý nội suy (giữ nguyên logic của bạn)
            date_ordinal = self.df[column_name].apply(
                lambda x: x.toordinal() if pd.notna(x) else np.nan
            )
            date_ordinal_filled = date_ordinal.interpolate(method='linear')
            date_ordinal_filled = date_ordinal_filled.round().astype('Int64')

            self.df[column_name] = date_ordinal_filled.apply(
                lambda x: datetime.date.fromordinal(x) if pd.notna(x) else pd.NaT
            )
            self.df[column_name] = pd.to_datetime(self.df[column_name])

            print(f"Đã nội suy cột '{column_name}'.")

            # 3. LOẠI BỎ CÁC NGÀY TRÙNG LẶP (BƯỚC BỔ SUNG ĐỂ ĐẢM BẢO TÍNH DUY NHẤT)
            initial_rows = len(self.df)
            self.df.drop_duplicates(subset=[column_name], keep='first', inplace=True)
            dropped_rows = initial_rows - len(self.df)

            if dropped_rows > 0:
                print(f"Đã loại bỏ {dropped_rows} hàng trùng lặp theo cột '{column_name}'.")

            # 4. Sắp xếp lại và đặt lại chỉ mục (như ban đầu)
            self.df.sort_values(by=column_name, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            print(f"Đã sắp xếp lại và reset index.")
            print(f"Cột '{column_name}' đã được chuyển đổi sang kiểu: {self.df[column_name].dtype}")

            return self.df
    # ************************************************
    # 2. XỬ LÝ CỘT NGÀY (Chuẩn hóa định dạng)
    # ************************************************
    def process_datetime_column(self, col="Date"):
        """
        Chuẩn hóa cột ngày theo nhiều định dạng khác nhau, tự phát hiện dạng ngày,
        cố gắng chuyển về Timestamp chính xác.

        Tham số
        -------
        col : str
            Tên cột cần xử lý.

        Trả về
        -------
        DataFrame
            DataFrame sau khi chuẩn hóa cột ngày.
        """
        def _normalize(x):
            if pd.isna(x):
                return pd.NaT
            if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.date)):
                return pd.to_datetime(x, errors="coerce")

            x = str(x).strip()
            dt = pd.to_datetime(x, errors="coerce", dayfirst=False)
            if pd.notna(dt):
                return dt

            dt = pd.to_datetime(x, errors="coerce", dayfirst=True)
            if pd.notna(dt):
                return dt

            x2 = x.replace(".", "/").replace("-", "/")
            parts = x2.split("/")

            if len(parts) == 3:
                try:
                    a, b, c = map(int, parts)
                    if c > 31: # Xem c là năm
                        if a <= 12 and b > 12: # Y M D
                            return pd.Timestamp(year=c, month=a, day=b)
                        if b <= 12 and a > 12: # Y D M (Trường hợp ít xảy ra nhưng cân nhắc)
                            return pd.Timestamp(year=c, month=b, day=a)
                        return pd.Timestamp(year=c, month=a, day=b) # Y M D
                    if a > 31: # Xem a là năm
                        return pd.Timestamp(year=a, month=b, day=c) # Y M D
                except:
                    pass
            return pd.NaT

        self.df[col] = self.df[col].apply(_normalize)
        return self.df

    # ************************************************
    # 3. XOÁ CỘT / HÀNG CHẤT LƯỢNG THẤP
    # ************************************************
    def drop_low_valid_columns(self, threshold_ratio=2 / 3):
        """
        Xoá cột hoặc hàng có tỷ lệ giá trị hợp lệ dưới một ngưỡng nhất định.

        Tham số
        -------
        threshold_ratio : float
            Ngưỡng tỷ lệ (0–1), mặc định 2/3.

        Trả về
        -------
        DataFrame
            Dữ liệu sau khi loại bỏ cột/hàng chất lượng thấp.
        """
        total_rows = len(self.df)
        threshold = total_rows * threshold_ratio

        cols_to_drop = [col for col in self.df.columns if self.df[col].notna().sum() < threshold]
        if cols_to_drop:
            print(" Cột bị xoá:")
            for col in cols_to_drop:
                print(" -", col)
            self.df.drop(columns=cols_to_drop, inplace=True)

        total_cols = self.df.shape[1]
        row_threshold = total_cols * threshold_ratio
        rows_to_drop = self.df.index[self.df.notna().sum(axis=1) < row_threshold]

        if len(rows_to_drop) > 0:
            print("\n Hàng bị xoá:")
            for r in rows_to_drop:
                print(" - Index", r)
            self.df.drop(index=rows_to_drop, inplace=True)

        return self.df

    # ************************************************
    # 4. XOÁ TRÙNG LẶP
    # ************************************************
    def check_and_drop_duplicates(self):
        """
        Kiểm tra và xoá các hàng trùng lặp trong DataFrame.

        Trả về
        -------
        DataFrame
            Dữ liệu sau khi xoá trùng lặp và reset index.
        """
        print("Các hàng trùng lặp:")
        duplicate_rows = self.df[self.df.duplicated(keep=False)]
        print(duplicate_rows)

        self.df = self.df.drop_duplicates().reset_index(drop=True)
        print("\nSố hàng sau khi xoá trùng:", len(self.df))
        return self.df

    # ************************************************
    # 5. ĐIỀN GIÁ TRỊ THIẾU
    # ************************************************

    def fill_missing(self, strategy="median", custom_value=None, neighbors=None):
        """
        Điền giá trị thiếu cho toàn bộ DataFrame bằng nhiều chiến lược:
        mean, median, mode, ffill, custom, interpolate.

        Tham số
        -------
        strategy : str
            Chiến lược điền giá trị.
        custom_value : any
            Giá trị điền thủ công (nếu dùng strategy='custom').
        neighbors : int hoặc None
            Giới hạn số lượng giá trị lân cận khi nội suy.

        Trả về
        -------
        DataFrame
            Dữ liệu sau khi điền giá trị thiếu.
        """
        for col in self.df.columns:
            if self.df[col].isnull().sum() == 0:
                continue

            try:
                if strategy == "mean":
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == "median":
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == "mode":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                elif strategy == "ffill":
                    self.df[col] = self.df[col].fillna(method="ffill")
                elif strategy == "custom":
                    self.df[col] = self.df[col].fillna(custom_value)
                elif strategy == "interpolate":
                    if neighbors is None:
                        self.df[col] = self.df[col].interpolate(method='linear')
                    else:
                        self.df[col] = self.df[col].interpolate(
                            method='linear',
                            limit=neighbors,
                            limit_direction='both'
                        )
            except:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        return self.df

    # ************************************************
    # 6. XỬ LÝ ĐỊNH DẠNG SỐ
    # ************************************************
    def clean_decimal_format(self, date_col_name: str = 'Date') -> pd.DataFrame:
        """
        Kiểm tra các cột có kiểu dữ liệu 'object' (ngoại trừ cột ngày tháng),
        thay thế dấu ',' bằng dấu '.' (nếu có) và chuyển về kiểu float.

        Tham số:
            date_col_name (str): Tên cột ngày tháng để loại trừ khỏi quá trình kiểm tra.

        Trả về:
            pd.DataFrame: DataFrame sau khi làm sạch định dạng số.
        """
        df_cleaned = self.df
        object_cols = df_cleaned.select_dtypes(include=['object']).columns.tolist()

        # Loại bỏ cột ngày tháng khỏi danh sách cần kiểm tra
        if date_col_name in object_cols:
            object_cols.remove(date_col_name)

        cleaned_cols = []

        for col in object_cols:
            try:
                # Lấy cột dưới dạng chuỗi
                col_as_str = df_cleaned[col].astype(str)

                # Chỉ thay thế và chuyển đổi nếu cột có chứa dấu phẩy
                if col_as_str.str.contains(',', regex=False).any():
                    print(f" Đang làm sạch định dạng số cho cột '{col}'...")
                    # Thay thế dấu phẩy bằng dấu chấm và chuyển về float
                    df_cleaned[col] = (col_as_str
                                        .str.replace(',', '.', regex=False)
                                        .astype(float))
                    cleaned_cols.append(col)
                else:
                    # Thử chuyển đổi sang số cho tất cả các cột object còn lại
                    df_cleaned[col] = pd.to_numeric(col_as_str, errors='coerce')
                    # Kiểm tra nếu kiểu dữ liệu đã chuyển thành số thực (float) hoặc số nguyên (int)
                    if df_cleaned[col].dtype.kind in 'fi' and col not in cleaned_cols:
                        cleaned_cols.append(col)

            except Exception as e:
                print(f" Lỗi không thể chuyển đổi cột '{col}' sang float: {e}")

        if cleaned_cols:
            print(f" ✔ Đã chuyển đổi định dạng số/chuỗi số cho {len(cleaned_cols)} cột: {', '.join(cleaned_cols)}")
        else:
            print(" ✔ Không tìm thấy cột kiểu 'object' nào cần làm sạch định dạng số (ngoài cột ngày).")

        self.df = df_cleaned
        return self.df
    # ************************************************
    # 7. XỬ LÍ NGOẠI LAI
    # ************************************************
    @staticmethod
    def detect_outliers_iqr(series):
        """Phát hiện ngoại lai bằng phương pháp IQR."""
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < Q1 - 1.5 * IQR) | (series > Q3 + 1.5 * IQR)

    @staticmethod
    def detect_outliers_zscore(series, threshold=3):
        """Phát hiện ngoại lai bằng Z-score."""
        # Chỉ áp dụng Z-score cho các giá trị đã điền đầy đủ
        return abs(zscore(series.dropna())) > threshold

    def handle_outliers(self, method="iqr"):
        """
        Xử lý ngoại lai bằng IQR, Z-score hoặc IsolationForest.
        Giá trị ngoại lai được thay bằng NaN rồi điền lại bằng median.

        Tham số
        -------
        method : str
            iqr, zscore hoặc isolation_forest.

        Trả về
        -------
        DataFrame
            Dữ liệu sau khi xử lý ngoại lai.
        """
        numeric_cols = self.df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if self.df[col].isnull().all():
                continue # Bỏ qua cột rỗng

            mask = pd.Series(False, index=self.df.index)

            if method == "iqr":
                mask = DataPreprocessor.detect_outliers_iqr(self.df[col])
            elif method == "zscore":
                data_clean = self.df[col].dropna()
                if not data_clean.empty:
                    outlier_indices = data_clean.index[DataPreprocessor.detect_outliers_zscore(data_clean)]
                    mask[outlier_indices] = True
            elif method == "isolation_forest":
                data_clean = self.df[[col]].dropna()
                if not data_clean.empty:
                    iso = IsolationForest(contamination=0.01, random_state=42)
                    preds = iso.fit_predict(data_clean)
                    mask[data_clean.index[preds == -1]] = True
            else:
                raise ValueError("Phương pháp xử lý ngoại lai không hợp lệ!")

            self.df.loc[mask, col] = np.nan # Thay ngoại lai bằng NaN

        print(f" Đã thay thế ngoại lai bằng NaN ({method}).")

        # Sau khi thay bằng NaN, điền lại bằng median
        print(" Tiến hành điền lại giá trị thiếu bằng median...")
        self.fill_missing("median")
        return self.df

    # ************************************************
    # 8. CHUYỂN KIỂU MỘT CỘT
    # ************************************************
    def convert_dtype(self, column, dtype, errors='raise', format=None):
        """
        Chuyển kiểu dữ liệu cho một cột.

        Tham số
        -------
        column : str
            Tên cột.
        dtype : str hoặc callable
            Kiểu đích (int, float, str, bool, datetime, category).
        errors : str
            Nếu 'raise' sẽ báo lỗi, 'coerce' chuyển lỗi thành NaN.
        format : str
            Định dạng ngày sử dụng khi dtype='datetime'.

        Trả về
        -------
        DataFrame
            Dữ liệu sau chuyển kiểu.
        """
        if column not in self.df.columns:
            print(f" Cột '{column}' không tồn tại.")
            return self.df

        if dtype in ["datetime", "date", "time"]:
            if pd.api.types.is_datetime64_any_dtype(self.df[column]):
                print(f" Cột '{column}' đã là datetime -> bỏ qua.")
                return self.df

        print(f" Chuyển kiểu '{column}' -> {dtype}")

        try:
            if callable(dtype):
                self.df[column] = self.df[column].apply(dtype)
            elif dtype in ["int", "integer"]:
                self.df[column] = self.df[column].astype(float).astype("Int64")
            elif dtype == "float":
                self.df[column] = self.df[column].astype(float)
            elif dtype == "str":
                self.df[column] = self.df[column].astype(str)
            elif dtype == "bool":
                self.df[column] = self.df[column].astype(bool)
            elif dtype in ["datetime", "date", "time"]:
                self.df[column] = pd.to_datetime(self.df[column], errors=errors, format=format)
            elif dtype == "category":
                self.df[column] = self.df[column].astype("category")

            print(" Thành công!")
        except Exception as e:
            print(f" Lỗi: {e}")
            if errors == "raise":
                raise e

        return self.df

    # ************************************************
    # 9. CHUYỂN KIỂU NHIỀU CỘT
    # ************************************************
    def convert_dtypes_bulk(self, mapping, errors="raise"):
        """
        Chuyển kiểu hàng loạt cột.

        Tham số
        -------
        mapping : dict hoặc (list, dtype)
            dict: {cột: kiểu}
            tuple: ([danh sách cột], kiểu)
        errors : str
            Điều khiển hành vi khi lỗi chuyển kiểu.

        Trả về
        -------
        DataFrame
            Dữ liệu sau chuyển kiểu.
        """
        if isinstance(mapping, dict):
            for col, dtype in mapping.items():
                self.convert_dtype(col, dtype, errors=errors)
        elif isinstance(mapping, (list, tuple)) and len(mapping) == 2:
            cols, dtype = mapping
            for col in cols:
                self.convert_dtype(col, dtype, errors=errors)
        else:
            print(" mapping phải là dict hoặc (list_cột, dtype)")
        return self.df

    # ************************************************
    # 10. DỰ ĐOÁN KIỂU DỮ LIỆU
    # ************************************************
    def detect_best_dtype(self, series: pd.Series):
        """
        Dự đoán kiểu dữ liệu phù hợp nhất cho một cột.

        Trả về
        -------
        str
            Kiểu dữ liệu được gợi ý: int, float, bool, datetime, str, category.
        """
        s = series.dropna()

        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"

        if pd.api.types.is_numeric_dtype(s):
            return "int" if (s % 1 == 0).all() else "float"

        if s.dtype == object or pd.api.types.is_string_dtype(s):
            if s.astype(str).str.contains(r"[-/:\.]").mean() > 0.3:
                parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
                if parsed.notna().mean() > 0.8:
                    return "datetime"

            if s.astype(str).str.lower().isin(["true", "false", "0", "1"]).mean() == 1:
                return "bool"

        if s.nunique() / len(s) < 0.05:
            return "category"

        return "str"

    # ************************************************
    # 11. AUTO CONVERT
    # ************************************************
    def auto_convert(self):
        """
        Tự động chuyển kiểu dữ liệu hợp lý cho toàn bộ DataFrame dựa trên detect_best_dtype().

        Trả về
        -------
        DataFrame
            Dữ liệu sau chuyển đổi tự động.
        """
        print("\n BẮT ĐẦU TỰ ĐỘNG CHUYỂN KIỂU...\n")

        changed = []

        for col in self.df.columns:
            cur = str(self.df[col].dtype)
            best = self.detect_best_dtype(self.df[col])

            if cur.startswith("int") or cur == "Int64":
                current_logic = "int"
            elif cur.startswith("float"):
                current_logic = "float"
            elif "datetime" in cur:
                current_logic = "datetime"
            elif cur == "bool":
                current_logic = "bool"
            elif "category" in cur:
                current_logic = "category"
            else:
                current_logic = "str"

            if current_logic != best:
                print(f" {col}: {current_logic} -> {best}")
                self.convert_dtype(col, best, errors="coerce")
                changed.append((col, current_logic, best))
            else:
                print(f" {col} giữ nguyên ({current_logic})")

        print("\n HOÀN TẤT!")
        if changed:
            print("🔧 Các cột đã chuyển kiểu:")
            for c, old, new in changed:
                print(f" - {c}: {old} -> {new}")
        else:
            print("✔ Không có cột nào cần chuyển kiểu.")

        return self.df

    # ************************************************
    # 12. MÃ HÓA
    # ************************************************
    def label_encode(self, col):
        """
        Mã hóa cột dạng chuỗi thành số bằng Label Encoding.

        Trả về
        -------
        DataFrame
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return self.df

        le = LabelEncoder()
        # Ép về string để LabelEncoder xử lý
        data_clean = self.df[col].astype(str).dropna()
        self.df[col].loc[data_clean.index] = le.fit_transform(data_clean)
        self.encoders[col] = le
        return self.df

    def onehot_encode(self, col):
        """
        Mã hóa One-hot một cột (tạo nhiều cột nhị phân).

        Trả về
        -------
        DataFrame
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return self.df

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Chỉ lấy các giá trị không thiếu để fit và transform
        temp_df = self.df[[col]].astype(str) # Đảm bảo là chuỗi
        arr = ohe.fit_transform(temp_df)

        new_cols = [f"{col}_{c}" for c in ohe.categories_[0]]

        # Tạo DataFrame mới cho các cột One-Hot
        ohe_df = pd.DataFrame(arr, columns=new_cols, index=self.df.index)

        # Nối lại
        self.df = pd.concat([self.df.drop(columns=[col]), ohe_df], axis=1)
        return self.df

    @staticmethod
    def text_to_number(text):
        """
        Chuyển chuỗi ký tự thành số bằng cách cộng mã ASCII từng ký tự.
        """
        return sum(ord(c) for c in str(text))

    def apply_text_encoding(self, col):
        """
        Mã hóa chuỗi thành số theo hàm text_to_number().
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return self.df

        self.df[col] = self.df[col].apply(DataPreprocessor.text_to_number)
        return self.df
    
class DataScaler:
    """
    Lớp thực hiện biến đổi và chuẩn hoá dữ liệu để các mô hình học máy hoạt động ổn định.
    """
    def __init__(self, df: pd.DataFrame, scalers: dict):
        """
        Khởi tạo đối tượng với DataFrame và bộ scale (cho StandardScaler, MinMaxScaler).

        Parameters
        ----------
        df : pandas.DataFrame
            Bộ dữ liệu cần chuẩn hóa.
        scalers : dict
            Dict để lưu trữ các Scaler đã fit.
        """
        self.df = df
        self.scalers = scalers

    # ************************************************
    # 1. CHUẨN HÓA DỮ LIỆU
    # ************************************************
    def scale(self, method="standard", columns=None):
        """
        Chuẩn hoá dữ liệu số bằng StandardScaler, MinMaxScaler hoặc custom scaling.

        Tham số
        -------
        method : str
            Kiểu chuẩn hóa: standard, minmax, custom.
        columns : list, optional
            Danh sách các cột số cần chuẩn hóa. Nếu None, sẽ chọn tất cả các cột số.

        Trả về
        -------
        DataFrame
            Dữ liệu đã chuẩn hóa.
        """
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=np.number).columns
        else:
            numeric_cols = [col for col in columns if col in self.df.columns and is_numeric_dtype(self.df[col])]

        if numeric_cols.empty:
            print(" Không có cột số nào để chuẩn hóa.")
            return self.df

        print(f" Chuẩn hóa bằng phương pháp '{method}' cho các cột: {list(numeric_cols)}")

        if method == "standard":
            scaler = StandardScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            self.scalers['standard'] = scaler
        elif method == "minmax":
            scaler = MinMaxScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            self.scalers['minmax'] = scaler
        elif method == "custom":
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val - min_val == 0:
                    self.df[col] = 0 # Tránh chia cho 0
                else:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        else:
            raise ValueError("Phương pháp chuẩn hóa không hợp lệ! (standard, minmax, custom)")

        return self.df

