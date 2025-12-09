from preprocessing.data_loader import DataOverview
from preprocessing.eda import StatisticsAnalyzer, Visualizer
from preprocessing.preprocessor import DataPreprocessor, DataScaler

# Đọc dữ liệu
import gdown

file_id = "10h8D0SEASQmL0AORY6okEP6HSHbgqggn"
url = f"https://drive.google.com/uc?id={file_id}"

# Tải file về Colab
output = "test.csv"
gdown.download(url, output, quiet=False)


FILE_PATH = "test.csv"
# Đọc và Thiết lập DataFrame gốc
print("--- 1. ĐỌC DỮ LIỆU ---")
overview = DataOverview(filepath=FILE_PATH)
df_raw = overview.read_data() # Đọc dữ liệu vào overview.df
overview.get_basic_info()

# Khởi tạo các lớp khác với tham chiếu đến df trong overview
# Các lớp này sẽ làm việc trực tiếp trên overview.df
analyzer = StatisticsAnalyzer(df=overview.df)
preprocessor = DataPreprocessor(df=overview.df, encoders=overview.encoders)
scaler = DataScaler(df=overview.df, scalers=overview.scalers)
visualizer = Visualizer(df=overview.df)

# Phân tích Thống kê
print("Thống kê mô tả:")
analyzer.get_describe_stats()

# # Báo cáo Trực quan tự động (Chạy Visualizer)
# print(" Báo cáo Trực quan Tự động:")
# visualizer.full_report()

# Gọi hàm để làm sạch và chuyển đổi cột 'Date'
df=preprocessor.clean_date_column(column_name='Date')
print(f"Kích thước sau khi làm sạch cột Date: {overview.df.shape}")
print("Kiểu dữ liệu cột Date sau xử lý:")
print(overview.df['Date'].dtype)

# Lọc ra các hàng bị trùng lặp dựa trên cột 'Date' (giữ lại hàng đầu tiên)
df_duplicates = df[df.duplicated(subset=['Date'], keep='first')]

print("--- Các hàng trùng lặp theo cột 'Date' (Loại trừ lần xuất hiện đầu tiên) ---")
print(df_duplicates)

# Xóa các hàng/cột chất lượng thấp
print(" Xử lý cột/hàng chất lượng thấp (dưới 2/3 giá trị)")
df=preprocessor.drop_low_valid_columns(threshold_ratio=2/3)

# Xóa trùng lặp
print("\n Xóa trùng lặp...")
df=preprocessor.check_and_drop_duplicates()

# Tự động tìm lỗi dùng sai dấu . thành , và thay thế
df=preprocessor.clean_numeric_format()

# Kiểm tra số NaN cho tất cả các cột
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
print(nan_counts)

# Thay giá trị NaN bằng cách nội suy 10 giá trị xung quanh
df = preprocessor.fill_missing(strategy="interpolate", neighbors=10)

# Kiểm tra số NaN cho tất cả các cột
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0]
nan_counts

print("\n Tự động chuyển kiểu dữ liệu...")
df=preprocessor.auto_convert()

# Chuẩn hóa bằng MinMax Scaling (đưa về [0, 1])
# Hàm scale sẽ tự động tìm các cột số và chuẩn hóa chúng
print("Thực hiện MinMax Scaling cho các cột số...")
scaler.scale(method="minmax")

# Kiểm tra DataFrame cuối cùng
print(f" DataFrame cuối cùng đã xử lý :")
df.head()
print("\nThông tin cuối cùng:")
df.info()

df = preprocessor.auto_convert()
# Cập nhật DataFrame bên trong đối tượng overview
overview.df = df
# Gọi hàm export()
OUTPUT_FILE_PATH = "Output.csv"
overview.export(path=OUTPUT_FILE_PATH)