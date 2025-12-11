import pandas as pd
from preprocessing.loader import DataLoader
from preprocessing.eda import StatisticsAnalyzer, Visualizer
from preprocessing.preprocessing import DataPreprocessor, DataScaler
from training.config import ConfigManager
from training.datamodule import DataModule, TimeSeriesSplitter, VisualizationModule
from training.trainer import ModelTrainer, ModelEvaluator, HyperParameterOptimizer
from utils.logger import get_preprocess_logger, get_training_logger

## TIỀN XỬ LÝ DỮ LIỆU
""" ================ TIỀN XỬ LÝ DỮ LIỆU ================ """
preprocess_logger = get_preprocess_logger("Main")
preprocess_logger.info("=== TIỀN XỬ LÝ DỮ LIỆU ===")
FILE_PATH = "data.csv"
# Đọc và Thiết lập DataFrame gốc
overview = DataLoader(filepath=FILE_PATH)
df_raw = overview.read_data() # Đọc dữ liệu vào overview.df
overview.get_basic_info()

# Khởi tạo các lớp khác với tham chiếu đến df trong overview
# Các lớp này sẽ làm việc trực tiếp trên overview.df
analyzer = StatisticsAnalyzer(df=overview.df)
preprocessor = DataPreprocessor(df=overview.df, encoders=overview.encoders)
scaler = DataScaler(df=overview.df, scalers=overview.scalers)
visualizer = Visualizer(df=overview.df)

""" --------------- THỐNG KÊ MÔ TẢ --------------- """

# Phân tích Thống kê
analyzer.get_describe_stats()

# Báo cáo Trực quan tự động (Chạy Visualizer)
visualizer.full_report()

""" --------------- TIỀN XỬ LÝ DỮ LIỆU --------------- """

# Gọi hàm để làm sạch và chuyển đổi cột 'Date'
df=preprocessor.clean_date_column(column_name='Date')

# Xóa các hàng/cột chất lượng thấp
df=preprocessor.drop_low_valid_columns(threshold_ratio=2/3)

# Xóa trùng lặp
df=preprocessor.check_and_drop_duplicates()

# Tự động tìm lỗi dùng sai dấu . thành , và thay thế
df=preprocessor.clean_decimal_format()

# Thay giá trị NaN bằng cách nội suy 10 giá trị xung quanh
df = preprocessor.fill_missing(strategy="interpolate", neighbors=10)

df=preprocessor.auto_convert()

visualizer.df = df

# Gọi hàm full_report()
visualizer.full_report()

""" --------------- CHUẨN HÓA DỮ LIỆU SỐ (DataScaler) --------------- """

# Chuẩn hóa bằng MinMax Scaling (đưa về [0, 1])
# Hàm scale sẽ tự động tìm các cột số và chuẩn hóa chúng
scaler.scale(method="minmax")

""" --------------- KIỂM TRA VÀ XUẤT DỮ LIỆU --------------- """

df = preprocessor.auto_convert()
# Cập nhật DataFrame bên trong đối tượng overview
overview.df = df
# Gọi hàm export()
OUTPUT_FILE_PATH = "data_clean.csv"
overview.export(path=OUTPUT_FILE_PATH)

""" ================ HUẤN LUYỆN MÔ HÌNH ================ """
# Setup Logging ra màn hình console
training_logger = get_training_logger("Main")
training_logger.info("=== HUẤN LUYỆN MÔ HÌNH ===")

# 1. Khởi tạo Manager
""" --------------- CẤU HÌNH CHUNG --------------- """
cfg = ConfigManager()

training_logger.info(">>> THIẾT LẬP <<<")

""" --- A. CẤU HÌNH DỮ LIỆU & TIỀN XỬ LÝ --- """
# Đường dẫn file dữ liệu
cfg._config['data']['raw_path'] = 'data_clean.csv' 
# [QUAN TRỌNG] Chiến lược Log Return để khử Non-stationary
cfg._config['data']['target_transform'] = 'log_return' 
# Số lượng features tối đa giữ lại (tránh nhiễu)
cfg._config['data']['selection']['max_features'] = 30 
# Cấu hình Lag (Quan sát quá khứ: 1 ngày, 2 ngày, 1 tuần, 1 tháng)
cfg._config['data']['lags'] = [1, 2, 3, 5, 21]
#####
# cfg._config['splitting']['purge'] = 5  # Bỏ 5 ngày giữa Train và Test

""" --- B. CẤU HÌNH TRAINING --- """

cfg._config['training']['models'] = ['xgboost', 'catboost', 'lightgbm', 'random_forest'] 
cfg._config['training']['metric'] = 'rmse'

""" --- C. CẤU HÌNH PHẦN CỨNG --- """
# Nếu dữ liệu > 20k dòng set TRUE, để GPU xử lí hiệu quả hơn CPU.
cfg._config['hardware']['use_gpu'] = False 
cfg._config['hardware']['n_jobs'] = -1 # CPU cores dùng cho training

""" --- D. CẤU HÌNH TUNING (OPTUNA) --- """
# Số lần thử nghiệm tìm tham số (Tăng lên 50-100 nếu máy mạnh)
cfg._config['optimization']['n_trials'] = 50
cfg._config['optimization']['timeout'] = 1200 

""" --- E. KIỂM TRA & LƯU --- """
training_logger.debug(f"\n[CHECK] Target Mode:    {cfg.data_config['target_transform'].upper()}")
training_logger.debug(f"[CHECK] Active Models:  {cfg.train_config['models']}")
training_logger.debug(f"[CHECK] Tuning Trials:  {cfg.optim_config['n_trials']}")

# Lưu lại config để các Class sau tự động đọc
cfg.save_config()
training_logger.info("'config.yaml'")

""" --------------- DỮ LIỆU VÀ PIPELINE --------------- """
dm = DataModule(cfg)

# Chạy pipeline: Load -> Clean -> Log Transform -> Feature Engineering -> Selection
X, y = dm.run_pipeline()

# Kiểm tra nhanh dữ liệu
X_check, y_check = dm.run_pipeline()

training_logger.debug(f"\nKiểm tra dữ liệu đầu ra:")

training_logger.debug(f"1. Kích thước X: {X_check.shape}")
training_logger.debug(f"2. Kích thước y: {y_check.shape}")
training_logger.debug(f"3. Dải giá trị y (Return): Min={y_check.min():.4f}, Max={y_check.max():.4f}")
training_logger.debug(f"4. Raw Price cuối cùng: {dm.raw_prices.iloc[-1]}")
training_logger.debug(X.tail(3))
training_logger.debug(f"Features: {X.shape[1]} columns")
training_logger.debug(f"Total Samples: {len(X)}")

# 1. Khởi tạo
X, y = dm.run_pipeline()
splitter = TimeSeriesSplitter(cfg)
viz = VisualizationModule()

# 2. Lấy dữ liệu X, y từ DataModule 
if 'X' in globals() and 'y' in globals():
    
    # 3. Vẽ biểu đồ chia Fold để kiểm tra logic
    training_logger.info("Đang vẽ sơ đồ chia dữ liệu...")
    viz.plot_cv_indices(splitter, X, y)
    
    # 4. Lấy tập dữ liệu cuối cùng để chuẩn bị Train
    X_train, X_test, y_train, y_test = splitter.get_holdout_split(X, y)
    
    training_logger.debug("\nCheck shape final split:")
    training_logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    training_logger.debug(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
else:
    training_logger.error("Vui lòng chạy DataModule trước để có biến X, y!")

""" --------------- TỐI ƯU HÓA HYPERPARAMETERS VÀ TRAINING --------------- """
ENABLE_TUNING = True

if ENABLE_TUNING:
    training_logger.info("=== HYPERPARAMETER TUNING ===")
    
    # 1. Khởi tạo Optimizer (Class độc lập)
    optimizer = HyperParameterOptimizer(cfg, splitter, X, y)
    
    # 2. Chạy tối ưu hóa
    best_params_dict = optimizer.run_optimization()
    training_logger.info("\nTham số tối ưu đã được lưu file JSON.")


else:
    training_logger.warning("\n=== BỎ QUA HYPERPARAMETER TUNING ===")

trainer = ModelTrainer(cfg, raw_prices_series=dm.raw_prices) 

X_train, X_test, y_train, y_test = splitter.get_holdout_split(X, y)
final_results = trainer.train_final_model(X_train, y_train, X_test, y_test)

# training_logger.debug(pd.DataFrame(final_results).T.sort_values("RMSE"))

""" --------------- ĐÁNH GIÁ MÔ HÌNH --------------- """
# 1. Khởi tạo Evaluator
evaluator = ModelEvaluator(cfg)

# 2. Vẽ Leaderboard (Nếu đã có final_results)
if 'final_results' in globals():
    evaluator.plot_leaderboard(final_results)
    ##  # ---(TEXT REPORT) ---
    training_logger.info("\n" + "="*60)
    training_logger.info("BÁO CÁO TỔNG HỢP HIỆU SUẤT MÔ HÌNH")
    training_logger.info("="*60)
    
    # Tạo DataFrame 
    df_report = pd.DataFrame(final_results).T
    df_report = df_report.sort_values("RMSE") 

    training_logger.info(df_report.round(4).to_string())
    
    # 3. Chọn Model tốt nhất
    best_name = min(final_results, key=lambda k: final_results[k]['RMSE'])
    training_logger.info(f"MODEL ĐƯỢC CHỌN: {best_name.upper()}")
    
    best_model = trainer.trained_models.get(best_name)
    
    if best_model:
        # 4. Vẽ biểu đồ dự báo (Chuẩn One-Step-Ahead)
        evaluator.visualize_prediction(
            model=best_model, 
            X_test=X_test, 
            y_test=y_test, 
            model_name=best_name, 
            raw_prices_series=dm.raw_prices
        )
        
        # 5. Phân tích lỗi
        evaluator.analyze_residuals(best_model, X_test, y_test, best_name)
        
        # 6. Giải thích SHAP
        evaluator.explain_shap(best_model, X_test, best_name)
    else:
        training_logger.error("Không tìm thấy object model trong trainer.")
else:
    training_logger.error("Chưa có kết quả train")
