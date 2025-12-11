import re
import numpy as np
import pandas as pd
from typing import Tuple, Generator
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
from utils.logger import get_training_logger

class DataModule:
    """
    DataModule 
    ----------------------------
    - Xử lý dữ liệu trung tâm, khớp nối ConfigManager và TimeSeriesSplitter.
    - Tự động xử lý tên cột Target nội bộ để tránh xung đột.
    """
    def __init__(self, config_manager):
        self.cfg = config_manager
        self.df, self.X, self.y = None, None, None
        self.raw_prices = None
        self.logger = get_training_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO DATA MODULE <<<")

    def run_pipeline(self) -> Tuple[pd.DataFrame, pd.Series]:
        """ 
        Pipeline xử lý dữ liệu chính 
        Trả về X, y đã sẵn sàng cho mô hình.
        """
        self.logger.info("--- THỰC HIỆN DATA PIPELINE ---")
        self._load_data()
        self._transform_target()
        self._generate_features()
        self._prepare_xy()
        self._select_features()
        
        # Làm sạch tên cột cho LightGBM
        self.X = self.X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        
        self.logger.info(f"Hoàn thành. X: {self.X.shape}, y: {self.y.shape}")
        return self.X, self.y
    
    
    def _load_data(self):
        """ 
        Đọc dữ liệu từ file và thiết lập chỉ số thời gian 
        """
        self.logger.info("*** ĐỌC DỮ LIỆU ***")
        path = self.cfg.data_config['raw_path']
        self.df = pd.read_csv(path)
        col_map = {c.lower(): c for c in self.df.columns}
        if 'date' in col_map:
            self.df[col_map['date']] = pd.to_datetime(self.df[col_map['date']])
            self.df.set_index(col_map['date'], inplace=True)
            self.df.sort_index(inplace=True)
        self.df.dropna(inplace=True)
        
        # Lưu giá gốc
        target_col = self.cfg.data_config['target_col']
        self.raw_prices = self.df[target_col].copy()

    def _transform_target(self):
        """ 
        Chuyển đổi cột Target nếu cần thiết 
        """
        target_col = self.cfg.data_config['target_col']
        if self.cfg.data_config.get('target_transform') == 'log_return':
            self.df = self.df[self.df[target_col] > 0]
            self.df['__Log_Return__'] = np.log(self.df[target_col] / self.df[target_col].shift(1))
            self.target_internal = '__Log_Return__'
        else:
            self.target_internal = target_col
        self.df.dropna(inplace=True)

    def _generate_features(self):
        """ 
        Khởi tạo các đặc trưng (features) từ dữ liệu thô 
        """
        self.logger.info("*** TẠO ĐẶC TRƯNG ***")
        lags = self.cfg.data_config['lags']
        windows = self.cfg.data_config['rolling_windows']
        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        
        key_cols = numeric_cols[:10] 
        if self.target_internal not in key_cols: key_cols.append(self.target_internal)

        new_cols = []
        for col in key_cols:
            for lag in lags:
                new_cols.append(self.df[col].shift(lag).rename(f"{col}_lag{lag}"))
            
            if col == self.target_internal:
                for w in windows:
                    new_cols.append(self.df[col].rolling(w).mean().rename(f"{col}_roll_mean_{w}"))
                    new_cols.append(self.df[col].rolling(w).std().rename(f"{col}_roll_std_{w}"))
        
        self.df = pd.concat([self.df] + new_cols, axis=1)
        self.df.dropna(inplace=True)

    def _prepare_xy(self):
        """
        Chuẩn bị tập X, y cho mô hình 
        """
        self.logger.info("*** CHUẨN BỊ X, y ***")
        self.final_target = '__Target_NextDay__'
        self.df[self.final_target] = self.df[self.target_internal].shift(-1)
        self.df.dropna(subset=[self.final_target], inplace=True)
        
        drop_cols = [self.final_target, self.target_internal]
        if self.target_internal != self.cfg.data_config['target_col']:
            drop_cols.append(self.cfg.data_config['target_col'])
            
        self.y = self.df[self.final_target]
        self.X = self.df.drop(columns=drop_cols, errors='ignore').select_dtypes(include=np.number).astype(np.float32)

    def _select_features(self):
        """ 
        Lựa chọn đặc trưng nếu được bật 
        """
        self.logger.info("*** LỰA CHỌN ĐẶC TRƯNG ***")
        if not self.cfg.data_config['selection']['enable']: return
        
        # 1. Correlation Filter
        corr = self.X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.98)]
        self.X.drop(columns=to_drop, inplace=True)
        
        # 2. Mutual Info (Chỉ lấy top features để giảm nhiễu)
        # Nếu X còn quá nhiều cột (>30), cắt bớt. tùy thuộc vào bộ dữ liệu để chọn số cột phù hợp
        if self.X.shape[1] > 30:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(self.X, self.y)
            mi = pd.Series(mi, index=self.X.columns)
            top_cols = mi.nlargest(30).index.tolist()
            self.X = self.X[top_cols]

class TimeSeriesSplitter:
    """
    TimeSeriesSplitter 
    -----------------------------------------
    Chuyên trách việc sinh ra các chỉ số (indices) để chia tập Train/Test.
    - Input: Nhận X, y từ DataModule.
    - Output: Trả về Train/Test set chuẩn để đưa vào Model.
    - Logic: Rolling Window Backtest (Walk-Forward Validation) với khoảng hở (Purge).
    """

    def __init__(self, config_manager):
        self.cfg = config_manager
        self.logger = get_training_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO TIME SERIES SPLITTER <<<")
        
        # Load cấu hình
        split_cfg = self.cfg.split_config
        self.train_window = split_cfg.get('train_window', 1200) # Kích thước cửa sổ huấn luyện
        self.test_window = split_cfg.get('test_window', 60)     # Kích thước cửa sổ kiểm tra (Step size)
        self.purge = split_cfg.get('purge', 0)                  # Khoảng hở an toàn

    def split(self, X: pd.DataFrame, y: pd.Series = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Tạo generator trả về index cho Cross-Validation.
        Tương thích chuẩn với Scikit-Learn (có thể dùng trong GridSearchCV).
        """
        self.logger.info("*** TẠO CÁC FOLD CHIA DỮ LIỆU ***")
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Validate Sort: Time Series bắt buộc dữ liệu phải theo thứ tự thời gian
        if isinstance(X.index, pd.DatetimeIndex) and not X.index.is_monotonic_increasing:
             self.logger.warning("CẢNH BÁO: Index thời gian chưa được sắp xếp! Đang sort lại index ảo...")

        # Logic tính toán cửa sổ trượt
        # Bắt đầu tại điểm đủ để có 1 cửa sổ train + 1 khoảng purge
        current_idx = self.train_window + self.purge
        
        fold_count = 0
        while current_idx + self.test_window <= n_samples:
            # 1. Định vị cửa sổ Train
            train_start = current_idx - self.purge - self.train_window
            train_end = current_idx - self.purge
            
            # 2. Định vị cửa sổ Test
            test_start = current_idx
            test_end = current_idx + self.test_window
            
            # Yield indices (trả về mảng chỉ số dòng)
            yield indices[train_start:train_end], indices[test_start:test_end]
            
            fold_count += 1
            
            # 3. Trượt cửa sổ (Walk Forward)
            # Ta trượt đi đúng bằng độ dài test_window để các tập test không trùng nhau (non-overlapping)
            current_idx += self.test_window
            
        self.logger.info(f"Đã tạo {fold_count} folds (Train Window: {self.train_window}, Test Window: {self.test_window}, Purge: {self.purge})")

    def get_holdout_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Lấy fold cuối cùng (gần hiện tại nhất) để làm Final Test (Hold-out set).
        Dùng cho việc đánh giá Model sau cùng.
        """
        # Lấy tất cả các fold có thể chia được
        splits = list(self.split(X, y))
        
        if not splits:
            raise ValueError(
                f"Dữ liệu quá ít ({len(X)} dòng) để chia fold với TrainWindow={self.train_window}!"
                " -> Hãy giảm 'train_window' trong Config hoặc nạp thêm dữ liệu."
            )
            
        # Lấy fold cuối cùng
        last_train_idx, last_test_idx = splits[-1]
        
        X_train, X_test = X.iloc[last_train_idx], X.iloc[last_test_idx]
        y_train, y_test = y.iloc[last_train_idx], y.iloc[last_test_idx]
        
        self.logger.info(f"*** KẾT QUẢ CHIA DỮ LIỆU (CV Fold) ***")
        self.logger.debug(f"Train Period: {X_train.index[0].date()} -> {X_train.index[-1].date()} ({len(X_train)} mẫu)")
        self.logger.info(f"Test Period : {X_test.index[0].date()} -> {X_test.index[-1].date()} ({len(X_test)} mẫu)")
        
        return X_train, X_test, y_train, y_test
    
class VisualizationModule:
    """
    VisualizationModule
    -------------------
    Chuyên trách vẽ biểu đồ báo cáo cho TSDataSplitter.
    """
    def __init__(self):
        sns.set_theme(style="whitegrid")
        self.logger = get_training_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO VISUALIZATION MODULE <<<")

    def plot_cv_indices(self, splitter: TimeSeriesSplitter, X: pd.DataFrame, y: pd.Series):
        """
        Vẽ sơ đồ minh họa cách chia dữ liệu Train/Test/Purge.
        """
        splits = list(splitter.split(X, y))
        n_splits = len(splits)
        
        if n_splits == 0:
            self.logger.error("Không có fold nào để vẽ!")
            return

        self.logger.info("Vẽ sơ đồ chia dữ liệu Train/Test/Purge...")
        plt.figure(figsize=(12, n_splits * 0.6))
        
        # Vẽ từng fold
        for i, (train_idx, test_idx) in enumerate(splits):
            # Lấy index thời gian tương ứng để vẽ trục X cho chuẩn
            # (Nếu vẽ theo số thứ tự dòng thì dùng train_idx trực tiếp)
            
            # Vẽ thanh Train (Màu xanh)
            plt.scatter(train_idx, [i] * len(train_idx), c='#1f77b4', marker='|', s=50, lw=2, label='Train' if i==0 else "")
            
            # Vẽ thanh Test (Màu cam)
            plt.scatter(test_idx, [i] * len(test_idx), c='#ff7f0e', marker='|', s=50, lw=2, label='Test' if i==0 else "")
            
            # Khoảng trắng giữa Xanh và Cam là Purge Gap

        plt.title(f"Cross-Validation Strategy: Rolling Window (Purge={splitter.purge})", fontsize=14)
        plt.xlabel("Sample Index", fontsize=12)
        plt.ylabel("Fold Number", fontsize=12)
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.yticks(range(n_splits), [f"Fold {i}" for i in range(n_splits)])
        plt.tight_layout()
        plt.show()
        self.logger.info("Đã hoàn thành vẽ sơ đồ chia dữ liệu.")

    def plot_predictions(self, y_true, y_pred, title="Model Prediction"):
        """Vẽ so sánh Giá thực vs Dự báo"""
        self.logger.info("Vẽ biểu đồ so sánh Giá thực vs Dự báo...")
        plt.figure(figsize=(15, 6))
        plt.plot(y_true.values, label='Actual Value', color='black', alpha=0.7)
        plt.plot(y_pred, label='Predicted Value', color='red', linestyle='--')
        plt.title(title)
        plt.legend()
        plt.show()
        self.logger.info("Đã hoàn thành vẽ biểu đồ dự báo.")