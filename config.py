import yaml
import argparse
from pathlib import Path
from typing import Any, Dict
from utils.logger import get_training_logger

class ConfigManager:
    """
    Quản lý cấu hình tập trung cho hệ thống dự báo đa mô hình.
    
    ENGINEERING NOTES:
    1. Resource Locking: Tự động điều phối số luồng (n_jobs) để tránh Deadlock khi Tuning.
    2. Algo Selection: Ưu tiên 'Histogram-based' cho Tree Models để tối ưu Cache CPU.
    3. Đảm bảo các mô hình đều dược cấu hình có thể chạy được
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = {}
        self.logger = get_training_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO CONFIG MANAGER <<<")
        
        # Load hoặc tạo mới cấu hình tối ưu
        if self.config_path.exists():
            self._load_from_file()
        else:
            self.logger.warning(f"Config '{self.config_path}' chưa tồn tại. Đang tạo cấu hình tối ưu mặc định...")
            self._config = self._get_optimized_default_config()
            self.save_config()
            
        self._override_from_args()
        self._validate_config()

    def _load_from_file(self):
        """ 
        Tải cấu hình từ file YAML.
        """
        with open(self.config_path, 'r', encoding='utf-8') as f:
            loaded = yaml.safe_load(f)
            self._config = loaded if loaded else self._get_optimized_default_config()
            self.logger.info(f"Đã tải cấu hình từ '{self.config_path}'")

    def _validate_config(self):
        """ 
        Kiểm tra và bổ sung các phần cấu hình còn thiếu. 
        """
        defaults = self._get_optimized_default_config()
        for section in ['data', 'splitting', 'training', 'optimization', 'hardware', 'output']:
            if section not in self._config:
                self._config[section] = defaults[section]

    def _override_from_args(self):
        """
        Cho phép ghi đè cấu hình từ dòng lệnh (Command Line Interface)
        """
        try:
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument("--gpu", type=str)
            parser.add_argument("--trials", type=int)
            parser.add_argument("--models", type=str) # Ví dụ: --models xgboost,catboost
            
            args, _ = parser.parse_known_args()
            
            if args.gpu: 
                self._config['hardware']['use_gpu'] = (args.gpu.lower() == 'true')
            if args.trials: 
                self._config['optimization']['n_trials'] = args.trials
            if args.models:
                self._config['training']['models'] = args.models.split(',')
            self.logger.info("Đã ghi đè cấu hình từ dòng lệnh nếu có.")
        except Exception as e:
            self.logger.error(f"Không có tham số dòng lệnh: {e}")

    def save_config(self):
        """ 
        Lưu cấu hình hiện tại ra file YAML. 
        """
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f: 
            yaml.dump(self._config, f, sort_keys=False)
            self.logger.info(f"Đã lưu cấu hình vào '{self.config_path}'")

    #####---CẤU HÌNH MẶC ĐỊNH CHUẨN---#####
    def _get_optimized_default_config(self) -> Dict[str, Any]:
        return {
            "project": {"name": "Gold_Price_Forecast_Pro", "seed": 42},
            "data": {
                "raw_path": "clean_data.csv",
                "target_col": "Adj Close",
                "target_transform": "log_return", 
                "lags": [1, 2, 3], 
                "rolling_windows": [5, 20],
                "selection": {"enable": True, "max_features": 20} 
            },
            "splitting": {
                "method": "rolling_window", 
                "test_size": 0.15,
                "train_window": 1000, ##### có thể đổi độ dài cửa sổ train
                "test_window": 60,    
                "purge": 0 # dữ liệu chỉ có hơn 1700 nên tạm thời chưa đặt vùng cấm
            },
            "training": {
                "models": ["xgboost", "lightgbm", "catboost", "random_forest"], 
                "metric": "rmse"
            },
            "optimization": {
                "n_trials": 20, "timeout": 600, "n_jobs_optuna": 4
            },
            "hardware": {
                "use_gpu": False, "n_jobs": -1
            },
            "output": {"model_dir": "./models", "result_dir": "./results"}
        }

    def get_model_params(self, model_name: str, is_tuning: bool = False) -> Dict[str, Any]:
        """
        Trả về tham số cấu hình mặc định cho từng mô hình.
        """

        n_jobs_model = 1 if is_tuning else self.hardware_config.get('n_jobs', -1)
        base = {"random_state": 42}
        
        # Các tham số đầu vào quan trọng của từng mô hình
        if model_name == "xgboost":
            return {**base, "n_estimators": 200, "learning_rate": 0.05, "max_depth": 5, 
                    "n_jobs": n_jobs_model, "tree_method": "hist", "verbosity": 0}
        elif model_name == "lightgbm":
            return {**base, "n_estimators": 200, "learning_rate": 0.05, "n_jobs": n_jobs_model, 
                    "verbose": -1, "force_col_wise": True}
        elif model_name == "catboost":
            return {**base, "iterations": 500, "learning_rate": 0.05, "depth": 6, 
                    "thread_count": n_jobs_model, "allow_writing_files": False, "verbose": 0, "task_type": "CPU"}
        elif model_name == "random_forest":
            return {**base, "n_estimators": 100, "max_depth": 10, "n_jobs": n_jobs_model} ### có thể giảm số lần thử 
        return base

    # --- Properties ---
    @property
    def data_config(self): return self._config['data']
    @property
    def train_config(self): return self._config['training']
    @property
    def optim_config(self): return self._config['optimization']
    @property
    def hardware_config(self): return self._config['hardware']
    @property
    def output_config(self): return self._config['output']
    @property
    def split_config(self): return self._config['splitting']
