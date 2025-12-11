import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from utils.logger import get_training_logger
    
# Style chuẩn báo cáo
plt.style.use("seaborn-whitegrid")
sns.set_palette("viridis")


class ModelTrainer:
    """
    ModelTrainer bộ não của cả chương trình.
    Thực hiện quá trình training sao khi đã có các tham số tôi ưu
    Lưu lại kết quả của từng mô hình chạy được
    """
    def __init__(self, config_manager, raw_prices_series=None):
        self.cfg = config_manager
        self.raw_prices = raw_prices_series 
        self.logger = get_training_logger(self.__class__.__name__)
        self.model_dir = Path(self.cfg.output_config.get('model_dir', './models'))
        self.result_dir = Path(self.cfg.output_config.get('result_dir', './results'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.best_params_cache = self._load_best_params()
        
        self.logger.info(">>> KHỞI TẠO MODEL TRAINER <<<")

    def _load_best_params(self):
        """ 
        Nạp tham số tối ưu từ file JSON nếu có 
        """
        
        path = self.result_dir / "best_hyperparameters.json"
        if path.exists():
            try: 
                self.logger.info("Đã nạp tham số tối ưu từ JSON.")
                return json.load(path.open())
            except: pass
        return {}

    def _get_model_instance(self, model_name: str):
        """ 
        Tạo instance model với tham số gộp (Base + Tuned) 
        """
        # Lấy tham số gộp (Base + Tuned)
        self.logger.info(f"*** TẠO MÔ HÌNH CUỐI CÙNG: {model_name} ***")
        final_params = self.cfg.get_model_params(model_name, is_tuning=False).copy()
        model_key = model_name.lower()
        
        if model_key in self.best_params_cache:
            tuned = self.best_params_cache[model_key]
            if 'n_jobs' in tuned: tuned.pop('n_jobs')
            final_params.update(tuned)

        # Khởi tạo & Xử lý xung đột tham số
        if model_key == 'xgboost': 
            final_params.pop('verbosity', None); final_params.pop('n_jobs', None)
            return xgb.XGBRegressor(**final_params, verbosity=0, n_jobs=-1)
            
        elif model_key == 'catboost': 
            final_params.pop('verbose', None); final_params.pop('allow_writing_files', None); final_params.pop('thread_count', None); final_params.pop('n_jobs', None)
            return CatBoostRegressor(**final_params, verbose=0, allow_writing_files=False, thread_count=-1)
            
        elif model_key == 'lightgbm': 
            final_params.pop('verbose', None); final_params.pop('verbosity', None); final_params.pop('n_jobs', None)
            # LightGBM init: verbose=-1 để tắt log
            return lgb.LGBMRegressor(**final_params, verbose=-1, n_jobs=-1)
            
        elif model_key == 'random_forest': 
            final_params.pop('n_jobs', None); final_params.pop('verbose', None)
            return RandomForestRegressor(**final_params, n_jobs=-1)
            
        else: 
            self.logger.error(f"Mô hình không xác định: {model_name}")
            raise ValueError(f"Mô hình không xác định: {model_name}")

    def train_final_model(self, X_train, y_train, X_test, y_test):
        self.logger.info("*** BẮT ĐẦU TRAINING MÔ HÌNH CUỐI CÙNG ***")
        
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        joblib.dump(scaler, self.model_dir / "final_scaler.joblib")
        
        results = {} 
        
        for model_name in self.cfg.train_config['models']:
            self.logger.info(f"Đang huấn luyện {model_name}...")
            try:
                model = self._get_model_instance(model_name)
                
                # ---Tách lệnh fit cho từng loại model ---
                if model_name == 'xgboost':
                    # XGBoost hỗ trợ verbose=False
                    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
                    
                elif model_name == 'catboost':
                    # CatBoost hỗ trợ verbose=False
                    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)
                    
                elif model_name == 'lightgbm':
                    # LightGBM KHÔNG được truyền verbose vào fit()
                    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)])
                    
                else: # Random Forest
                    model.fit(X_train_s, y_train)
                
                # Predict
                y_pred_log = model.predict(X_test_s)
                
                # Metrics
                price_rmse = -1.0; price_r2 = -1.0
                if self.raw_prices is not None and X_test.index[0] in self.raw_prices.index:
                    prev_prices = self.raw_prices.shift(1).loc[X_test.index]
                    true_future_prices = prev_prices * np.exp(y_test)
                    pred_future_prices = prev_prices * np.exp(y_pred_log)
                    
                    valid = ~np.isnan(true_future_prices) & ~np.isnan(pred_future_prices)
                    if np.sum(valid) > 0:
                        price_rmse = np.sqrt(mean_squared_error(true_future_prices[valid], pred_future_prices[valid]))
                        price_r2 = r2_score(true_future_prices[valid], pred_future_prices[valid])
                    
                    self.logger.info(f"{model_name.upper()}: Price R2={price_r2:.4f} | RMSE=${price_rmse:.2f}")

                true_dir = np.sign(y_test); pred_dir = np.sign(y_pred_log)
                da = accuracy_score(true_dir[true_dir!=0], pred_dir[true_dir!=0])
                
                results[model_name] = {"RMSE": price_rmse, "R2": price_r2, "Dir_Accuracy": da}
                self.trained_models[model_name] = model
                joblib.dump(model, self.model_dir / f"{model_name}_final.joblib")
                
            except Exception as e:
                self.logger.error(f"Huấn luyện thất bại {model_name}: {e}")
                # import traceback; traceback.print_exc() # Uncomment để xem chi tiết lỗi nếu cần

        # Lưu file JSON
        output_file = self.result_dir / "final_scores.json"
        with open(output_file, "w") as f: 
            json.dump(results, f, indent=4)
            
        self.logger.info(f"Kết quả lưu tại: {output_file}")
        return results


class HyperParameterOptimizer:
    """
    HyperParameterOptimizer
    Tối ưu tham siêu tham số cho công tác training
    Tham số tối ưu được tìm bằng việc chạy nhiều lần các CV_fold
    Lưu kết quả tuning ra JSON để tái sử dụng
    """
    def __init__(self, config_manager, splitter, X, y):
        self.cfg = config_manager
        self.splitter = splitter
        self.X, self.y = X, y
        self.results_dir = Path(self.cfg.output_config.get('result_dir', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_training_logger(self.__class__.__name__)
        # Pre-calculate CV splits
        self.cv_splits = list(self.splitter.split(self.X, self.y))[-3:] 

        self.logger.info(">>> KHỞI TẠO HYPERPARAMETER OPTIMIZER <<<")

    def _get_search_space(self, trial, model_name):
        """ 
        Định nghĩa không gian tìm kiếm tham số cho từng mô hình
        """
        if model_name == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9)
            }
        elif model_name == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True)
            }
        elif model_name == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50)
            }
        elif model_name == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 134),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
            }
        return {}

    def _objective(self, trial, model_name):
        """ 
        Hàm mục tiêu tối ưu cho Optuna
        """
        search_params = self._get_search_space(trial, model_name)
        # Lấy base params nhưng KHÔNG lấy n_jobs hay verbosity từ config để tránh trùng
        base_params = self.cfg.get_model_params(model_name, is_tuning=True).copy()
        
        # Merge tham số
        final_params = {**base_params, **search_params}

        scores = []
        try:
            for train_idx, val_idx in self.cv_splits:
                # Prepare Data
                X_train = self.X.iloc[train_idx]; y_train = self.y.iloc[train_idx]
                X_val = self.X.iloc[val_idx]; y_val = self.y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_val_s = scaler.transform(X_val)

                # --- INIT & FIT THEO TỪNG MODEL  ---
                if model_name == 'xgboost':
                    final_params.pop('verbosity', None)
                    final_params.pop('n_jobs', None)
                    # Khởi tạo 
                    model = xgb.XGBRegressor(**final_params, n_jobs=1, verbosity=0)
                    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
                    
                elif model_name == 'catboost':
                    final_params.pop('verbose', None)
                    final_params.pop('thread_count', None)
                    final_params.pop('allow_writing_files', None)
                    final_params.pop('n_jobs', None) 
                    
                    model = CatBoostRegressor(**final_params, thread_count=1, verbose=0, allow_writing_files=False)
                    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False, early_stopping_rounds=20)
                    
                elif model_name == 'lightgbm':
                    final_params.pop('verbose', None)
                    final_params.pop('n_jobs', None)
                    final_params.pop('verbosity', None)
                    
                    model = lgb.LGBMRegressor(**final_params, n_jobs=1, verbose=-1)
                    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)])
                    
                elif model_name == 'random_forest':
                    final_params.pop('n_jobs', None)
                    model = RandomForestRegressor(**final_params, n_jobs=1)
                    model.fit(X_train_s, y_train)

                # Eval
                preds = model.predict(X_val_s)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                scores.append(rmse)

            return np.mean(scores)

        except Exception as e:
            # In lỗi
            self.logger.error(f"{model_name} gặp lỗi trong quá trình tối ưu: {str(e)}")
            return float('inf')
        
    def run_optimization(self):
        """
        Chạy tối ưu tham số cho tất cả mô hình được chỉ định
        """
        models = self.cfg.train_config['models']
        best_results = {} 
        
        self.logger.info("*** TỐI ƯU SIÊU THAM SỐ CHO CÁC MÔ HÌNH ***")
        
        for model in models: 
            self.logger.info(f"Bắt đầu tối ưu tham số cho {model}...")
            
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda t: self._objective(t, model), 
                           n_trials=self.cfg.optim_config.get('n_trials', 15), 
                           n_jobs=1) 
            

            if study.best_value == float('inf'):
                self.logger.debug(f"{model} failed.")

                best_results[model] = self.cfg.get_model_params(model, is_tuning=False)
            else:
                self.logger.debug(f"Best {model}: RMSE={study.best_value:.5f}")
                # Merge tham số
                best_results[model] = {**self.cfg.get_model_params(model, is_tuning=False), **study.best_params}
        
        json_path = self.results_dir / "best_hyperparameters.json"
        with open(json_path, 'w') as f:
            json.dump(best_results, f, indent=4)
            
        self.logger.info(f"Đã lưu tham số tối ưu vào: {json_path}.")
        return best_results

class ModelEvaluator:
    """
    ModelEvaluator (Professional & Accurate)
    ----------------------------------------
    Bộ công cụ trực quan hóa toàn diện: Leaderboard, Forecast, SHAP.
    """
    def __init__(self, config_manager):
        self.cfg = config_manager
        self.save_dir = Path(self.cfg.output_config.get('result_dir', './results')) / "plots"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_training_logger(self.__class__.__name__)
        try:
            self.scaler = joblib.load(Path(self.cfg.output_config.get('model_dir')) / "final_scaler.joblib")
        except: 
            self.logger.warning("Không tìm thấy scaler đã lưu. Sẽ không chuẩn hóa dữ liệu đầu vào.")
            self.scaler = None

        self.logger.info(">>> KHỞI TẠO MODEL EVALUATOR <<<")

    def _finalize_plot(self, filename: str, title: str):
        """ 
        Biểu đồ hoàn chỉnh & lưu file. 
        """
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.show()
        self.logger.info(f"Đã lưu biểu đồ: {filename}")

    def plot_leaderboard(self, final_results: dict):
        """ 
        Biểu đồ Leaderboard tổng hợp. 
        """
        if not final_results: 
            self.logger.warning("Không có kết quả để vẽ biểu đồ Leaderboard.")
            return
        df = pd.DataFrame(final_results).T.reset_index().rename(columns={'index': 'Model'})
        df = df.sort_values(by='RMSE')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # RMSE
        sns.barplot(x='RMSE', y='Model', data=df, palette='Reds_r', ax=ax1)
        ax1.set_title("Price RMSE (Sai số USD - Thấp hơn là tốt)")
        for i, v in enumerate(df['RMSE']): ax1.text(v, i, f" ${v:.2f}", va='center', fontweight='bold')

        # Direction Accuracy
        df_acc = df.sort_values(by='Dir_Accuracy', ascending=False)
        sns.barplot(x='Dir_Accuracy', y='Model', data=df_acc, palette='Greens_r', ax=ax2)
        ax2.set_title("Direction Accuracy (Đoán đúng hướng - Cao hơn là tốt)")
        ax2.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='50% Random')
        for i, v in enumerate(df_acc['Dir_Accuracy']): ax2.text(v, i, f" {v:.1%}", va='center', fontweight='bold')
        
        self._finalize_plot("leaderboard.png", "KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")

    def visualize_prediction(self, model, X_test, y_test, model_name, raw_prices_series=None):
        """
        Vẽ dự báo theo cơ chế One-Step-Ahead (Sát thực tế nhất).
        """
        # 1. Scale & Predict
        if self.scaler:
            # Ép về DataFrame để giữ tên cột cho SHAP/LGBM
            X_s = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        else:
            X_s = X_test
            
        y_pred_log = model.predict(X_s)

        # 2. Logic Khôi phục Giá (SỬA LẠI CHUẨN)
        # Price(t+1) = Price(t) * exp(Return(t+1))
        # X_test.index là ngày T. Ta cần giá ngày T.
        
        if raw_prices_series is not None:
            # Lấy giá đóng cửa tại ngày T (Inputs) để dự báo T+1
            # raw_prices_series index là Date. X_test index cũng là Date.
            # Align dữ liệu:
            common_idx = X_test.index.intersection(raw_prices_series.index)
            
            if len(common_idx) == 0:
                self.logger.error("Lỗi: Index không khớp giữa X_test và Raw Prices.")
                return

            base_prices = raw_prices_series.loc[common_idx]
            
            # Tính giá thực tế ngày mai (Target)
            # y_test là log return từ T -> T+1
            true_next_prices = base_prices * np.exp(y_test.loc[common_idx])
            
            # Tính giá dự báo ngày mai
            pred_next_prices = base_prices * np.exp(y_pred_log)
            
            plot_data_true = true_next_prices
            plot_data_pred = pred_next_prices
            ylabel, suffix = "Giá Vàng (USD)", "(Dự báo T+1)"
        else:
            # Fallback nếu không có giá gốc
            plot_data_true = y_test
            plot_data_pred = pd.Series(y_pred_log, index=y_test.index)
            ylabel, suffix = "Log Return", "(Chưa khôi phục)"

        # 3. Vẽ đồ thị (Zoom vào 100 điểm cuối)
        plt.figure(figsize=(16, 7))
        limit = 100 
        
        # Vẽ vùng sai số
        plt.fill_between(plot_data_true.index[-limit:], 
                         plot_data_true.values[-limit:], 
                         plot_data_pred.values[-limit:], 
                         color='gray', alpha=0.1, label='Sai số (Spread)')

        plt.plot(plot_data_true.index[-limit:], plot_data_true.values[-limit:], 
                 label='Thực tế (Actual)', color='#2C3E50', lw=2.5)
        
        plt.plot(plot_data_pred.index[-limit:], plot_data_pred.values[-limit:], 
                 label=f'Dự báo AI ({model_name})', color='#C0392B', lw=2, linestyle='--')
        
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, alpha=0.3)
        
        self._finalize_plot(f"forecast_{model_name}.png", f"DỰ BÁO GIÁ VÀNG - {model_name.upper()} {suffix}")

    def analyze_residuals(self, model, X_test, y_test, model_name):
        """
        Phân tích lỗi.
        """


        if self.scaler:
            X_s = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        else:
            X_s = X_test
            
        y_pred = model.predict(X_s)
        residuals = y_test - y_pred

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Histogram
        sns.histplot(residuals, kde=True, ax=ax1, color='#3498DB', bins=40)
        ax1.axvline(0, color='red', linestyle='--', lw=2)
        ax1.set_title("Phân phối độ lệch (Residuals)")
        ax1.set_xlabel("Mức độ sai lệch (Log Return)")

        # Confusion Matrix
        true_sign = np.sign(y_test).astype(int)
        pred_sign = np.sign(y_pred).astype(int)
        mask = true_sign != 0
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_sign[mask], pred_sign[mask], labels=[1, -1])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['Tăng', 'Giảm'], yticklabels=['Tăng', 'Giảm'], cbar=False, annot_kws={"size": 14})
        ax2.set_title("Khả năng bắt sóng (Trend Accuracy)")
        ax2.set_xlabel("AI Dự báo"); ax2.set_ylabel("Thị trường Thực tế")

        self._finalize_plot(f"residuals_{model_name}.png", f"PHÂN TÍCH HIỆU NĂNG - {model_name.upper()}")

    def explain_shap(self, model, X_test, model_name):
        """
        Giải thích SHAP 
        """
        try:
            self.logger.info(f"Đang tính SHAP cho {model_name}...")
            if self.scaler:
                X_s = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
            else:
                X_s = X_test
            
            # Lấy mẫu nhỏ để vẽ cho nhanh (50 mẫu cuối)
            X_sample = X_s.iloc[-50:] 
            
            if model_name.lower() in ['xgboost', 'catboost', 'lightgbm', 'random_forest']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.Explainer(model, X_sample)
            
            shap_values = explainer.shap_values(X_sample, check_additivity=False)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15, cmap='coolwarm')
            self._finalize_plot(f"shap_{model_name}.png", f"TOP YẾU TỐ ẢNH HƯỞNG - {model_name.upper()}")
            
        except Exception as e:
            self.logger.error(f"Lỗi SHAP: {e}")