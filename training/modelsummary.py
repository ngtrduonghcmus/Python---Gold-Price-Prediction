import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib
import shap
from pathlib import Path
from utils.logger import get_training_logger

class ModelVisualizer:
    """
    ModelVisualizer (Professional & Accurate)
    ----------------------------------------
    Bộ công cụ trực quan hóa toàn diện: Leaderboard, Forecast, SHAP.
    """
    def __init__(self, config_manager):
        self.cfg = config_manager
        self.save_dir = Path(self.cfg.output_config.get('metrics_dir', './results')) / "plots"
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