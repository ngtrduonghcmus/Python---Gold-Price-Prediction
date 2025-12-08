import pandas as pd
import numpy as np
import seaborn as sns
import logging

import matplotlib.pyplot as plt

class Visualizer:
    """
    Lớp trực quan hóa dữ liệu chuyên sâu cho Time Series.
    Tập trung vào EDA: Xu hướng (Trend), Mùa vụ (Seasonality), và Tương quan (Correlation).
    """

    # =========================================================
    # 0. KHỞI TẠO & CẤU HÌNH
    # =========================================================
    def __init__(self, df: pd.DataFrame, target_col: str = "Adj Close"):
        """
        Args:
            df (pd.DataFrame): DataFrame đã được làm sạch từ DataPreprocessor (Date là Index).
            target_col (str): Tên cột mục tiêu (thường là giá Vàng).
        """
        self.df = df.copy()
        self.target = target_col
        
        # Cấu hình Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(handler)

        # Định nghĩa bảng màu chuyên nghiệp (Financial Theme)
        self.colors = {
            "gold": "#D4AF37",      # Màu vàng kim
            "primary": "#2C3E50",   # Màu xanh đậm (cho trục phụ)
            "up": "#27AE60",        # Xanh lá (Tăng)
            "down": "#C0392B",      # Đỏ (Giảm)
            "grid": "#ECF0F1",      # Lưới mờ
            "corr_pos": "#E74C3C",  # Tương quan dương nóng
            "corr_neg": "#3498DB"   # Tương quan âm lạnh
        }

        # Mapping các nhóm dữ liệu (Dựa trên prefix của bộ dữ liệu 80 cột)
        self.prefix_groups = {
            "Gold": ["Adj Close", "Close"],
            "Oil (USO)": ["USO_"],
            "Brent Oil": ["OF_"],
            "WTI Oil": ["OS_"],
            "Silver": ["SF_"],
            "Platinum": ["PLT_"],
            "Palladium": ["PLD_"],
            "USD Index": ["USDI_"],
            "US Bond 10Y": ["USB_"],
            "S&P 500": ["SP_"],
            "Dow Jones": ["DJ_"],
            "Gold Miners": ["GDX_"],
        }
        
        # Validate Target
        if self.target not in self.df.columns:
            self.logger.warning(f"Không tìm thấy cột target '{self.target}'. Đang tìm cột thay thế...")
            alt_col = self._get_best_col(self.prefix_groups["Gold"])
            if alt_col:
                self.target = alt_col
                self.logger.info(f"Đã chuyển target sang: {self.target}")
            else:
                raise ValueError("Không tìm thấy cột dữ liệu giá Vàng hợp lệ.")

    # =========================================================
    # UTIL: HÀM HỖ TRỢ
    # =========================================================
    def _get_best_col(self, prefix_list):
        """Tìm cột tốt nhất trong danh sách prefix."""
        candidates = []
        for p in prefix_list:
            candidates += [c for c in self.df.columns if c.startswith(p) or c == p]
        
        if not candidates: return None
        
        # Ưu tiên theo thứ tự: Adj Close > Close > Price > Open
        keywords = ["Adj Close", "Ajclose", "Close", "Price"]
        for k in keywords:
            for c in candidates:
                if k.lower() in c.lower(): return c
        return candidates[0]

    def _check_col(self, col_name):
        """Kiểm tra cột có tồn tại không để tránh crash."""
        if col_name and col_name in self.df.columns:
            return True
        return False

    # =========================================================
    # 1. PHÂN TÍCH XU HƯỚNG (TREND ANALYSIS)
    # =========================================================
    def gold_trend(self):
        """Vẽ biểu đồ giá vàng kèm đường trung bình động (MA)."""
        plt.figure(figsize=(15, 7))
        
        # Giá đóng cửa
        plt.plot(self.df.index, self.df[self.target], color=self.colors["gold"], label="Giá Vàng", linewidth=1.5)
        
        # MA50 và MA200 (Chỉ số kỹ thuật cơ bản)
        ma50 = self.df[self.target].rolling(50).mean()
        ma200 = self.df[self.target].rolling(200).mean()
        
        plt.plot(self.df.index, ma50, color=self.colors["primary"], linestyle="--", linewidth=1, label="MA 50 (Ngắn hạn)")
        plt.plot(self.df.index, ma200, color="red", linestyle="-", linewidth=1.5, label="MA 200 (Dài hạn)")

        plt.title(f"Xu hướng Giá Vàng & Đường Trung Bình Động ({self.target})", fontsize=14)
        plt.xlabel("Thời gian")
        plt.ylabel("Giá (USD)")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 2.1. GOLD TREND SEGMENTS 
    # =========================================================
    def gold_trend_segments(self, window_years=2):
        """Chia nhỏ xu hướng theo từng giai đoạn (ví dụ: 2 năm/khung hình)."""
        # Tạo bản copy để không ảnh hưởng dữ liệu gốc
        df_plot = self.df[[self.target]].copy()
        
        # Lấy năm từ Index (Vì Date đã là Index)
        df_plot["year"] = df_plot.index.year
        unique_years = sorted(df_plot["year"].unique())
        
        if not unique_years:
            self.logger.warning("Không tìm thấy dữ liệu năm.")
            return

        # Tính toán số lượng hàng cho Subplots
        n_rows = (len(unique_years) + window_years - 1) // window_years
        
        # Tạo khung hình
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 5 * n_rows), sharex=False)
        if n_rows == 1: axes = [axes] # Xử lý trường hợp chỉ có 1 row

        for i, ax in enumerate(axes):
            # Xác định năm bắt đầu và kết thúc của khung hình này
            idx_start = i * window_years
            if idx_start >= len(unique_years): 
                fig.delaxes(ax) # Xóa khung hình thừa
                continue
                
            y_start = unique_years[idx_start]
            y_end = min(y_start + window_years, unique_years[-1] + 1)
            
            # Lọc dữ liệu theo năm
            seg = df_plot[(df_plot["year"] >= y_start) & (df_plot["year"] < y_end)]
            
            if seg.empty:
                fig.delaxes(ax)
                continue

            # --- VẼ BIỂU ĐỒ ---
            # Dùng self.colors thay vì COLORS
            ax.plot(seg.index, seg[self.target], color=self.colors["gold"], linewidth=2, label="Giá Vàng")
            
            ax.set_title(f"Giai đoạn: {y_start} - {y_end - 1}", fontsize=12, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_ylabel("USD")
            
            # --- ĐÁNH DẤU ĐỈNH / ĐÁY ---
    
            max_idx = seg[self.target].idxmax()
            min_idx = seg[self.target].idxmin()
            
            max_val = seg.loc[max_idx, self.target]
            min_val = seg.loc[min_idx, self.target]
            
            # Dùng self.colors["up"] và ["down"]
            ax.scatter(max_idx, max_val, c=self.colors["up"], s=80, zorder=5, edgecolors='white', label="Đỉnh")
            ax.scatter(min_idx, min_val, c=self.colors["down"], s=80, zorder=5, edgecolors='white', label="Đáy")

            # Chỉ hiện legend cho subplot đầu tiên để đỡ rối
            if i == 0:
                ax.legend(loc="upper left")

        plt.tight_layout()
        plt.show()

    # =========================================================
    # 2. PHÂN TÍCH BIẾN ĐỘNG (VOLATILITY)
    # =========================================================
    def volatility_analysis(self):
        """Phân tích độ biến động rủi ro."""
        df_plot = self.df[[self.target]].copy()
        # Tính phần trăm thay đổi hàng ngày
        df_plot["Return"] = df_plot[self.target].pct_change()
        # Annualized Volatility (Window 30 ngày)
        df_plot["Volatility"] = df_plot["Return"].rolling(30).std() * np.sqrt(252) * 100

        fig, ax1 = plt.subplots(figsize=(15, 7))

        # Trục 1: Giá
        ax1.plot(df_plot.index, df_plot[self.target], color=self.colors["gold"], label="Giá Vàng")
        ax1.set_ylabel("Giá (USD)", color=self.colors["gold"], fontweight='bold')
        
        # Trục 2: Biến động
        ax2 = ax1.twinx()
        ax2.fill_between(df_plot.index, df_plot["Volatility"], color=self.colors["primary"], alpha=0.2, label="Độ biến động (Annualized)")
        ax2.plot(df_plot.index, df_plot["Volatility"], color=self.colors["primary"], alpha=0.6, linewidth=1)
        ax2.set_ylabel("Độ biến động (%)", color=self.colors["primary"], fontweight='bold')

        plt.title("Rủi ro vs Lợi nhuận: Giá Vàng và Độ biến động", fontsize=14)
        plt.show()

    # =========================================================
    # 3. PHÂN TÍCH MÙA VỤ (SEASONALITY)
    # =========================================================
    def seasonal_heatmap(self):
        """Vẽ Heatmap lợi nhuận trung bình theo Tháng và Năm."""
        df_plot = self.df[[self.target]].copy()
        df_plot['Year'] = df_plot.index.year
        df_plot['Month'] = df_plot.index.month
        
        # Tính lợi nhuận tháng
        monthly_return = df_plot.groupby(['Year', 'Month'])[self.target].last().pct_change() * 100
        pivot_table = monthly_return.unstack(level=1)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap="RdYlGn", center=0, annot=True, fmt=".1f", cbar_kws={'label': 'Lợi nhuận (%)'})
        plt.title("Heatmap Lợi nhuận Vàng theo Tháng/Năm", fontsize=14)
        plt.ylabel("Năm")
        plt.xlabel("Tháng")
        plt.show()

    # =========================================================
    # 4. PHÂN TÍCH TƯƠNG QUAN (CORRELATION)
    # =========================================================
    
    def correlation_matrix(self, top_n=15):
        """Heatmap tương quan giữa Vàng và các tài sản khác."""
        # Chỉ lấy cột số
        numeric_df = self.df.select_dtypes(include=np.number)
        
        # Tìm Top N cột tương quan nhất với Target
        corr_series = numeric_df.corrwith(numeric_df[self.target]).abs().sort_values(ascending=False)
        top_cols = corr_series.head(top_n).index
        
        corr_matrix = numeric_df[top_cols].corr()

        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Che nửa trên
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
        plt.title(f"Ma trận tương quan (Top {top_n} yếu tố ảnh hưởng)", fontsize=14)
        plt.show()

    def rolling_correlation(self, window=90):
        """Biểu đồ tương quan trượt (Rolling Correlation) với USD và Dầu."""
        usd_col = self._get_best_col(self.prefix_groups["USD Index"])
        oil_col = self._get_best_col(self.prefix_groups["Oil (USO)"])
        
        if not (usd_col and oil_col):
            self.logger.warning("Thiếu dữ liệu USD hoặc Oil để vẽ Rolling Correlation.")
            return

        df_plot = self.df[[self.target, usd_col, oil_col]].dropna()
        
        corr_usd = df_plot[self.target].rolling(window).corr(df_plot[usd_col])
        corr_oil = df_plot[self.target].rolling(window).corr(df_plot[oil_col])

        plt.figure(figsize=(15, 6))
        plt.plot(corr_usd.index, corr_usd, label=f"Vàng vs USD Index", color=self.colors["down"])
        plt.plot(corr_oil.index, corr_oil, label=f"Vàng vs Dầu (USO)", color=self.colors["up"])
        
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.title(f"Sự thay đổi mối tương quan theo thời gian (Window {window} ngày)", fontsize=14)
        plt.ylabel("Hệ số tương quan (-1 đến 1)")
        plt.legend()
        plt.show()
    # =========================================================
    # HELPER: TÍNH TƯƠNG QUAN & VẼ TRỤC (Refactored)
    # =========================================================
    def _calc_corr(self, col1, col2):
        """Hàm phụ trợ tính tương quan an toàn."""
        if col1 not in self.df.columns or col2 not in self.df.columns:
            return 0.0
        # Tự động loại bỏ NaN để tính chính xác
        valid_data = self.df[[col1, col2]].dropna()
        if valid_data.empty: return 0.0
        return valid_data[col1].corr(valid_data[col2])

    def _plot_dual_axis(self, col_name, title_prefix, color2=None):
        """Vẽ biểu đồ 2 trục Y (Dual Axis) để so sánh."""
        if not self._check_col(col_name): return
        
        # Mặc định màu nếu không truyền
        if color2 is None: color2 = self.colors["primary"]

        # Lấy dữ liệu (Dùng Index làm trục X)
        df_sub = self.df[[self.target, col_name]].dropna()
        corr = self._calc_corr(self.target, col_name)
        
        fig, ax1 = plt.subplots(figsize=(16, 6))
        
        # Trục 1: Giá Vàng
        ax1.plot(df_sub.index, df_sub[self.target], color=self.colors["gold"], label="Giá Vàng", linewidth=2)
        ax1.set_ylabel("Giá Vàng (USD)", color=self.colors["gold"], fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=self.colors["gold"])
        
        # Trục 2: Tài sản đối sánh
        ax2 = ax1.twinx()
        ax2.plot(df_sub.index, df_sub[col_name], color=color2, label=col_name, alpha=0.7, linewidth=1.5)
        ax2.set_ylabel(col_name, color=color2, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f"{title_prefix} (Corr: {corr:.2f})", fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Gộp Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.show()

    def _plot_scaled(self, cols, title):
        """Chuẩn hóa Min-Max về 0-1 để so sánh xu hướng."""
        valid_cols = [c for c in cols if self._check_col(c)]
        if not valid_cols: return

        plot_cols = [self.target] + valid_cols
        df_sub = self.df[plot_cols].dropna()
        
        # Min-Max Scaling
        normalized = (df_sub - df_sub.min()) / (df_sub.max() - df_sub.min())
        
        plt.figure(figsize=(16, 6))
        
        # Vẽ Vàng nổi bật
        plt.plot(df_sub.index, normalized[self.target], label="Vàng", linewidth=3, color=self.colors["gold"])
        
        # Vẽ các đường còn lại mờ hơn
        for c in valid_cols:
            corr = self._calc_corr(self.target, c)
            # Tự động chọn màu nếu có thể, hoặc để matplotlib tự chọn
            plt.plot(df_sub.index, normalized[c], label=f"{c} (r={corr:.2f})", alpha=0.6, linewidth=1.5)
            
        plt.title(title, fontsize=14)
        plt.ylabel("Giá trị chuẩn hóa (0-1)")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.show()

    # =========================================================
    # 4.1. CÁC BIỂU ĐỒ TƯƠNG QUAN CỤ THỂ
    # =========================================================
    def gold_vs_oil(self):
        col = self._get_best_col(self.prefix_groups["Oil (USO)"])
        self._plot_dual_axis(col, "Tương quan: Vàng vs Quỹ dầu USO", color2="black")

    def gold_vs_usd(self):
        col = self._get_best_col(self.prefix_groups["USD Index"])
        # USD thường nghịch đảo với Vàng -> dùng màu đỏ hoặc xanh dương đậm
        self._plot_dual_axis(col, "Tương quan Nghịch đảo: Vàng vs USD Index", color2=self.colors["down"])

    def gold_vs_crude_oil(self):
        cols = [self._get_best_col(self.prefix_groups[g]) for g in ["Brent Oil", "WTI Oil"]]
        self._plot_scaled(cols, "So sánh xu hướng: Vàng vs Dầu thô (Brent/WTI)")

    def gold_vs_equity(self):
        cols = [self._get_best_col(self.prefix_groups[g]) for g in ["S&P 500", "Dow Jones"]]
        self._plot_scaled(cols, "Vàng vs Thị trường Chứng khoán Mỹ")

    def gold_vs_metals(self):
        cols = [self._get_best_col(self.prefix_groups[g]) for g in ["Silver", "Platinum", "Palladium"]]
        self._plot_scaled(cols, "Vàng vs Nhóm Kim loại quý")

###
    # =========================================================
    # 4.2 HEATMAP ĐỘ TRỄ GIỮA CÁC TÀI SẢN (CROSS-LAG)
    # =========================================================
    def cross_asset_lag_heatmap(self, max_lag=7):
        """
        Vẽ Heatmap thể hiện tương quan giữa Giá Vàng (Target) và 
        QUÁ KHỨ của các tài sản khác (Lag Features).
        """
        # 1. Xác định các tài sản cần so sánh (Tự động lấy cột tốt nhất)
        assets_to_check = {
            "USD Index": "USD Index",
            "Oil (USO)": "Oil (USO)",
            "S&P 500": "S&P 500",
            "Bạc (Silver)": "Silver",
            "Gold Miners (GDX)": "Gold Miners",
            "Lợi suất trái phiếu (10Y)": "US Bond 10Y"
        }
        
        # Dictionary lưu kết quả correlation
        corr_data = {}
        
        # 2. Tính toán Correlation cho từng Lag
        # Với mỗi tài sản, ta tính corr(Gold_t, Asset_t-lag)
        for label, group_key in assets_to_check.items():
            # Tìm tên cột thực tế trong DataFrame
            if group_key not in self.prefix_groups: continue
            col_name = self._get_best_col(self.prefix_groups[group_key])
            
            if not self._check_col(col_name): continue
            
            asset_corrs = []
            for lag in range(1, max_lag + 1):
                # Tạo chuỗi lag của tài sản đó
                shifted_series = self.df[col_name].shift(lag)
                
                # Tính tương quan với Target (Gold) hiện tại
                # Dùng dropna để loại bỏ các giá trị NaN sinh ra do shift
                temp_df = pd.DataFrame({
                    'Target': self.df[self.target], 
                    'Lagged_Asset': shifted_series
                }).dropna()
                
                if not temp_df.empty:
                    r = temp_df['Target'].corr(temp_df['Lagged_Asset'])
                else:
                    r = 0
                asset_corrs.append(r)
            
            corr_data[label] = asset_corrs

        # 3. Chuyển đổi sang DataFrame để vẽ
        if not corr_data:
            self.logger.warning("Không đủ dữ liệu để vẽ Cross-Lag Heatmap.")
            return

        lag_df = pd.DataFrame(corr_data, index=[f"Lag {i} ngày" for i in range(1, max_lag + 1)])
        
        # Transpose để: Trục Y = Tên Tài Sản, Trục X = Độ trễ
        lag_df = lag_df.T 

        # 4. Vẽ biểu đồ
        plt.figure(figsize=(12, 8))
        sns.heatmap(lag_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", 
                    linewidths=0.5, linecolor='white', cbar_kws={'label': 'Hệ số tương quan'})
        
        plt.title(f"Tác động trễ của các tài sản lên Giá Vàng (Max Lag: {max_lag})", fontsize=14)
        plt.xlabel("Độ trễ (Thời gian phản ứng)")
        plt.ylabel("Loại tài sản")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # =========================================================
    # 5. BIỂU ĐỒ NẾN (CANDLESTICK)
    # =========================================================
    def plot_candlestick(self, days=60):
        """Vẽ biểu đồ nến Nhật bản sử dụng matplotlib thuần (Không cần thư viện phụ)."""
        # Lấy dữ liệu gần nhất
        df_sub = self.df.tail(days).copy()
        
        # Mapping cột OHLC
        high = df_sub[self._get_best_col(["High"])]
        low = df_sub[self._get_best_col(["Low"])]
        close = df_sub[self.target]
        
        # Nếu không tìm thấy cột Open, giả lập bằng Close hôm trước
        open_col = self._get_best_col(["Open"])
        if open_col:
            opn = df_sub[open_col]
        else:
            opn = close.shift(1).fillna(close)

        # Xác định màu nến
        colors = [self.colors["up"] if c >= o else self.colors["down"] for c, o in zip(close, opn)]

        plt.figure(figsize=(16, 8))
        
        # Vẽ bóng nến (Shadow)
        plt.vlines(x=df_sub.index, ymin=low, ymax=high, color="#555555", linewidth=1)
        
        # Vẽ thân nến (Body) - Dùng linewidth lớn để giả lập thân nến
        plt.vlines(x=df_sub.index, ymin=opn, ymax=close, color=colors, linewidth=6)
        
        plt.title(f"Biểu đồ giá {days} phiên gần nhất", fontsize=14)
        plt.ylabel("Giá (USD)")
        plt.grid(True, alpha=0.3)
        plt.show()

    # =========================================================
    # FULL REPORT PIPELINE
    # =========================================================
    def generate_report(self):
        """Chạy toàn bộ các biểu đồ phân tích."""
        self.logger.info("Bắt đầu tạo báo cáo trực quan hóa...")
        
        print("\n--- 1. XU HƯỚNG ---")
        self.gold_trend()
        self.gold_trend_segments()
        
        print("\n--- 2. RỦI RO & BIẾN ĐỘNG ---")
        self.volatility_analysis()
        
        print("\n--- 3. MÙA VỤ ---")
        self.seasonal_heatmap()
        
        print("\n--- 4. TƯƠNG QUAN ĐA BIẾN ---")
        self.gold_vs_oil()
        self.gold_vs_usd()
        self.gold_vs_equity()
        self.gold_vs_metals()
        
        self.correlation_matrix()
        self.cross_asset_lag_heatmap()
        self.rolling_correlation()
        
        print("\n--- 5. PRICE ACTION ---")
        self.plot_candlestick()
        

        self.logger.info("Hoàn tất báo cáo.")
