import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from scipy.stats import gaussian_kde

class StatisticsAnalyzer:
    """
    Lớp đưa ra các thống kê mô tả để phân tích và khám phá bộ dữ liệu.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo đối tượng với DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Bộ dữ liệu cần phân tích.
        """
        self.df = df.copy()

    # ************************************************
    # 1. THỐNG KÊ MÔ TẢ
    # ************************************************
    def get_describe_stats(self):
        """
        In ra các thống kê mô tả của DataFrame cho cả cột số và cột phân loại.
        """
        if self.df.empty:
            print(" DataFrame trống.")
            return

        print("\n========== DESCRIBE (NUMERIC) ==========")
        print(self.df.describe())

        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            print("\n========== DESCRIBE (CATEGORICAL) ==========")
            print(self.df[cat_cols].describe())

class Visualizer:
    """
    Lớp hỗ trợ trực quan hóa dữ liệu và kết quả phân tích bằng các biểu đồ.

    Parameters
    ----------
    df : pandas.DataFrame
        Bộ dữ liệu cần trực quan hóa.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo lớp Visualizer và chuẩn hóa cột datetime.

        Parameters
        ----------
        df : pandas.DataFrame
            Tập dữ liệu gốc.
        """
        self.df = df.copy()

        # Convert object -> datetime nếu có thể
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    pass

    # ************************************************
    # 1. Histogram + KDE
    # ************************************************
    def plot_hist(self, col):
        """
        Vẽ biểu đồ Histogram + KDE cho cột số.

        Nếu cột chỉ chứa giá trị 0/1 -> chuyển sang bar chart.

        Parameters
        ----------
        col : str
            Tên cột cần vẽ.

        Notes
        -----
        - KDE thực hiện bằng scipy nếu có.
        - Histogram được chuẩn hóa (density=True).
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return

        if not is_numeric_dtype(self.df[col]):
            print(f" Cột '{col}' không phải số -> bỏ qua Histogram.")
            return

        data = self.df[col].dropna()
        if data.empty:
            print(f" Cột '{col}' không có dữ liệu hợp lệ để vẽ.")
            return

        unique_vals = data.unique()

        if set(unique_vals) <= {0, 1} and len(unique_vals) <= 2:
            print(f" Cột '{col}' là nhị phân (0/1) -> vẽ Bar chart thay vì Histogram")
            self.plot_bar(col)
            return

        plt.figure(figsize=(9, 5))
        plt.hist(
            data,
            bins=30,
            density=True,
            color="#4C72B0",
            edgecolor="black",
            linewidth=1,
            alpha=0.7
        )

        try:
            kde = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 300)
            plt.plot(x_vals, kde(x_vals), color="red", linewidth=2.2)
        except:
            print(" Không load được scipy hoặc lỗi tính toán -> bỏ qua KDE")

        plt.title(f"Histogram + KDE – {col}", fontsize=16, fontweight="bold")
        plt.xlabel(col, fontsize=13)
        plt.ylabel("Density", fontsize=13)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ************************************************
    # 2. Boxplot
    # ************************************************
    def plot_box(self, col):
        """
        Vẽ boxplot theo chiều ngang cho một cột số.

        Parameters
        ----------
        col : str
            Tên cột cần vẽ.
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return

        if not is_numeric_dtype(self.df[col]):
            print(f" Cột '{col}' không phải số -> bỏ qua Boxplot.")
            return

        data = self.df[col].dropna()
        if data.empty:
            print(f" Cột '{col}' không có dữ liệu hợp lệ để vẽ.")
            return

        plt.figure(figsize=(12, 3.8))
        plt.boxplot(
            data,
            vert=False,
            patch_artist=True,
            widths=0.6,
            boxprops=dict(facecolor="#55A868", alpha=0.6, color="black"),
            medianprops=dict(color="red", linewidth=2.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.6)
        )

        plt.title(f"Boxplot – {col}", fontsize=16, fontweight="bold")
        plt.xlabel(col, fontsize=13)
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ************************************************
    # 3. Bar chart
    # ************************************************
    def plot_bar(self, col):
        """
        Vẽ biểu đồ Bar chart cho dữ liệu phân loại hoặc nhị phân.

        Parameters
        ----------
        col : str
            Tên cột cần vẽ.
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return

        counts = self.df[col].value_counts().sort_index()
        if counts.empty:
            print(f" Cột '{col}' không có dữ liệu hợp lệ để vẽ.")
            return

        plt.figure(figsize=(7, 4))
        counts.plot(
            kind="bar",
            color="#4C72B0",
            edgecolor="black"
        )
        plt.title(f"Bar Chart – {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=0 if len(counts) <= 10 else 45, ha='right')
        plt.tight_layout()
        plt.show()


    # ************************************************
    # 4. Scatter Plot
    # ************************************************
    def plot_scatter(self, x, y):
        """
        Vẽ scatter plot cho hai cột số.

        Parameters
        ----------
        x : str
            Tên cột trục x.
        y : str
            Tên cột trục y.
        """
        if x not in self.df.columns or y not in self.df.columns:
            print(f" Cột '{x}' hoặc '{y}' không tồn tại.")
            return

        if not (is_numeric_dtype(self.df[x]) and is_numeric_dtype(self.df[y])):
            print(" Scatter plot yêu cầu cả hai cột phải là số.")
            return

        plt.figure(figsize=(7, 4))
        plt.scatter(self.df[x], self.df[y], s=10)
        plt.title(f"Scatter Plot – {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ************************************************
    # 5. Heatmap tương quan
    # ************************************************
    def plot_correlation(self):
        """
        Vẽ heatmap ma trận tương quan giữa các cột số.

        Notes
        -----
        - Nếu số lượng cột quá lớn (>100), chỉ lấy 100 cột đầu tiên để tránh lag.
        """
        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.empty:
            print(" Không có cột số nào để tính tương quan.")
            return

        if numeric_df.shape[1] > 100:
            print(" Dataset có quá nhiều cột số -> chỉ vẽ 100 cột đầu để tránh lag.")
            numeric_df = numeric_df.iloc[:, :100]

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 10))
        plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
        plt.colorbar()
        plt.title("Correlation Matrix Heatmap")

        # Thiết lập nhãn trục
        col_names = corr.columns.tolist()
        plt.xticks(range(len(col_names)), col_names, rotation=90, fontsize=8)
        plt.yticks(range(len(col_names)), col_names, fontsize=8)

        # Thêm giá trị tương quan vào ô (chỉ cho ma trận nhỏ)
        if len(col_names) <= 15:
            for i in range(len(col_names)):
                for j in range(len(col_names)):
                    plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                             ha="center", va="center", color="black", fontsize=7)

        plt.tight_layout()
        plt.show()


    # ************************************************
    # 6. Auto plot
    # ************************************************
    def auto_plot(self, col):
        """
        Tự động chọn loại biểu đồ phù hợp cho cột.

        Các trường hợp:
        - datetime -> line chart.
        - numeric -> histogram + boxplot.
        - numeric nhị phân -> bar chart.
        - categorical -> bar chart.

        Parameters
        ----------
        col : str
            Cột cần vẽ.
        """
        if col not in self.df.columns:
            print(f" Cột '{col}' không tồn tại.")
            return

        print(f"\n Auto-plot cho cột: {col}")

        if is_datetime64_any_dtype(self.df[col]):
            print(" Dạng datetime -> vẽ Line Chart theo thời gian")

            # Đếm số lượng xảy ra cho mỗi ngày
            s = self.df[col].dropna().value_counts().sort_index()

            if s.empty:
                print(" Không có dữ liệu datetime hợp lệ để vẽ.")
                return

            plt.figure(figsize=(10, 4))
            s.plot()
            plt.title(f"Time Series Count - {col}")
            plt.xlabel("Date/Time")
            plt.ylabel("Count")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
            return

        if is_numeric_dtype(self.df[col]):
            data_clean = self.df[col].dropna()
            if data_clean.empty:
                print(" Không có dữ liệu số hợp lệ để vẽ.")
                return

            unique_vals = data_clean.unique()
            if set(unique_vals) <= {0, 1} and len(unique_vals) <= 2:
                print(" Numeric nhị phân 0/1 -> Bar chart")
                self.plot_bar(col)
            else:
                print(" Numeric -> Histogram + Boxplot")
                self.plot_hist(col)
                self.plot_box(col)
            return

        print(" Dạng phân loại -> Bar chart")
        self.plot_bar(col)

    # ************************************************
    # 7. Full Report
    # ************************************************
    def full_report(self):
        """
        Tạo báo cáo trực quan hóa tự động cho toàn bộ dataset.

        Notes
        -----
        - Mỗi cột được vẽ bằng auto_plot().
        - Giới hạn tối đa 200 biểu đồ (đã cập nhật từ 100).
        - Cuối cùng thêm heatmap tương quan.
        """
        print("\n TẠO BÁO CÁO HÌNH ẢNH TỰ ĐỘNG")
        print("-" * 50)

        count = 0
        MAX_PLOTS = 200

        for col in self.df.columns:
            if count >= MAX_PLOTS:
                print(f" Dừng tại {MAX_PLOTS} biểu đồ để tránh quá tải.")
                break

            print(f"\n Cột: {col}")
            self.auto_plot(col)
            count += 1

        print("\n Thêm Heatmap tương quan…")
        self.plot_correlation()
