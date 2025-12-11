import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import get_preprocess_logger

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
        self.logger = get_preprocess_logger(self.__class__.__name__)
        self.logger.info(">>> KHỞI TẠO STATISTICS ANALYZER <<<")

    # ************************************************
    # 1. THỐNG KÊ MÔ TẢ
    # ************************************************
    def get_describe_stats(self):
        """
        In ra các thống kê mô tả của DataFrame cho cả cột số và cột phân loại.
        """
        self.logger.info("*** LẤY THỐNG KÊ MÔ TẢ ***")
        if self.df.empty:
            self.logger.warning("DataFrame trống, không thể lấy thống kê mô tả.")
            return

        self.logger.info("Thống kê mô tả cho cột số.")
        self.logger.debug(f"Số cột số: {len(self.df.select_dtypes(include=np.number).columns)}")
        self.logger.debug(f"Cột số: {self.df.select_dtypes(include=np.number).columns.tolist()}")
        self.logger.debug(f"Thống kê mô tả:\n{self.df.describe()}")

        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            self.logger.info("Thống kê mô tả cho cột phân loại.")
            self.logger.debug(f"Số cột phân loại: {len(cat_cols)}")
            self.logger.debug(f"Cột phân loại: {cat_cols.tolist()}")
            self.logger.debug(f"Thống kê mô tả:\n{self.df[cat_cols].describe()}")