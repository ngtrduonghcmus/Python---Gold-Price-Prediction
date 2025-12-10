import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime


class Logger:
    """
    Logger chuẩn cho toàn bộ pipeline.

    - Tự động tạo thư mục logs/
    - Log vào file + console
    - Tách file theo ngày
    - Có file error.log riêng
    """

    def __init__(self, name: str = "project", log_dir: str = "logs"):
        self.name = name
        self.log_dir = log_dir

        # Tạo thư mục logs nếu chưa có
        os.makedirs(self.log_dir, exist_ok=True)

        # Tên file dạng: training_2025-12-10.log
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"{self.name}_{date_str}.log")

        # Logger chính
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Tránh thêm handler trùng lặp khi import nhiều lần
        if not self.logger.handlers:
            self._add_handlers(log_file)

    def _add_handlers(self, log_file):
        # Format log chuẩn, đọc dễ
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        # --- Console Handler ---
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # --- File Handler (rotating) ---
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,     # Lưu 7 ngày gần nhất
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # --- Error File Handler ---
        error_handler = logging.FileHandler(
            os.path.join(self.log_dir, "error.log")
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)

        # Add vào logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def get_logger(self):
        return self.logger
