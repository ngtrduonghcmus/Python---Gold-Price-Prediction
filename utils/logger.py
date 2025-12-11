import logging
import os
from logging.handlers import RotatingFileHandler


# ---------------------------------------------------
# Tạo thư mục logs nếu chưa có
# ---------------------------------------------------
LOG_DIR = "results/logs"
os.makedirs(LOG_DIR, exist_ok=True)


def create_file_logger(name, filename, level=logging.INFO):
    """
    Tạo logger ghi log vào logs/filename
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # đường dẫn file log đầy đủ
    filepath = os.path.join(LOG_DIR, filename)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(class)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler = RotatingFileHandler(
        filepath,
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8"     # cần thiết để ghi tiếng Việt
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger

_preprocess_base_logger = create_file_logger("preprocess", "preprocessing.log", level=logging.DEBUG)
_training_base_logger = create_file_logger("training", "training.log", level=logging.DEBUG)
preprocess_logger = _preprocess_base_logger
training_logger = _training_base_logger

def get_preprocess_logger(class_name: str):
    """Trả về logger có đính kèm tên class."""
    return logging.LoggerAdapter(_preprocess_base_logger, {"class": class_name})

def get_training_logger(class_name: str):
    """Trả về logger có đính kèm tên class."""
    return logging.LoggerAdapter(_training_base_logger, {"class": class_name})
