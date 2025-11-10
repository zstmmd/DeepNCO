# python
# File: `utils/logging_config.py`
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

def configure_logging(level: int = logging.INFO,
                      fmt: Optional[str] = None,
                      datefmt: Optional[str] = None,
                      log_dir: str = "log",
                      filename: str = "app.log",
                      max_bytes: int = 10 * 1024 * 1024,
                      backup_count: int = 5) -> None:
    """
    全局日志配置：
    - 日志文件位于 `{log_dir}/{filename}`（会创建目录）
    - 同时输出到 stdout
    - 使用 RotatingFileHandler 控制文件大小
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    if datefmt is None:
        datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()

    # 避免重复添加 handlers（若已配置则直接返回）
    if root.handlers:
        return

    root.setLevel(level)
    formatter = logging.Formatter(fmt, datefmt)

    # Stream handler -> stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # Ensure log directory exists
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        # 若无法创建目录，仍然继续，不抛出异常
        pass

    # File handler -> rotating file in log_dir
    log_path = os.path.join(log_dir, filename)
    try:
        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:
        # 如果文件句柄创建失败（权限等问题），只保留 stdout
        root.warning("Failed to create file handler for logging at %s; continuing with stdout only.", log_path)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取命名 logger（建议模块使用此函数或直接 logging.getLogger）。"""
    return logging.getLogger(name)

# 默认在模块导入时进行配置：输出到 `log/app.log`
configure_logging()