# utils/sync_logger.py
import logging

sync_monitor_logger = logging.getLogger("SyncMonitor")
sync_monitor_logger.setLevel(logging.DEBUG)
if not sync_monitor_logger.handlers:
    _sync_handler = logging.FileHandler("sync_monitor.log", encoding="utf-8")
    _sync_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S.%f"))
    sync_monitor_logger.addHandler(_sync_handler)
sync_monitor_logger.propagate = False