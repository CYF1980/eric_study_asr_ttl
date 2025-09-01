# init_env.py

from dotenv import load_dotenv
import os

def init_env(dotenv_path=".env"):
    """
    初始化並載入指定的 .env 檔案
    :param dotenv_path: .env 檔案路徑 (預設為當前目錄下的 .env)
    """
    load_dotenv(dotenv_path)
    print(f"[INFO] Loaded environment variables from {dotenv_path}")

def get_env_variable(key, default=None):
    """
    取得指定的環境變數
    :param key: 環境變數名稱
    :param default: 如果未找到變數則返回此預設值
    :return: 環境變數值或預設值
    """
    return os.getenv(key, default)
