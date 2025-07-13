from datetime import datetime
import os

# 基础配置
START_YEAR = 2023
CURRENT_YEAR = datetime.now().year  # 自动获取当前年份
API_URL = "https://history.macaumarksix.com/history/macaujc2/y/"
BACKTEST_WINDOW = 200  # 回测期数

# 错误日志配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ERROR_LOG_PATH = os.path.join(DATA_DIR, 'error_log.csv')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 调试信息
if __name__ == "__main__":
    print("===== 配置文件信息 =====")
    print(f"起始年份: {START_YEAR}")
    print(f"当前年份: {CURRENT_YEAR}")
    print(f"API地址: {API_URL}")
    print(f"回测期数: {BACKTEST_WINDOW}")
    print(f"错误日志路径: {ERROR_LOG_PATH}")
    print("=======================")
