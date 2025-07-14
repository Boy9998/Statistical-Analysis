from datetime import datetime
import os

# ==================== 基础配置 ====================
START_YEAR = 2023
CURRENT_YEAR = datetime.now().year  # 自动获取当前年份
API_URL = "https://history.macaumarksix.com/history/macaujc2/y/"
BACKTEST_WINDOW = 200  # 回测期数

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ML_MODEL_PATH = MODEL_DIR  # 添加ML_MODEL_PATH定义

# ==================== 文件配置 ====================
ERROR_LOG_PATH = os.path.join(DATA_DIR, 'error_log.csv')
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, 'historical_data.csv')

# ==================== 特征工程配置 ====================
ROLLING_WINDOWS = [7, 30, 100]  # 滚动窗口大小（天）
ZODIAC_LIST = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]

# ==================== 通知配置 ====================
DINGTALK_RETRY_TIMES = 3  # 钉钉通知重试次数
EMAIL_TIMEOUT = 10  # 邮件发送超时时间（秒）

# ==================== 初始化检查 ====================
# 确保必要目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 调试信息 ====================
if __name__ == "__main__":
    print("===== 配置文件信息 =====")
    print("\n[基础配置]")
    print(f"起始年份: {START_YEAR}")
    print(f"当前年份: {CURRENT_YEAR}")
    print(f"API地址: {API_URL}")
    print(f"回测期数: {BACKTEST_WINDOW}")
    
    print("\n[路径配置]")
    print(f"数据目录: {DATA_DIR}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"ML模型路径: {ML_MODEL_PATH}")  # 添加ML_MODEL_PATH显示
    
    print("\n[文件配置]")
    print(f"错误日志路径: {ERROR_LOG_PATH}")
    print(f"历史数据路径: {HISTORICAL_DATA_PATH}")
    
    print("\n[特征工程]")
    print(f"滚动窗口: {ROLLING_WINDOWS}")
    print(f"生肖列表: {ZODIAC_LIST}")
    
    print("\n[通知配置]")
    print(f"钉钉重试次数: {DINGTALK_RETRY_TIMES}")
    print(f"邮件超时时间: {EMAIL_TIMEOUT}s")
    
    print("\n[目录状态]")
    print(f"数据目录存在: {os.path.exists(DATA_DIR)}")
    print(f"模型目录存在: {os.path.exists(MODEL_DIR)}")
    print("=======================")
