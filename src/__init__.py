# src/__init__.py

# 导入关键函数和类
from .utils import fetch_historical_data, zodiac_mapping, send_dingtalk, send_email
from .analysis import LotteryAnalyzer

__all__ = [
    'fetch_historical_data',
    'zodiac_mapping',
    'send_dingtalk',
    'send_email',
    'LotteryAnalyzer'
]
