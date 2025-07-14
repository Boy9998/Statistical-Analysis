import pandas as pd
import numpy as np
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
from lunarcalendar import Converter, Solar, Lunar
import holidays
import re

def preprocess_data(df):
    """数据清洗和预处理"""
    # 确保日期是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 添加年份列
    df['year'] = df['date'].dt.year
    
    # 处理缺失值
    df = df.dropna(subset=['special'])
    
    print(f"数据预处理完成: 处理了 {len(df)} 条记录")
    return df

def add_temporal_features(df):
    """添加星期几、月份等基础时序特征"""
    # 确保日期是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 添加星期几特征 (1-7 表示周一到周日)
    df['weekday'] = df['date'].dt.dayofweek + 1
    
    # 添加月份特征
    df['month'] = df['date'].dt.month
    
    # 添加季度特征
    df['quarter'] = df['date'].dt.quarter
    
    print("已添加基础时序特征: 星期几, 月份, 季度")
    return df

def add_lunar_features(df):
    """添加农历特征"""
    def get_lunar_date(dt):
        """精确转换公历到农历"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
    def detect_festival(dt):
        """识别传统节日"""
        lunar = get_lunar_date(dt)
        if not lunar:
            return "无"
        
        lunar_date = (lunar.month, lunar.day)
        solar_date = (dt.month, dt.day)
        
        # 农历节日映射
        lunar_festivals = {
            (1, 1): "春节",
            (1, 15): "元宵",
            (5, 5): "端午",
            (7, 7): "七夕",
            (7, 15): "中元",
            (8, 15): "中秋",
            (9, 9): "重阳"
        }
        
        # 公历节日映射
        solar_festivals = {
            (4, 4): "清明",
            (4, 5): "清明",
            (12, 22): "冬至"
        }
        
        # 春节范围：农历正月初一至十五
        if lunar.month == 1 and 1 <= lunar.day <= 15:
            return "春节"
        
        # 精确匹配
        if lunar_date in lunar_festivals:
            return lunar_festivals[lunar_date]
        if solar_date in solar_festivals:
            return solar_festivals[solar_date]
        
        return "无"
    
    # 添加农历日期
    df['lunar'] = df['date'].apply(get_lunar_date)
    
    # 添加农历月份
    df['lunar_month'] = df['lunar'].apply(lambda x: x.month if x else None)
    
    # 添加农历日
    df['lunar_day'] = df['lunar'].apply(lambda x: x.day if x else None)
    
    # 添加节日特征
    df['festival'] = df['date'].apply(detect_festival)
    
    print("已添加农历特征: 农历月份, 农历日, 节日")
    return df

def add_rolling_features(df):
    """添加滚动窗口特征"""
    # 确保数据按日期排序
    df = df.sort_values('date')
    
    # 生肖列表
    zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
    
    # 创建滚动窗口特征
    for zodiac in zodiacs:
        # 7天滚动频率
        df[f'{zodiac}_7d_freq'] = df['zodiac'].rolling(window=7, min_periods=1).apply(lambda x: (x == zodiac).mean())
        
        # 30天滚动频率
        df[f'{zodiac}_30d_freq'] = df['zodiac'].rolling(window=30, min_periods=1).apply(lambda x: (x == zodiac).mean())
        
        # 100天滚动频率
        df[f'{zodiac}_100d_freq'] = df['zodiac'].rolling(window=100, min_periods=1).apply(lambda x: (x == zodiac).mean())
    
    # 添加生肖热度指数 (最近30天频率)
    df['zodiac_heat'] = df.apply(lambda row: max([row[f'{z}_30d_freq'] for z in zodiacs]), axis=1)
    
    print("已添加滚动窗口特征: 7天/30天/100天频率, 生肖热度指数")
    return df

def create_features(df):
    """执行完整的特征工程流程"""
    if df.empty:
        print("警告: 数据为空，无法创建特征")
        return df
    
    print("开始特征工程...")
    df = preprocess_data(df)
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_rolling_features(df)
    print("特征工程完成")
    return df

if __name__ == "__main__":
    # 测试代码
    test_data = pd.DataFrame({
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'special': [12, 25],
        'zodiac': ['鼠', '牛']
    })
    print("测试数据:")
    print(test_data)
    
    processed = create_features(test_data)
    print("\n处理后的数据:")
    print(processed.head())
