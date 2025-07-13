import pandas as pd
from lunarcalendar import Converter, Solar
import holidays
import re
from config import START_YEAR, CURRENT_YEAR
from datetime import datetime

def preprocess_data(df):
    """数据预处理函数"""
    # 确保日期格式正确
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 添加年份列（如果不存在）
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    return df

def add_temporal_features(df):
    """添加星期几、月份等基础时序特征"""
    df = preprocess_data(df)
    
    # 添加星期几特征 (1-7 表示周一到周日)
    df['weekday'] = df['date'].dt.dayofweek + 1
    
    # 添加月份特征
    df['month'] = df['date'].dt.month
    
    # 添加季度特征
    df['quarter'] = df['date'].dt.quarter
    
    return df

def add_lunar_features(df):
    """添加农历特征"""
    df = preprocess_data(df)
    
    # 计算农历日期
    df['lunar_date'] = df['date'].apply(
        lambda d: Converter.Solar2Lunar(Solar(d.year, d.month, d.day))
    
    # 提取农历月份和日期
    df['lunar_month'] = df['lunar_date'].apply(lambda x: x.month)
    df['lunar_day'] = df['lunar_date'].apply(lambda x: x.day)
    
    # 添加是否闰月特征
    df['is_leap_month'] = df['lunar_date'].apply(lambda x: 1 if x.isleap else 0)
    
    return df

def add_festival_features(df):
    """添加节日特征"""
    df = preprocess_data(df)
    
    # 创建节日检测函数
    def detect_festival(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            
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
            if (lunar.month, lunar.day) in lunar_festivals:
                return lunar_festivals[(lunar.month, lunar.day)]
            
            if (dt.month, dt.day) in solar_festivals:
                return solar_festivals[(dt.month, dt.day)]
            
            return "无"
        except Exception as e:
            print(f"节日检测错误: {e}, 日期: {dt}")
            return "无"
    
    # 应用节日检测
    df['festival'] = df['date'].apply(detect_festival)
    
    return df

def add_season_features(df):
    """添加季节特征"""
    df = preprocess_data(df)
    
    def get_season(dt):
        month = dt.month
        if 3 <= month <= 5:
            return "春"
        elif 6 <= month <= 8:
            return "夏"
        elif 9 <= month <= 11:
            return "秋"
        else:
            return "冬"
    
    df['season'] = df['date'].apply(get_season)
    return df

# 测试函数
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    test_df = pd.DataFrame({'date': test_dates})
    
    print("\n添加时序特征...")
    test_df = add_temporal_features(test_df)
    print(test_df[['date', 'weekday', 'month', 'quarter']].head())
    
    print("\n添加农历特征...")
    test_df = add_lunar_features(test_df)
    print(test_df[['date', 'lunar_month', 'lunar_day', 'is_leap_month']].head())
    
    print("\n添加节日特征...")
    test_df = add_festival_features(test_df)
    print(test_df[test_df['festival'] != '无'].head(10))
    
    print("\n添加季节特征...")
    test_df = add_season_features(test_df)
    print(test_df[['date', 'season']].head())
    
    print("\n===== 测试完成 =====")
