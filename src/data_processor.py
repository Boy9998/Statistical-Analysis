import pandas as pd
from lunarcalendar import Converter, Solar
from datetime import datetime

def preprocess_data(df):
    """数据预处理函数 - 确保日期格式正确"""
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    return df

def add_temporal_features(df):
    """添加基础时序特征：星期几、月份、季度"""
    df = preprocess_data(df)
    df['weekday'] = df['date'].dt.dayofweek + 1
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    print(f"已添加时序特征: 星期几, 月份, 季度")
    return df

def add_lunar_features(df):
    """添加农历特征：农历月份、日期、是否闰月"""
    df = preprocess_data(df)
    
    def get_lunar_date(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            return Converter.Solar2Lunar(solar)
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
    df['lunar'] = df['date'].apply(get_lunar_date)
    df['lunar_month'] = df['lunar'].apply(lambda x: x.month if x else 0)
    df['lunar_day'] = df['lunar'].apply(lambda x: x.day if x else 0)
    df['is_leap_month'] = df['lunar'].apply(lambda x: 1 if x and x.isleap else 0)
    print(f"已添加农历特征: 月份, 日期, 闰月")
    return df

def add_festival_features(df):
    """添加节日特征"""
    df = preprocess_data(df)
    
    def detect_festival(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
        except Exception as e:
            print(f"节日检测错误: {e}, 日期: {dt}")
            return "无"
        
        if not lunar:
            return "无"
        
        lunar_festivals = {
            (1, 1): "春节",
            (1, 15): "元宵",
            (5, 5): "端午",
            (7, 7): "七夕",
            (7, 15): "中元",
            (8, 15): "中秋",
            (9, 9): "重阳"
        }
        
        solar_festivals = {
            (4, 4): "清明",
            (4, 5): "清明",
            (12, 22): "冬至"
        }
        
        # 春节识别：仅正月初一
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        if (lunar.month, lunar.day) in lunar_festivals:
            return lunar_festivals[(lunar.month, lunar.day)]
        
        if (dt.month, dt.day) in solar_festivals:
            return solar_festivals[(dt.month, dt.day)]
        
        return "无"
    
    df['festival'] = df['date'].apply(detect_festival)
    print(f"已添加节日特征")
    return df

def add_season_features(df):
    """添加季节特征"""
    df = preprocess_data(df)
    
    def get_season(dt):
        month = dt.month
        if 3 <= month <= 5: return "春"
        if 6 <= month <= 8: return "夏"
        if 9 <= month <= 11: return "秋"
        return "冬"
    
    df['season'] = df['date'].apply(get_season)
    print(f"已添加季节特征")
    return df

def add_all_features(df):
    """一次性添加所有特征"""
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_festival_features(df)
    df = add_season_features(df)
    return df

# 测试函数
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    test_dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    test_df = pd.DataFrame({'date': test_dates})
    test_df = add_all_features(test_df)
    print(test_df)
    print("\n===== 测试完成 =====")
