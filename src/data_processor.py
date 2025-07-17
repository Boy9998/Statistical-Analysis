import pandas as pd
import numpy as np
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
    # 新增: 节日标志特征
    df['is_festival'] = df['festival'].apply(lambda x: 1 if x != "无" else 0)
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

def add_rolling_features(df):
    """
    添加滚动窗口统计特征
    包括7天、30天、100天滚动频率和热度指数
    """
    if df.empty:
        return df
    
    print("添加滚动窗口统计特征...")
    df = df.sort_values('date')
    
    # 创建生肖虚拟变量
    zodiac_dummies = pd.get_dummies(df['zodiac'])
    
    # 确保所有生肖列都存在
    all_zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
    for zodiac in all_zodiacs:
        if zodiac not in zodiac_dummies.columns:
            zodiac_dummies[zodiac] = 0
    
    # 计算滚动频率
    rolling_features = {}
    for window in [7, 30, 100]:
        rolling = zodiac_dummies.rolling(window=window, min_periods=1).mean()
        rolling.columns = [f'rolling_{window}d_{zodiac}' for zodiac in rolling.columns]
        rolling_features[window] = rolling
    
    # 计算长期平均频率
    long_term_avg = zodiac_dummies.expanding().mean()
    
    # 计算热度指数
    heat_index_features = {}
    for window in [7, 30, 100]:
        heat_index = rolling_features[window] / long_term_avg
        heat_index.columns = [f'heat_index_{window}d_{zodiac}' for zodiac in heat_index.columns]
        heat_index_features[window] = heat_index
    
    # 合并所有特征
    all_rolling = pd.concat([
        pd.concat(rolling_features.values(), axis=1),
        pd.concat(heat_index_features.values(), axis=1)
    ], axis=1)
    
    # 替换无穷大值为0
    all_rolling = all_rolling.replace([np.inf, -np.inf], 0)
    
    # 合并到原始数据
    df = pd.concat([df, all_rolling], axis=1)
    print(f"已添加滚动窗口统计特征: {len(all_rolling.columns)}个新特征")
    return df

def add_all_features(df):
    """一次性添加所有特征"""
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_festival_features(df)
    df = add_season_features(df)
    df = add_rolling_features(df)
    return df

# 测试函数
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'date': test_dates,
        'zodiac': ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡'] * 10
    })
    
    # 添加所有特征
    test_df = add_all_features(test_df)
    
    # 打印结果
    print("\n测试结果:")
    print(test_df[['date', 'zodiac', 'rolling_7d_鼠', 'heat_index_7d_鼠']].tail(10))
    
    # 验证特征存在
    expected_features = [
        'rolling_7d_鼠', 'rolling_30d_鼠', 'rolling_100d_鼠',
        'heat_index_7d_鼠', 'heat_index_30d_鼠', 'heat_index_100d_鼠'
    ]
    
    missing = [f for f in expected_features if f not in test_df.columns]
    if missing:
        print(f"\n错误: 缺失特征: {', '.join(missing)}")
    else:
        print("\n所有滚动特征添加成功")
    
    print("\n===== 测试完成 =====")
