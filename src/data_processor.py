import pandas as pd
from lunarcalendar import Converter, Solar
from datetime import datetime

def preprocess_data(df):
    """
    数据预处理函数 - 确保日期格式正确
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        预处理后的DataFrame
    """
    # 确保日期格式正确
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 添加年份列（如果不存在）
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    return df

def add_temporal_features(df):
    """
    添加基础时序特征：星期几、月份、季度
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        添加特征后的DataFrame
    """
    df = preprocess_data(df)
    
    # 添加星期几特征 (1-7 表示周一到周日)
    df['weekday'] = df['date'].dt.dayofweek + 1
    
    # 添加月份特征
    df['month'] = df['date'].dt.month
    
    # 添加季度特征
    df['quarter'] = df['date'].dt.quarter
    
    print(f"已添加时序特征: 星期几, 月份, 季度")
    return df

def add_lunar_features(df):
    """
    添加农历特征：农历月份、日期、是否闰月
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        添加特征后的DataFrame
    """
    df = preprocess_data(df)
    
    def get_lunar_date(dt):
        """精确转换公历到农历"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
    # 计算农历日期
    df['lunar'] = df['date'].apply(get_lunar_date)
    
    # 提取农历月份和日期
    df['lunar_month'] = df['lunar'].apply(
        lambda x: x.month if x is not None else 0
    )
    df['lunar_day'] = df['lunar'].apply(
        lambda x: x.day if x is not None else 0
    )
    
    # 添加是否闰月特征
    df['is_leap_month'] = df['lunar'].apply(
        lambda x: 1 if x and x.isleap else 0
    )
    
    print(f"已添加农历特征: 月份, 日期, 闰月")
    return df

def add_festival_features(df):
    """
    添加节日特征
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        添加特征后的DataFrame
    """
    df = preprocess_data(df)
    
    # 创建节日检测函数
    def detect_festival(dt):
        lunar = None
        try:
            # 获取农历日期
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
        except Exception as e:
            print(f"节日检测农历转换错误: {e}, 日期: {dt}")
            return "无"
        
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
        
        # 春节识别：仅正月初一
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        # 精确匹配
        if lunar_date in lunar_festivals:
            return lunar_festivals[lunar_date]
        if solar_date in solar_festivals:
            return solar_festivals[solar_date]
        
        return "无"
    
    # 应用节日检测
    df['festival'] = df['date'].apply(detect_festival)
    
    print(f"已添加节日特征")
    return df

def add_season_features(df):
    """
    添加季节特征
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        添加特征后的DataFrame
    """
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
    print(f"已添加季节特征")
    return df

def add_all_features(df):
    """
    一次性添加所有特征
    参数:
        df: pandas DataFrame 包含彩票数据
    返回:
        添加所有特征后的DataFrame
    """
    print("开始添加所有特征...")
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_festival_features(df)
    df = add_season_features(df)
    print("所有特征添加完成")
    return df

# 测试函数
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    test_df = pd.DataFrame({'date': test_dates})
    print(f"创建测试数据: {len(test_df)}条记录")
    
    # 测试添加所有特征
    test_df = add_all_features(test_df)
    
    # 展示结果
    print("\n测试结果示例:")
    print(test_df.head())
    
    # 验证特征列是否存在
    expected_columns = ['weekday', 'month', 'quarter', 'lunar_month', 'lunar_day', 
                       'is_leap_month', 'festival', 'season']
    missing = [col for col in expected_columns if col not in test_df.columns]
    
    if missing:
        print(f"\n错误: 缺失特征列: {', '.join(missing)}")
    else:
        print("\n所有特征列添加成功")
    
    # 特殊日期验证
    spring_festival = test_df[test_df['festival'] == '春节']
    if not spring_festival.empty:
        print(f"\n春节日期检测: {spring_festival['date'].dt.strftime('%Y-%m-%d').values[0]}")
    else:
        print("\n错误: 未检测到春节日期")
    
    # 季节分布
    season_counts = test_df['season'].value_counts()
    print(f"\n季节分布:\n{season_counts}")
    
    print("\n===== 测试完成 =====")
