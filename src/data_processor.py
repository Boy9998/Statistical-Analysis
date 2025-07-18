import pandas as pd
import numpy as np
from lunarcalendar import Converter, Solar
from datetime import datetime
import hashlib
import json
import os
from config import ML_MODEL_PATH

# 常量定义
FIXED_ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
FIXED_FESTIVALS = ["春节", "元宵", "端午", "七夕", "中元", "中秋", "重阳", "清明", "冬至", "无"]

def preprocess_data(df):
    """数据预处理（确保基础列存在）"""
    # 强制日期转换
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 初始化必要列
    df['year'] = df['date'].dt.year
    if 'zodiac' not in df.columns:
        df['zodiac'] = np.random.choice(FIXED_ZODIACS, len(df))
    if 'festival' not in df.columns:
        df['festival'] = "无"
    
    # 数据清洗
    df['zodiac'] = df['zodiac'].apply(
        lambda x: x if x in FIXED_ZODIACS else np.random.choice(FIXED_ZODIACS)
    )
    return df

def add_temporal_features(df):
    """添加时序特征"""
    df = preprocess_data(df)
    df['weekday'] = df['date'].dt.dayofweek + 1  # 1=周一,7=周日
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    return df

def add_lunar_features(df):
    """添加农历特征（带异常处理）"""
    df = preprocess_data(df)
    
    def safe_conversion(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception:
            return None
    
    df['lunar'] = df['date'].apply(safe_conversion)
    df['lunar_month'] = df['lunar'].apply(lambda x: x.month if x else 0)
    df['lunar_day'] = df['lunar'].apply(lambda x: x.day if x else 0)
    df['is_leap_month'] = df['lunar'].apply(lambda x: 1 if x and x.isleap else 0)
    return df

def add_festival_features(df):
    """添加节日特征（固定维度）"""
    df = preprocess_data(df)
    
    festival_map = {
        (1, 1): "春节", (1, 15): "元宵", (5, 5): "端午",
        (7, 7): "七夕", (7, 15): "中元", (8, 15): "中秋",
        (9, 9): "重阳", (4, 4): "清明", (4, 5): "清明",
        (12, 22): "冬至"
    }
    
    def detect_festival(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            if lunar.month == 1 and lunar.day == 1:
                return "春节"
            return festival_map.get(
                (lunar.month, lunar.day),
                festival_map.get((dt.month, dt.day), "无")
            )
        except Exception:
            return "无"
    
    df['festival'] = df['date'].apply(detect_festival)
    df['is_festival'] = (df['festival'] != "无").astype(int)
    return df

def add_season_features(df):
    """添加季节特征"""
    season_map = {
        1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
        6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋",
        11: "秋", 12: "冬"
    }
    df = preprocess_data(df)
    df['season'] = df['date'].dt.month.map(season_map)
    return df

def add_rolling_features(df):
    """添加滚动窗口特征（严格维度控制）"""
    if df.empty:
        return df
    
    # 确保生肖列存在
    if 'zodiac' not in df.columns:
        df['zodiac'] = np.random.choice(FIXED_ZODIACS, len(df))
    
    # 创建生肖虚拟变量
    zodiac_dummies = pd.get_dummies(df['zodiac']).reindex(columns=FIXED_ZODIACS, fill_value=0)
    
    # 计算滚动特征
    for window in [7, 30, 100]:
        # 滚动频率
        rolling = zodiac_dummies.rolling(window, min_periods=1).mean()
        rolling.columns = [f'rolling_{window}d_{z}' for z in FIXED_ZODIACS]
        df = pd.concat([df, rolling], axis=1)
        
        # 热度指数
        with np.errstate(divide='ignore', invalid='ignore'):
            heat = rolling / zodiac_dummies.expanding().mean()
        heat = heat.replace([np.inf, -np.inf], 0)
        heat.columns = [f'heat_{window}d_{z}' for z in FIXED_ZODIACS]
        df = pd.concat([df, heat], axis=1)
    
    return df

def generate_feature_signature(df):
    """生成特征签名"""
    signature = {
        "version": "2.1",
        "timestamp": datetime.now().isoformat(),
        "columns": sorted(df.columns.tolist()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape
    }
    
    os.makedirs(ML_MODEL_PATH, exist_ok=True)
    with open(os.path.join(ML_MODEL_PATH, 'feature_signature.json'), 'w') as f:
        json.dump(signature, f, indent=2)

def add_all_features(df):
    """主处理函数（带严格顺序）"""
    # 处理步骤
    steps = [
        add_temporal_features,
        add_lunar_features,
        add_festival_features,
        add_season_features,
        add_rolling_features
    ]
    
    # 执行处理
    for step in steps:
        df = step(df)
    
    # 最终校验
    required_cols = {
        'date', 'year', 'weekday', 'month', 'quarter',
        'lunar_month', 'lunar_day', 'is_leap_month',
        'festival', 'is_festival', 'season'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺失必要列: {missing}")
    
    generate_feature_signature(df)
    return df

if __name__ == "__main__":
    # 测试数据
    test_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'zodiac': np.random.choice(FIXED_ZODIACS, 100)
    })
    
    # 测试处理流程
    try:
        processed = add_all_features(test_data)
        print("处理成功！特征列示例：")
        print(processed.iloc[0][['date', 'zodiac', 'festival', 'rolling_7d_鼠']])
    except Exception as e:
        print(f"处理失败: {str(e)}")
