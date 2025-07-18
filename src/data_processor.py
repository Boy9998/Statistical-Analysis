import pandas as pd
import numpy as np
from lunarcalendar import Converter, Solar
from datetime import datetime
import hashlib
import json
import os
from config import ML_MODEL_PATH

# 保持原有常量定义
FIXED_ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
FIXED_FESTIVALS = ["春节", "元宵", "端午", "七夕", "中元", "中秋", "重阳", "清明", "冬至", "无"]

def preprocess_data(df):
    """完整保留原有预处理逻辑"""
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()]  # 移除无效日期
    
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    # 保持原有生肖处理逻辑
    if 'zodiac' not in df.columns:
        df['zodiac'] = "未知"
    df['zodiac'] = df['zodiac'].apply(
        lambda x: x if x in FIXED_ZODIACS else "未知"
    )
    return df

def add_temporal_features(df):
    """完整保留原有时序特征"""
    df = preprocess_data(df)
    df['weekday'] = df['date'].dt.dayofweek + 1  # 保持原有周一=1的设定
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    print(f"[原有时序特征] 已添加: 星期{df['weekday'].unique()}")
    return df

def add_lunar_features(df):
    """完整保留农历特征计算逻辑"""
    df = preprocess_data(df)
    
    def get_lunar_date(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            return Converter.Solar2Lunar(solar)
        except Exception as e:
            print(f"[原有警告] 农历转换失败: {e}")
            return None
    
    df['lunar'] = df['date'].apply(get_lunar_date)
    df['lunar_month'] = df['lunar'].apply(lambda x: x.month if x else 0)
    df['lunar_day'] = df['lunar'].apply(lambda x: x.day if x else 0)
    df['is_leap_month'] = df['lunar'].apply(lambda x: 1 if x and x.isleap else 0)
    print("[原有农历特征] 已添加")
    return df

def add_festival_features(df):
    """完整保留节日检测逻辑"""
    df = preprocess_data(df)
    
    def detect_festival(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
        except Exception as e:
            print(f"[原有警告] 节日检测失败: {e}")
            return "无"
        
        if not lunar:
            return "无"
        
        # 保持原有节日映射关系
        lunar_festivals = {
            (1, 1): "春节", (1, 15): "元宵", (5, 5): "端午",
            (7, 7): "七夕", (7, 15): "中元", (8, 15): "中秋",
            (9, 9): "重阳"
        }
        
        solar_festivals = {
            (4, 4): "清明", (4, 5): "清明", (12, 22): "冬至"
        }
        
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        if (lunar.month, lunar.day) in lunar_festivals:
            return lunar_festivals[(lunar.month, lunar.day)]
        
        if (dt.month, dt.day) in solar_festivals:
            return solar_festivals[(dt.month, dt.day)]
        
        return "无"
    
    df['festival'] = df['date'].apply(detect_festival)
    df['is_festival'] = (df['festival'] != "无").astype(int)
    print(f"[原有节日特征] 已检测到{df['is_festival'].sum()}个节日")
    return df

def add_season_features(df):
    """完整保留季节特征逻辑"""
    df = preprocess_data(df)
    
    def get_season(dt):
        month = dt.month
        if 3 <= month <= 5: return "春"
        if 6 <= month <= 8: return "夏"
        if 9 <= month <= 11: return "秋"
        return "冬"
    
    df['season'] = df['date'].apply(get_season)
    print("[原有季节特征] 已添加")
    return df

def create_fixed_rolling_features(df, window_size):
    """完整保留原有滚动特征生成逻辑"""
    template = pd.DataFrame(columns=[
        f'rolling_{window_size}d_{z}' for z in FIXED_ZODIACS
    ] + [
        f'heat_index_{window_size}d_{z}' for z in FIXED_ZODIACS
    ])
    
    if df.empty or 'zodiac' not in df.columns:
        return template
    
    # 保持原有生肖虚拟变量生成方式
    zodiac_dummies = pd.get_dummies(df['zodiac'])
    for z in FIXED_ZODIACS:
        if z not in zodiac_dummies.columns:
            zodiac_dummies[z] = 0
    zodiac_dummies = zodiac_dummies[FIXED_ZODIACS]
    
    # 保持原有滚动计算逻辑
    rolling = zodiac_dummies.rolling(window=window_size, min_periods=1).mean()
    rolling.columns = [f'rolling_{window_size}d_{z}' for z in FIXED_ZODIACS]
    
    long_term_avg = zodiac_dummies.expanding().mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        heat_index = rolling / long_term_avg
    heat_index = heat_index.replace([np.inf, -np.inf], 0)
    heat_index.columns = [f'heat_index_{window_size}d_{z}' for z in FIXED_ZODIACS]
    
    result = pd.concat([rolling, heat_index], axis=1)
    for col in template.columns:
        if col not in result.columns:
            result[col] = 0.0
    
    return result[template.columns]

def add_rolling_features(df):
    """完整保留滚动特征添加逻辑"""
    if df.empty:
        return df
    
    print("[原有滚动特征] 开始计算...")
    df = df.sort_values('date')
    
    rolling_7d = create_fixed_rolling_features(df, 7)
    rolling_30d = create_fixed_rolling_features(df, 30)
    rolling_100d = create_fixed_rolling_features(df, 100)
    
    all_rolling = pd.concat([rolling_7d, rolling_30d, rolling_100d], axis=1)
    df = pd.concat([df, all_rolling], axis=1)
    
    generate_feature_signature(df)
    print(f"[原有滚动特征] 已添加 {len(all_rolling.columns)} 个特征")
    return df

def generate_feature_signature(df):
    """完整保留特征签名生成逻辑"""
    feature_signature = {
        'columns': sorted(df.columns.tolist()),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'shape': df.shape,
        'hash': hashlib.sha256(','.join(sorted(df.columns)).encode('utf-8')).hexdigest(),
        'created_at': datetime.now().isoformat()
    }
    
    os.makedirs(ML_MODEL_PATH, exist_ok=True)
    with open(os.path.join(ML_MODEL_PATH, 'feature_signature.json'), 'w') as f:
        json.dump(feature_signature, f, indent=2, ensure_ascii=False)
    print("[原有特征签名] 已生成")

def add_all_features(df):
    """完整保留原有处理流程"""
    # 保持原有执行顺序
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_festival_features(df)
    df = add_season_features(df)
    df = add_rolling_features(df)
    
    # 保持原有生肖过滤
    df = df[df['zodiac'].isin(FIXED_ZODIACS)]
    
    # 保持原有验证
    if df.isnull().values.any():
        print("[原有警告] 存在空值，自动填充")
        df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    """完整保留原有测试代码"""
    print("===== 测试原有数据处理器 =====")
    test_dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'date': test_dates,
        'zodiac': ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡'] * 10
    })
    
    test_df = add_all_features(test_df)
    print("\n测试结果:")
    print(test_df[['date', 'zodiac', 'rolling_7d_鼠', 'heat_index_7d_鼠']].tail(3))
    
    expected_features = [
        'rolling_7d_鼠', 'rolling_30d_鼠', 'rolling_100d_鼠',
        'heat_index_7d_鼠', 'heat_index_30d_鼠', 'heat_index_100d_鼠'
    ]
    
    missing = [f for f in expected_features if f not in test_df.columns]
    print("\n缺失特征:" if missing else "\n所有特征完整")
    print(missing)
