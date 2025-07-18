import pandas as pd
import numpy as np
from lunarcalendar import Converter, Solar
from datetime import datetime
import hashlib
import json
import os
from config import ML_MODEL_PATH

# 确保模型目录存在
os.makedirs(ML_MODEL_PATH, exist_ok=True)

# 定义固定的生肖列表（确保特征维度稳定）
FIXED_ZODIACS = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]

def preprocess_data(df):
    """数据预处理函数 - 确保日期格式正确"""
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    
    # 确保生肖列有效
    if 'zodiac' not in df.columns:
        df['zodiac'] = "未知"
    df['zodiac'] = df['zodiac'].apply(lambda x: x if x in FIXED_ZODIACS else "未知")
    
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
        
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        if (lunar.month, lunar.day) in lunar_festivals:
            return lunar_festivals[(lunar.month, lunar.day)]
        
        if (dt.month, dt.day) in solar_festivals:
            return solar_festivals[(dt.month, dt.day)]
        
        return "无"
    
    df['festival'] = df['date'].apply(detect_festival)
    df['is_festival'] = (df['festival'] != "无").astype(int)
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

def create_fixed_rolling_features(df, window_size):
    """
    创建固定维度的滚动窗口特征（关键修复）
    确保无论数据量多少，都生成相同维度的特征
    """
    # 创建固定维度的DataFrame模板
    template_cols = (
        [f'rolling_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS] +
        [f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS]
    )
    template = pd.DataFrame(columns=template_cols)
    
    # 如果数据为空，直接返回模板
    if df.empty or 'zodiac' not in df.columns:
        return template
    
    # 创建生肖虚拟变量（确保固定维度）
    zodiac_dummies = pd.get_dummies(df['zodiac'])
    for zodiac in FIXED_ZODIACS:
        if zodiac not in zodiac_dummies.columns:
            zodiac_dummies[zodiac] = 0
    zodiac_dummies = zodiac_dummies[FIXED_ZODIACS]
    
    # 计算滚动频率（关键修复：处理空值）
    rolling = zodiac_dummies.rolling(
        window=window_size,
        min_periods=1
    ).mean().fillna(0)
    
    # 计算长期平均频率（关键修复：处理空值）
    long_term_avg = zodiac_dummies.expanding().mean().fillna(0)
    
    # 计算热度指数（关键修复：安全除法）
    # 使用数组运算避免维度问题
    rolling_arr = rolling.values
    long_term_avg_arr = long_term_avg.values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        heat_index_arr = np.where(
            long_term_avg_arr > 0,
            rolling_arr / long_term_avg_arr,
            0
        )
    
    # 创建结果DataFrame
    rolling_df = pd.DataFrame(
        rolling_arr,
        columns=[f'rolling_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS],
        index=df.index
    )
    
    heat_index_df = pd.DataFrame(
        heat_index_arr,
        columns=[f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS],
        index=df.index
    )
    
    # 合并特征（确保列顺序固定）
    result = pd.concat([rolling_df, heat_index_df], axis=1)[template_cols]
    return result

def add_rolling_features(df):
    """
    添加滚动窗口统计特征（稳定性增强版）
    包括7天、30天、100天滚动频率和热度指数
    """
    if df.empty:
        return df
    
    print("添加滚动窗口统计特征...")
    df = df.sort_values('date').reset_index(drop=True)
    
    # 为每个窗口创建固定维度的特征
    rolling_features = []
    for window in [7, 30, 100]:
        rolling = create_fixed_rolling_features(df, window)
        rolling_features.append(rolling)
    
    # 合并所有特征（关键修复：确保列顺序）
    all_rolling = pd.concat(rolling_features, axis=1)
    
    # 合并到原始数据（关键修复：使用join避免索引问题）
    df = df.join(all_rolling)
    
    # 生成特征签名
    generate_feature_signature(df)
    
    print(f"已添加滚动窗口统计特征: {len(all_rolling.columns)}个特征")
    return df

def generate_feature_signature(df):
    """生成特征签名并保存到文件"""
    feature_signature = {
        'columns': sorted(df.columns.tolist()),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'shape': df.shape,
        'hash': hashlib.sha256(
            ','.join(sorted(df.columns)).encode('utf-8')
        ).hexdigest(),
        'created_at': datetime.now().isoformat()
    }
    
    signature_path = os.path.join(ML_MODEL_PATH, 'feature_signature.json')
    with open(signature_path, 'w') as f:
        json.dump(feature_signature, f, indent=2)
    
    print(f"特征签名已生成并保存: {signature_path}")

def add_all_features(df):
    """一次性添加所有特征（稳定性增强版）"""
    # 固定特征添加顺序
    df = add_temporal_features(df)
    df = add_lunar_features(df)
    df = add_festival_features(df)
    df = add_season_features(df)
    df = add_rolling_features(df)  # 最后添加滚动特征
    
    # 确保只包含固定生肖
    df = df[df['zodiac'].isin(FIXED_ZODIACS)]
    
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
    
    # 验证特征维度
    feature_count = len(test_df.columns)
    print(f"\n特征总数: {feature_count}")
    
    print("\n===== 测试完成 =====")
