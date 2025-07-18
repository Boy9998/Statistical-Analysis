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

# 定义固定节日列表（确保特征维度稳定）
FIXED_FESTIVALS = ["春节", "元宵", "端午", "七夕", "中元", "中秋", "重阳", "清明", "冬至", "无"]

def preprocess_data(df):
    """数据预处理函数 - 确保日期格式正确并初始化必要列"""
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 初始化必要列（关键修复）
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year
    if 'zodiac' not in df.columns:
        df['zodiac'] = "未知"  # 默认值
    if 'festival' not in df.columns:
        df['festival'] = "无"  # 默认值
    
    # 确保生肖列只包含固定生肖
    df['zodiac'] = df['zodiac'].apply(lambda x: x if x in FIXED_ZODIACS else "未知")
    return df

def add_temporal_features(df):
    """添加基础时序特征：星期几、月份、季度"""
    df = preprocess_data(df)
    df['weekday'] = df['date'].dt.dayofweek + 1  # 1=周一, 7=周日
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
            lunar = Converter.Solar2Lunar(solar)
            return lunar
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
    """添加节日特征（稳定性增强版）"""
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
    
    # 确保只使用固定节日类别
    df['festival'] = df['date'].apply(detect_festival).apply(
        lambda x: x if x in FIXED_FESTIVALS else "无"
    )
    
    # 添加是否为节日的标志列（关键修复）
    df['is_festival'] = (df['festival'] != "无").astype(int)
    
    print(f"已添加节日特征: 共{len(FIXED_FESTIVALS)}种节日类型")
    return df

def add_season_features(df):
    """添加季节特征（固定维度）"""
    df = preprocess_data(df)
    
    season_map = {
        1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
        6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋",
        11: "秋", 12: "冬"
    }
    
    df['season'] = df['date'].dt.month.map(season_map)
    print(f"已添加季节特征")
    return df

def create_fixed_rolling_features(df, window_size):
    """
    创建固定维度的滚动窗口特征（稳定性关键修复）
    确保无论数据量多少，都生成相同维度的特征
    """
    # 创建固定维度的DataFrame模板
    template_cols = [
        f'rolling_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS
    ] + [
        f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS
    ]
    template = pd.DataFrame(columns=template_cols)
    
    # 如果数据为空，直接返回模板
    if df.empty or 'zodiac' not in df.columns:
        return template
    
    # 创建生肖虚拟变量（确保只包含固定生肖）
    zodiac_dummies = pd.get_dummies(df['zodiac']).reindex(columns=FIXED_ZODIACS, fill_value=0)
    
    # 计算滚动频率（关键修复：处理NaN值）
    rolling = zodiac_dummies.rolling(
        window=window_size,
        min_periods=1
    ).mean().fillna(0)
    rolling.columns = [f'rolling_{window_size}d_{zodiac}' for zodiac in rolling.columns]
    
    # 计算长期平均频率（关键修复：处理NaN值）
    long_term_avg = zodiac_dummies.expanding().mean().fillna(0)
    
    # 计算热度指数（关键修复：处理除零错误）
    with np.errstate(divide='ignore', invalid='ignore'):
        heat_index = np.where(
            long_term_avg > 0,
            rolling / long_term_avg,
            0
        )
    heat_index = pd.DataFrame(
        heat_index,
        columns=[f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS],
        index=df.index
    )
    
    # 合并特征（确保列顺序固定）
    result = pd.concat([rolling, heat_index], axis=1)[template_cols]
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
    rolling_7d = create_fixed_rolling_features(df, 7)
    rolling_30d = create_fixed_rolling_features(df, 30)
    rolling_100d = create_fixed_rolling_features(df, 100)
    
    # 合并所有特征（关键修复：确保列顺序）
    all_rolling = pd.concat([rolling_7d, rolling_30d, rolling_100d], axis=1)
    
    # 合并到原始数据（关键修复：使用join避免索引问题）
    df = df.join(all_rolling)
    
    # 生成特征签名
    generate_feature_signature(df)
    
    print(f"已添加滚动窗口统计特征: {len(all_rolling.columns)}个特征")
    return df

def generate_feature_signature(df):
    """生成特征签名并保存到文件（关键修复：增强校验）"""
    # 创建特征签名
    feature_signature = {
        'columns': sorted(df.columns.tolist()),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'shape': df.shape,
        'hash': hashlib.sha256(
            ','.join(sorted(df.columns)).encode('utf-8')
        ).hexdigest(),
        'created_at': datetime.now().isoformat(),
        'version': '1.1.0'  # 特征工程版本号
    }
    
    # 保存到文件
    signature_path = os.path.join(ML_MODEL_PATH, 'feature_signature.json')
    with open(signature_path, 'w', encoding='utf-8') as f:
        json.dump(feature_signature, f, indent=2, ensure_ascii=False)
    
    print(f"特征签名已生成并保存: {signature_path}")

def add_all_features(df):
    """
    一次性添加所有特征（稳定性增强版）
    确保特征添加顺序和维度一致性
    """
    # 固定特征添加顺序
    feature_steps = [
        ('基础时序', add_temporal_features),
        ('农历', add_lunar_features),
        ('节日', add_festival_features),
        ('季节', add_season_features),
        ('滚动窗口', add_rolling_features)
    ]
    
    for step_name, step_func in feature_steps:
        print(f"\n=== 正在添加 {step_name} 特征 ===")
        try:
            df = step_func(df)
            # 检查特征维度
            if df.isnull().values.any():
                print(f"警告: {step_name} 步骤发现空值，已自动填充")
                df = df.fillna(0)
        except Exception as e:
            print(f"特征添加错误 [{step_name}]: {str(e)}")
            raise
    
    # 最终校验
    validate_features(df)
    return df

def validate_features(df):
    """验证特征完整性（关键修复）"""
    required_columns = {
        'date', 'year', 'weekday', 'month', 'quarter',
        'lunar_month', 'lunar_day', 'is_leap_month',
        'festival', 'is_festival', 'season'
    }
    
    # 检查基础列
    missing_base = required_columns - set(df.columns)
    if missing_base:
        raise ValueError(f"缺失基础特征列: {missing_base}")
    
    # 检查滚动特征
    for window in [7, 30, 100]:
        for zodiac in FIXED_ZODIACS:
            if f'rolling_{window}d_{zodiac}' not in df.columns:
                raise ValueError(f"缺失滚动特征: rolling_{window}d_{zodiac}")
            if f'heat_index_{window}d_{zodiac}' not in df.columns:
                raise ValueError(f"缺失热度指数特征: heat_index_{window}d_{zodiac}")
    
    print("所有特征验证通过")

# 测试函数
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'date': test_dates,
        'zodiac': ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡'] * 10
    })
    
    # 测试空数据情况
    empty_df = pd.DataFrame(columns=['date', 'zodiac'])
    print("\n测试空数据处理...")
    try:
        empty_processed = add_all_features(empty_df)
        print("空数据处理测试通过")
    except Exception as e:
        print(f"空数据处理失败: {str(e)}")
    
    # 添加所有特征
    print("\n测试正常数据处理...")
    test_df = add_all_features(test_df)
    
    # 打印结果
    print("\n测试结果:")
    print(test_df[['date', 'zodiac', 'festival', 'is_festival', 'rolling_7d_鼠', 'heat_index_7d_鼠']].tail(10))
    
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
    print(f"\n特征总数: {feature_count} (预期: 53+)")
    
    print("\n===== 测试完成 =====")
