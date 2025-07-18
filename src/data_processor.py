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
    required_columns = {
        'year': df['date'].dt.year,
        'zodiac': np.random.choice(FIXED_ZODIACS, size=len(df)),
        'festival': "无",
        'is_festival': 0  # 先初始化为0，后续会正确填充
    }
    
    for col, default in required_columns.items():
        if col not in df.columns:
            df[col] = default
    
    # 确保生肖列只包含固定生肖
    df['zodiac'] = df['zodiac'].apply(
        lambda x: x if x in FIXED_ZODIACS else np.random.choice(FIXED_ZODIACS)
    )
    return df

def add_temporal_features(df):
    """添加基础时序特征：星期几、月份、季度"""
    df = preprocess_data(df)
    df['weekday'] = df['date'].dt.dayofweek + 1  # 1=周一, 7=周日
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    print(f"[时序特征] 添加完成 | 形状: {df.shape}")
    return df

def add_lunar_features(df):
    """添加农历特征：农历月份、日期、是否闰月"""
    df = preprocess_data(df)
    
    def safe_lunar_conversion(dt):
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
    df['lunar'] = df['date'].apply(safe_lunar_conversion)
    df['lunar_month'] = df['lunar'].apply(lambda x: x.month if x else 0)
    df['lunar_day'] = df['lunar'].apply(lambda x: x.day if x else 0)
    df['is_leap_month'] = df['lunar'].apply(lambda x: 1 if x and x.isleap else 0)
    print(f"[农历特征] 添加完成 | 形状: {df.shape}")
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
        
        # 农历节日映射
        lunar_map = {
            (1, 1): "春节",
            (1, 15): "元宵",
            (5, 5): "端午",
            (7, 7): "七夕",
            (7, 15): "中元",
            (8, 15): "中秋",
            (9, 9): "重阳"
        }
        
        # 公历节日映射
        solar_map = {
            (4, 4): "清明",
            (4, 5): "清明",
            (12, 22): "冬至"
        }
        
        # 春节识别
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        lunar_date = (lunar.month, lunar.day)
        if lunar_date in lunar_map:
            return lunar_map[lunar_date]
        
        solar_date = (dt.month, dt.day)
        if solar_date in solar_map:
            return solar_map[solar_date]
        
        return "无"
    
    # 确保只使用固定节日类别
    df['festival'] = df['date'].apply(detect_festival)
    df['festival'] = df['festival'].apply(
        lambda x: x if x in FIXED_FESTIVALS else "无"
    )
    
    # 安全添加is_festival列（关键修复）
    df['is_festival'] = (df['festival'] != "无").astype(int)
    print(f"[节日特征] 添加完成 | 节日类型: {len(FIXED_FESTIVALS)} | 形状: {df.shape}")
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
    print(f"[季节特征] 添加完成 | 形状: {df.shape}")
    return df

def create_fixed_rolling_features(df, window_size):
    """
    创建固定维度的滚动窗口特征（关键修复）
    确保无论数据量多少，都生成相同维度的特征
    """
    # 创建固定维度的DataFrame模板
    template_cols = [
        f'rolling_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS
    ] + [
        f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS
    ]
    template = pd.DataFrame(columns=template_cols)
    
    # 如果数据为空或缺少必要列，直接返回模板
    if df.empty or 'zodiac' not in df.columns:
        return template
    
    try:
        # 创建生肖虚拟变量（确保维度稳定）
        zodiac_dummies = pd.get_dummies(
            df['zodiac'],
            columns=FIXED_ZODIACS
        ).reindex(columns=FIXED_ZODIACS, fill_value=0)
        
        # 计算滚动频率（处理边界情况）
        rolling = zodiac_dummies.rolling(
            window=window_size,
            min_periods=1,
            closed='both'
        ).mean().fillna(0)
        rolling.columns = [f'rolling_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS]
        
        # 计算长期平均频率（安全计算）
        long_term_avg = zodiac_dummies.expanding(min_periods=1).mean().fillna(0)
        
        # 安全计算热度指数（处理除零）
        with np.errstate(divide='ignore', invalid='ignore'):
            heat_index = np.where(
                long_term_avg > 0.001,  # 避免极小值
                rolling.values / long_term_avg.values,
                1.0  # 默认值
            )
        
        heat_index = pd.DataFrame(
            heat_index,
            columns=[f'heat_index_{window_size}d_{zodiac}' for zodiac in FIXED_ZODIACS],
            index=df.index
        )
        
        # 合并特征（确保列顺序）
        result = pd.concat([rolling, heat_index], axis=1)[template_cols]
        return result
    except Exception as e:
        print(f"滚动特征生成失败: {str(e)}")
        return template

def add_rolling_features(df):
    """
    添加滚动窗口统计特征（稳定性增强版）
    包括7天、30天、100天滚动频率和热度指数
    """
    if df.empty:
        return df
    
    print(f"[滚动特征] 开始添加... | 输入形状: {df.shape}")
    df = df.sort_values('date').reset_index(drop=True)
    
    try:
        # 为每个窗口创建固定维度的特征
        rolling_features = []
        for window in [7, 30, 100]:
            features = create_fixed_rolling_features(df, window)
            rolling_features.append(features)
            print(f"窗口 {window} 天特征生成完成 | 特征数: {len(features.columns)}")
        
        # 合并所有滚动特征
        all_rolling = pd.concat(rolling_features, axis=1)
        
        # 安全合并到原始数据
        df = pd.concat([df, all_rolling], axis=1)
        
        # 生成特征签名
        generate_feature_signature(df)
        
        print(f"[滚动特征] 添加完成 | 总特征数: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"滚动特征添加失败: {str(e)}")
        raise ValueError(f"滚动特征生成错误: {str(e)}")

def generate_feature_signature(df):
    """生成特征签名并保存到文件"""
    signature = {
        'version': '1.3.0',
        'timestamp': datetime.now().isoformat(),
        'columns': sorted(df.columns.tolist()),
        'dtypes': {col: str(df[col].dtype) for col in df.columns},
        'shape': df.shape,
        'hash': hashlib.sha256(
            str(sorted(df.columns)).encode('utf-8')
        ).hexdigest()
    }
    
    os.makedirs(ML_MODEL_PATH, exist_ok=True)
    signature_path = os.path.join(ML_MODEL_PATH, 'feature_signature.json')
    
    with open(signature_path, 'w', encoding='utf-8') as f:
        json.dump(signature, f, indent=2, ensure_ascii=False)
    
    print(f"特征签名已保存: {signature_path}")

def add_all_features(df):
    """
    一次性添加所有特征（生产环境使用）
    返回:
        DataFrame: 包含所有特征的DataFrame
    """
    # 特征添加步骤（固定顺序）
    steps = [
        ('基础时序', add_temporal_features),
        ('农历', add_lunar_features),
        ('节日', add_festival_features),
        ('季节', add_season_features),
        ('滚动窗口', add_rolling_features)
    ]
    
    for name, func in steps:
        try:
            print(f"\n=== 开始添加 {name} 特征 ===")
            start_shape = df.shape
            df = func(df)
            print(f"添加成功 | 形状变化: {start_shape} -> {df.shape}")
            
            # 空值检查
            if df.isnull().any().any():
                null_cols = df.columns[df.isnull().any()].tolist()
                print(f"警告: 发现空值列 - {null_cols}，正在自动填充...")
                df = df.fillna(0)
        except Exception as e:
            print(f"特征添加失败 [{name}]: {str(e)}")
            raise ValueError(f"{name} 特征添加错误: {str(e)}")
    
    # 最终验证
    validate_features(df)
    return df

def validate_features(df):
    """验证特征完整性"""
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
                raise ValueError(f"缺失热度指数: heat_index_{window}d_{zodiac}")
    
    print(f"特征验证通过 | 总列数: {len(df.columns)}")

if __name__ == "__main__":
    print("===== 数据处理器测试 =====")
    
    # 测试数据
    test_data = {
        'date': pd.date_range('2023-01-01', periods=100),
        'zodiac': np.random.choice(FIXED_ZODIACS, 100)
    }
    test_df = pd.DataFrame(test_data)
    
    # 空数据测试
    print("\n[测试1] 空数据处理...")
    try:
        empty_result = add_all_features(pd.DataFrame())
        print("空数据处理成功")
    except Exception as e:
        print(f"空数据处理失败: {str(e)}")
    
    # 正常数据测试
    print("\n[测试2] 正常数据处理...")
    try:
        result = add_all_features(test_df.copy())
        print("\n处理结果样例:")
        print(result.iloc[-5:, :5])
        
        # 验证关键特征
        assert 'is_festival' in result.columns
        assert result['is_festival'].isin([0, 1]).all()
        print("\n所有测试通过")
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
