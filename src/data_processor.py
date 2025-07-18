import pandas as pd
import numpy as np
from lunarcalendar import Converter, Solar
from datetime import datetime
import hashlib
import json
import os
from typing import Dict, List
from config import ML_MODEL_PATH

# 常量定义（不可变）
FIXED_ZODIACS: List[str] = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
FIXED_FESTIVALS: List[str] = ["春节", "元宵", "端午", "七夕", "中元", "中秋", "重阳", "清明", "冬至", "无"]
SEASONS: Dict[int, str] = {
    1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春",
    6: "夏", 7: "夏", 8: "夏", 9: "秋", 10: "秋",
    11: "秋", 12: "冬"
}

class FeatureEngineer:
    """特征工程处理器（线程安全）"""
    
    def __init__(self):
        self.feature_template = self._create_feature_template()
        
    def _create_feature_template(self) -> Dict[str, type]:
        """创建特征模板（保证维度一致）"""
        template = {
            # 基础特征
            'date': np.datetime64,
            'year': np.int16,
            'zodiac': object,
            'festival': object,
            'is_festival': np.int8,
            
            # 时序特征
            'weekday': np.int8,
            'month': np.int8,
            'quarter': np.int8,
            
            # 农历特征
            'lunar_month': np.int8,
            'lunar_day': np.int8,
            'is_leap_month': np.int8,
            
            # 季节特征
            'season': object
        }
        
        # 动态生成固定维度特征
        for z in FIXED_ZODIACS:
            template.update({
                f'freq_{z}': np.float32,
                f'season_{z}': np.float32,
                f'festival_{z}': np.float32
            })
            
        for w in [7, 30, 100]:
            for z in FIXED_ZODIACS:
                template.update({
                    f'rolling_{w}d_{z}': np.float32,
                    f'heat_{w}d_{z}': np.float32
                })
                
        for f in FIXED_FESTIVALS:
            if f != "无":
                template[f'fest_{f}'] = np.int8
                
        return template

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理（强制类型转换）"""
        # 基础列检查
        if 'date' not in df.columns:
            raise ValueError("输入数据必须包含date列")
            
        # 初始化DataFrame
        processed = pd.DataFrame(index=df.index)
        
        # 强制类型转换
        processed['date'] = pd.to_datetime(df['date'])
        processed['year'] = processed['date'].dt.year
        
        # 处理生肖列
        processed['zodiac'] = (
            df.get('zodiac', '未知')
            .apply(lambda x: x if x in FIXED_ZODIACS else np.random.choice(FIXED_ZODIACS))
            
        # 初始化其他列
        for col, dtype in self.feature_template.items():
            if col not in processed.columns:
                if dtype == object:
                    processed[col] = ""
                elif issubclass(dtype, np.number):
                    processed[col] = 0
                    
        return processed

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时序特征（线程安全）"""
        df = df.copy()
        df['weekday'] = df['date'].dt.dayofweek + 1  # 1-7
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        return df

    def add_lunar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加农历特征（带异常保护）"""
        def safe_lunar_conversion(dt):
            try:
                solar = Solar(dt.year, dt.month, dt.day)
                lunar = Converter.Solar2Lunar(solar)
                return lunar
            except Exception:
                return None
                
        df = df.copy()
        lunar_data = df['date'].apply(safe_lunar_conversion)
        
        df['lunar_month'] = lunar_data.apply(lambda x: x.month if x else 0)
        df['lunar_day'] = lunar_data.apply(lambda x: x.day if x else 0)
        df['is_leap_month'] = lunar_data.apply(lambda x: 1 if x and x.isleap else 0)
        return df

    def add_festival_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """节日特征处理（严格维度控制）"""
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
                
                # 春节特殊处理
                if lunar.month == 1 and lunar.day == 1:
                    return "春节"
                    
                return festival_map.get(
                    (lunar.month, lunar.day),
                    festival_map.get((dt.month, dt.day), "无")
            except Exception:
                return "无"
                
        df = df.copy()
        df['festival'] = df['date'].apply(detect_festival)
        df['is_festival'] = df['festival'].ne("无").astype(np.int8)
        
        # 节日虚拟变量
        for fest in FIXED_FESTIVALS:
            if fest != "无":
                df[f'fest_{fest}'] = (df['festival'] == fest).astype(np.int8)
                
        return df

    def add_season_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """季节特征（固定映射）"""
        df = df.copy()
        df['season'] = df['date'].dt.month.map(SEASONS)
        return df

    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """滚动特征（维度保证）"""
        if df.empty:
            return df
            
        # 创建生肖虚拟变量
        zodiac_dummies = pd.get_dummies(
            df['zodiac']
        ).reindex(columns=FIXED_ZODIACS, fill_value=0)
        
        # 计算滚动特征
        for window in [7, 30, 100]:
            # 滚动频率
            rolling = zodiac_dummies.rolling(window, min_periods=1).mean()
            rolling.columns = [f'rolling_{window}d_{z}' for z in FIXED_ZODIACS]
            df = df.join(rolling)
            
            # 热度指数
            with np.errstate(divide='ignore', invalid='ignore'):
                heat = rolling / zodiac_dummies.expanding().mean()
            heat = heat.replace([np.inf, -np.inf], 0)
            heat.columns = [f'heat_{window}d_{z}' for z in FIXED_ZODIACS]
            df = df.join(heat)
            
        return df

    def validate(self, df: pd.DataFrame) -> None:
        """严格特征验证"""
        missing = set(self.feature_template) - set(df.columns)
        if missing:
            raise ValueError(f"缺失特征列: {missing}")
            
        if df.isnull().any().any():
            raise ValueError("存在空值特征")
            
        # 检查生肖值
        invalid_zodiac = ~df['zodiac'].isin(FIXED_ZODIACS)
        if invalid_zodiac.any():
            raise ValueError(f"无效生肖值: {df.loc[invalid_zodiac, 'zodiac'].unique()}")

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整特征工程"""
        # 预处理
        df = self.preprocess(df)
        
        # 特征流水线
        processors = [
            self.add_temporal_features,
            self.add_lunar_features,
            self.add_festival_features,
            self.add_season_features,
            self.add_rolling_features
        ]
        
        for processor in processors:
            df = processor(df)
            
        # 最终验证
        self.validate(df)
        self._save_feature_signature(df)
        
        return df

    def _save_feature_signature(self, df: pd.DataFrame) -> None:
        """保存特征签名"""
        signature = {
            "version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "columns": sorted(df.columns.tolist()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape
        }
        
        os.makedirs(ML_MODEL_PATH, exist_ok=True)
        with open(os.path.join(ML_MODEL_PATH, 'feature_signature.json'), 'w') as f:
            json.dump(signature, f, indent=2)

# 兼容旧接口
def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """对外接口（保持兼容）"""
    return FeatureEngineer().process(df)
