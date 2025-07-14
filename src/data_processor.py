import pandas as pd
from lunarcalendar import Converter, Solar
import holidays
import re
from config import START_YEAR, CURRENT_YEAR
from datetime import datetime
from src.utils import log_error

class DataProcessor:
    """数据处理器类，封装所有数据预处理和特征工程方法"""
    
    def __init__(self):
        self.festival_mappings = {
            'lunar': {
                (1, 1): "春节",
                (1, 15): "元宵",
                (5, 5): "端午",
                (7, 7): "七夕",
                (7, 15): "中元",
                (8, 15): "中秋",
                (9, 9): "重阳"
            },
            'solar': {
                (4, 4): "清明",
                (4, 5): "清明",
                (12, 22): "冬至"
            }
        }
    
    def preprocess_data(self, df):
        """
        数据预处理
        参数:
            df: 原始数据DataFrame
        返回:
            处理后的DataFrame
        """
        try:
            # 确保日期格式正确
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # 添加年份列（如果不存在）
            if 'year' not in df.columns:
                df['year'] = df['date'].dt.year
            
            return df
        except Exception as e:
            error_msg = f"数据预处理错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'preprocess_data',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return pd.DataFrame()

    def add_temporal_features(self, df):
        """
        添加时序特征
        包括: 星期几、月份、季度、是否周末
        """
        df = self.preprocess_data(df)
        
        try:
            # 星期几 (1-7 表示周一到周日)
            df['weekday'] = df['date'].dt.dayofweek + 1
            
            # 月份 (1-12)
            df['month'] = df['date'].dt.month
            
            # 季度 (1-4)
            df['quarter'] = df['date'].dt.quarter
            
            # 是否周末
            df['is_weekend'] = df['weekday'].isin([6, 7]).astype(int)
            
            # 是否月末
            df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
            
            return df
        except Exception as e:
            error_msg = f"添加时序特征错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'add_temporal_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df

    def add_lunar_features(self, df):
        """添加农历特征"""
        df = self.preprocess_data(df)
        
        try:
            # 计算农历日期
            def convert_to_lunar(dt):
                try:
                    return Converter.Solar2Lunar(Solar(dt.year, dt.month, dt.day))
                except Exception as e:
                    print(f"农历转换错误: {e}, 日期: {dt}")
                    return None
            
            df['lunar_date'] = df['date'].apply(convert_to_lunar)
            
            # 提取农历月份和日期
            df['lunar_month'] = df['lunar_date'].apply(
                lambda x: x.month if x else None)
            df['lunar_day'] = df['lunar_date'].apply(
                lambda x: x.day if x else None)
            
            # 是否闰月
            df['is_leap_month'] = df['lunar_date'].apply(
                lambda x: 1 if x and x.isleap else 0)
            
            # 农历年份生肖
            zodiac_order = ["鼠", "牛", "虎", "兔", "龙", "蛇", 
                          "马", "羊", "猴", "鸡", "狗", "猪"]
            df['lunar_zodiac'] = df['lunar_date'].apply(
                lambda x: zodiac_order[(x.year - 4) % 12] if x else None)
            
            return df
        except Exception as e:
            error_msg = f"添加农历特征错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'add_lunar_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df

    def add_festival_features(self, df):
        """添加节日特征"""
        df = self.preprocess_data(df)
        
        try:
            def detect_festival(dt):
                try:
                    solar = Solar(dt.year, dt.month, dt.day)
                    lunar = Converter.Solar2Lunar(solar)
                    
                    # 春节范围：农历正月初一至十五
                    if lunar.month == 1 and 1 <= lunar.day <= 15:
                        return "春节"
                    
                    # 农历节日
                    if (lunar.month, lunar.day) in self.festival_mappings['lunar']:
                        return self.festival_mappings['lunar'][(lunar.month, lunar.day)]
                    
                    # 公历节日
                    if (dt.month, dt.day) in self.festival_mappings['solar']:
                        return self.festival_mappings['solar'][(dt.month, dt.day)]
                    
                    return "无"
                except Exception as e:
                    print(f"节日检测错误: {e}, 日期: {dt}")
                    return "无"
            
            df['festival'] = df['date'].apply(detect_festival)
            
            # 是否节日
            df['is_festival'] = df['festival'].apply(lambda x: 1 if x != "无" else 0)
            
            return df
        except Exception as e:
            error_msg = f"添加节日特征错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'add_festival_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df

    def add_season_features(self, df):
        """添加季节特征"""
        df = self.preprocess_data(df)
        
        try:
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
            
            # 季节编码
            season_map = {"春": 1, "夏": 2, "秋": 3, "冬": 4}
            df['season_code'] = df['season'].map(season_map)
            
            return df
        except Exception as e:
            error_msg = f"添加季节特征错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'add_season_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df

    def add_rolling_features(self, df, zodiac_col='zodiac', window_sizes=[7, 30, 90]):
        """
        添加滚动窗口特征
        参数:
            df: 输入DataFrame
            zodiac_col: 生肖列名
            window_sizes: 窗口大小列表
        返回:
            添加了滚动特征的DataFrame
        """
        df = self.preprocess_data(df)
        
        try:
            # 确保数据按日期排序
            df = df.sort_values('date')
            
            for window in window_sizes:
                # 滚动窗口频率
                roll_col = f'roll_{window}d_freq'
                df[roll_col] = df[zodiac_col].rolling(
                    window=window, 
                    min_periods=1
                ).apply(lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0)
                
                # 滚动窗口热度指数
                hot_col = f'roll_{window}d_hot'
                total = len(df)
                df[hot_col] = df[zodiac_col].rolling(
                    window=window, 
                    min_periods=1
                ).apply(lambda x: len(x) / total * 100 if len(x) > 0 else 0)
            
            return df
        except Exception as e:
            error_msg = f"添加滚动特征错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'add_rolling_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df

    def process_all_features(self, df):
        """执行所有特征处理"""
        try:
            print("开始特征工程处理...")
            df = self.add_temporal_features(df)
            df = self.add_lunar_features(df)
            df = self.add_festival_features(df)
            df = self.add_season_features(df)
            df = self.add_rolling_features(df)
            print("特征工程处理完成")
            return df
        except Exception as e:
            error_msg = f"特征工程处理错误: {str(e)}"
            print(error_msg)
            log_error({
                'error_type': 'process_all_features',
                'error_msg': error_msg,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            return df


# 测试代码
if __name__ == "__main__":
    print("===== 测试数据处理器 =====")
    
    # 创建测试数据
    test_dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    test_df = pd.DataFrame({
        'date': test_dates,
        'special': [i % 49 + 1 for i in range(365)],  # 模拟特别号码
        'zodiac': ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"] * 30 + ["鼠"]*5
    })
    
    processor = DataProcessor()
    
    print("\n测试预处理...")
    test_df = processor.preprocess_data(test_df)
    print(test_df.info())
    
    print("\n测试时序特征...")
    test_df = processor.add_temporal_features(test_df)
    print(test_df[['date', 'weekday', 'month', 'quarter', 'is_weekend']].head(10))
    
    print("\n测试农历特征...")
    test_df = processor.add_lunar_features(test_df)
    print(test_df[['date', 'lunar_month', 'lunar_day', 'is_leap_month', 'lunar_zodiac']].head(10))
    
    print("\n测试节日特征...")
    test_df = processor.add_festival_features(test_df)
    print(test_df[test_df['festival'] != '无'][['date', 'festival', 'is_festival']].head(10))
    
    print("\n测试季节特征...")
    test_df = processor.add_season_features(test_df)
    print(test_df[['date', 'season', 'season_code']].head(10))
    
    print("\n测试滚动特征...")
    test_df = processor.add_rolling_features(test_df)
    print(test_df[['date', 'zodiac', 'roll_7d_freq', 'roll_30d_hot']].tail(10))
    
    print("\n测试完整处理流程...")
    full_test_df = processor.process_all_features(test_df.copy())
    print(full_test_df.info())
    
    print("\n===== 测试完成 =====")
