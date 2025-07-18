import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping, log_error
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar  # 精确农历计算
from collections import defaultdict
from src.strategy_manager import StrategyManager  # 导入增强的策略管理器
import warnings
import os
from sklearn.model_selection import TimeSeriesSplit, KFold
from scipy.stats import variation
from src.ml_predictor import MLPredictor  # 导入ML预测器

# 兼容不同版本 Pandas 的 SettingWithCopyWarning
try:
    # 新版 Pandas
    from pandas.errors import SettingWithCopyWarning
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
except ImportError:
    try:
        # 旧版 Pandas
        from pandas.core.common import SettingWithCopyWarning
        warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    except (ImportError, AttributeError):
        # 如果两个位置都没有，忽略警告
        pass

class LotteryAnalyzer:
    def __init__(self):
        """初始化分析器，获取历史数据并处理生肖映射"""
        print("开始获取历史数据...")
        self.df = fetch_historical_data()
        if not self.df.empty:
            print(f"成功获取 {len(self.df)} 条历史开奖记录")
            
            # 应用基于年份的动态生肖映射
            self.df['zodiac'] = self.df.apply(
                lambda row: zodiac_mapping(row['special'], row['year']), axis=1
            )
            
            # 确保生肖数据有效
            self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
            valid_zodiacs = self.df['zodiac'].isin(self.zodiacs)
            if not valid_zodiacs.all():
                invalid_count = len(self.df) - valid_zodiacs.sum()
                print(f"警告：发现 {invalid_count} 条记录的生肖映射无效")
                self.df = self.df[valid_zodiacs].copy()
            
            # 添加农历和节日信息
            print("添加农历和节日信息...")
            self.df['lunar'] = self.df['date'].apply(self.get_lunar_date)
            self.df['festival'] = self.df['date'].apply(self.detect_festival)
            # 确保 is_festival 列存在
            if 'is_festival' not in self.df.columns:
                self.df['is_festival'] = (self.df['festival'] != "无").astype(int)
            self.df['season'] = self.df['date'].apply(self.get_season)
            
            # 添加时序特征
            print("添加时序特征...")
            self.df['weekday'] = self.df['date'].dt.weekday  # 0=周一, 6=周日
            self.df['month'] = self.df['date'].dt.month
            self.df['quarter'] = self.df['date'].dt.quarter
            
            # 添加生肖特征（关键修复）
            self.add_zodiac_features()
            
            # 添加滚动窗口特征（关键修复）
            self.add_rolling_features()
            
            # 初始化增强的策略管理器
            self.strategy_manager = StrategyManager()
            
            # 初始化ML预测器
            self.ml_predictor = MLPredictor()
            
            # 检测历史模式
            print("检测历史模式...")
            self.patterns = self.detect_patterns()
            
            # 更新组合概率和特征重要性
            self.strategy_manager.update_combo_probs(self.df)
            self.strategy_manager.evaluate_feature_importance(self.df)
            
            # 打印最新开奖信息
            latest = self.df.iloc[-1]
            print(f"最新开奖记录: 期号 {latest['expect']}, 日期 {latest['date'].date()}, 生肖 {latest['zodiac']}")
            
            # 创建错误分析目录
            os.makedirs('error_analysis', exist_ok=True)
            os.makedirs('data', exist_ok=True)  # 确保数据目录存在
            
            # 初始化预测记录文件 - 修复：确保文件存在
            self.initialize_prediction_file()
        else:
            print("警告：未获取到任何有效数据")
            self.strategy_manager = StrategyManager()
            self.ml_predictor = MLPredictor()
            self.patterns = {}
            self.initialize_prediction_file()
    
    def initialize_prediction_file(self):
        """初始化预测记录文件 - 修复：确保文件存在"""
        prediction_file = 'data/predictions.csv'
        
        # 如果文件不存在，创建并写入表头
        if not os.path.exists(prediction_file):
            with open(prediction_file, 'w') as f:
                f.write("期号,日期,上期生肖,预测生肖,实际生肖,是否命中\n")
            print(f"已创建预测记录文件: {prediction_file}")
    
    def save_prediction_record(self, expect, date, last_zodiac, predicted_zodiacs, actual_zodiac, is_hit):
        """保存预测记录到文件 - 修复：确保文件存在"""
        prediction_file = 'data/predictions.csv'
        
        # 如果文件不存在，初始化
        if not os.path.exists(prediction_file):
            self.initialize_prediction_file()
        
        # 将预测生肖列表转换为字符串
        predicted_str = ','.join(predicted_zodiacs)
        
        # 追加记录
        with open(prediction_file, 'a') as f:
            f.write(f"{expect},{date},{last_zodiac},{predicted_str},{actual_zodiac},{int(is_hit)}\n")
    
    def add_zodiac_features(self):
        """添加生肖相关特征 - 关键修复"""
        print("添加生肖特征...")
        
        # 1. 生肖频率特征
        zodiac_counts = self.df['zodiac'].value_counts(normalize=True)
        for zodiac in self.zodiacs:
            self.df[f'freq_{zodiac}'] = zodiac_counts.get(zodiac, 0.0)
        
        # 2. 季节生肖特征
        for zodiac in self.zodiacs:
            # 为每个生肖创建季节特征列
            self.df[f'season_{zodiac}'] = 0.0
        
        # 计算每个季节中每个生肖的频率
        for season in ['春', '夏', '秋', '冬']:
            # 使用 .loc 进行显式行选择（关键修复）
            season_mask = (self.df['season'] == season)
            season_data = self.df.loc[season_mask]
            if not season_data.empty:
                season_counts = season_data['zodiac'].value_counts(normalize=True)
                for zodiac in self.zodiacs:
                    # 更新季节特征值
                    self.df.loc[season_mask, f'season_{zodiac}'] = season_counts.get(zodiac, 0.0)
        
        # 3. 节日生肖特征
        for zodiac in self.zodiacs:
            # 为每个生肖创建节日特征列
            self.df[f'festival_{zodiac}'] = 0.0
        
        # 计算节日中每个生肖的频率
        # 使用 .loc 进行显式行选择（关键修复）
        festival_mask = (self.df['is_festival'] == 1)
        festival_data = self.df.loc[festival_mask]
        if not festival_data.empty:
            festival_counts = festival_data['zodiac'].value_counts(normalize=True)
            for zodiac in self.zodiacs:
                # 更新节日特征值
                self.df.loc[festival_mask, f'festival_{zodiac}'] = festival_counts.get(zodiac, 0.0)
        
        print(f"已添加生肖特征: {len(self.zodiacs)*3}个新特征")
    
    def add_rolling_features(self):
        """添加滚动窗口特征 - 关键修复（使用期数窗口替代自然日窗口）"""
        print("添加滚动窗口特征...")
        # 使用期数窗口替代自然日窗口
        periods = [7, 30]  # 7期和30期窗口
        
        # 创建生肖出现标志
        for zodiac in self.zodiacs:
            self.df[f'occur_{zodiac}'] = (self.df['zodiac'] == zodiac).astype(int)
        
        for period in periods:
            # 为每个生肖添加滚动窗口特征
            for zodiac in self.zodiacs:
                # 计算滚动频率 - 使用期数窗口
                self.df[f'rolling_{period}p_{zodiac}'] = (
                    self.df[f'occur_{zodiac}'].rolling(window=period, min_periods=1).mean()
                )
        
        print(f"已添加滚动窗口特征: {len(periods)*len(self.zodiacs)}个新特征（使用期数窗口）")
    
    def get_lunar_date(self, dt):
        """精确转换公历到农历 - 增强健壮性"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            # 返回一个默认的农历日期对象
            return Lunar(dt.year, dt.month, dt.day, False)
    
    def detect_festival(self, dt):
        """识别传统节日 - 增强健壮性"""
        lunar = self.get_lunar_date(dt)
        
        # 处理无效的农历日期
        if not isinstance(lunar, Lunar):
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
        
        # 春节识别优化：仅正月初一
        if lunar.month == 1 and lunar.day == 1:
            return "春节"
        
        # 精确匹配
        if lunar_date in lunar_festivals:
            return lunar_festivals[lunar_date]
        if solar_date in solar_festivals:
            return solar_festivals[solar_date]
        
        return "无"
    
    def get_season(self, dt):
        """获取季节"""
        month = dt.month
        if 3 <= month <= 5:
            return "春"
        elif 6 <= month <= 8:
            return "夏"
        elif 9 <= month <= 11:
            return "秋"
        else:
            return "冬"
    
    def detect_patterns(self):
        """检测历史模式（连续出现、间隔模式、节日效应） - 增强量化处理"""
        patterns = {
            'consecutive': defaultdict(int),
            'intervals': defaultdict(list),
            'festival_boost': defaultdict(lambda: defaultdict(int)),
            'festival_strength': defaultdict(float)  # 新增：节日效应强度
        }
        
        # 确保 is_festival 列存在
        if 'is_festival' not in self.df.columns:
            self.df['is_festival'] = (self.df['festival'] != "无").astype(int)
        
        # 1. 连续出现模式
        consecutive_count = 1
        last_zodiac = None
        for idx, row in self.df.iterrows():
            if row['zodiac'] == last_zodiac:
                consecutive_count += 1
            else:
                if consecutive_count > 1 and last_zodiac:
                    patterns['consecutive'][last_zodiac] = max(patterns['consecutive'][last_zodiac], consecutive_count)
                consecutive_count = 1
                last_zodiac = row['zodiac']
        
        # 处理最后一条记录
        if consecutive_count > 1 and last_zodiac:
            patterns['consecutive'][last_zodiac] = max(patterns['consecutive'][last_zodiac], consecutive_count)
        
        # 2. 间隔出现模式
        last_occurrence = {}
        for idx, row in self.df.iterrows():
            zodiac = row['zodiac']
            if zodiac in last_occurrence:
                interval = idx - last_occurrence[zodiac]
                patterns['intervals'][zodiac].append(interval)
            last_occurrence[zodiac] = idx
        
        # 计算平均间隔和标准差
        for zodiac, intervals in patterns['intervals'].items():
            if intervals:
                patterns['intervals'][zodiac] = {
                    'mean': sum(intervals) / len(intervals),
                    'std': np.std(intervals) if len(intervals) > 1 else 0
                }
        
        # 3. 节日效应模式 - 量化处理
        festival_zodiacs = defaultdict(lambda: defaultdict(int))
        total_festivals = 0
        
        for idx, row in self.df.iterrows():
            if row['is_festival'] == 1:  # 显式检查
                festival = row['festival']
                festival_zodiacs[festival][row['zodiac']] += 1
                total_festivals += 1
        
        # 计算每个节日的生肖频率和效应强度
        for festival, zodiac_counts in festival_zodiacs.items():
            if zodiac_counts:
                total = sum(zodiac_counts.values())
                # 找出主要生肖
                most_common = max(zodiac_counts.items(), key=lambda x: x[1])
                patterns['festival_boost'][festival] = most_common[0]
                
                # 计算效应强度：主要生肖占比
                strength = most_common[1] / total
                patterns['festival_strength'][festival] = strength
                
                print(f"节日效应量化: {festival} -> {most_common[0]} (强度={strength:.2%})")
        
        print(f"检测到模式: {len(patterns['consecutive'])}个连续模式, {len(patterns['intervals'])}个间隔模式, {len(patterns['festival_boost'])}个节日效应")
        return patterns
    
    def analyze_zodiac_patterns(self):
        """分析生肖出现规律"""
        if self.df.empty:
            print("无法进行分析 - 数据为空")
            return {}
        
        print("开始分析生肖出现规律...")
        
        # 1. 生肖频率分析（基于全部历史数据）
        freq = self.df['zodiac'].value_counts().reset_index()
        freq.columns = ['生肖', '出现次数']
        freq['频率(%)'] = round(freq['出现次数'] / len(self.df) * 100, 2)
        
        # 2. 生肖转移分析（基于最近200期）
        if len(self.df) >= BACKTEST_WINDOW:
            recent = self.df.tail(BACKTEST_WINDOW)
            print(f"使用最近{BACKTEST_WINDOW}期数据计算转移矩阵")
            
            # 创建转移矩阵
            transition = pd.crosstab(
                recent['zodiac'].shift(-1), 
                recent['zodiac'], 
                normalize=1
            ).round(4) * 100
        else:
            print(f"数据不足{BACKTEST_WINDOW}期，使用全部数据计算转移矩阵")
            transition = pd.crosstab(
                self.df['zodiac'].shift(-1), 
                self.df['zodiac'], 
                normalize=1
            ).round(4) * 100
        
        return {
            'frequency': freq,
            'transition_matrix': transition
        }
    
    def backtest_strategy(self):
        """动态回测预测策略（自适应窗口+交叉验证）"""
        if self.df.empty:
            print("无法回测 - 数据为空")
            return pd.DataFrame(), 0.0
        
        # 最小和最大窗口设置
        min_window = 100
        max_window = 300
        default_window = BACKTEST_WINDOW
        
        total_records = len(self.df)
        
        # 确保有足够的数据进行回测
        if total_records < min_window + 1:
            print(f"警告：数据不足{min_window+1}期，实际只有{total_records}期")
            return pd.DataFrame(), 0.0
        
        # 计算数据波动性（用于动态调整窗口）
        volatility = self._calculate_data_volatility()
        print(f"数据波动性: {volatility:.4f}")
        
        # 根据波动性动态调整窗口大小
        # 波动性高 -> 使用小窗口快速适应变化
        # 波动性低 -> 使用大窗口利用更多历史数据
        if volatility > 0.6:  # 高波动性
            window = min_window
        elif volatility > 0.4:  # 中波动性
            window = min_window + int((max_window - min_window) * 0.3)
        else:  # 低波动性
            window = max_window
        
        print(f"自适应窗口大小: {window}期 (基于波动性 {volatility:.2%})")
        
        # 计算实际回测次数
        test_count = total_records - window - 1
        if test_count <= 0:
            print(f"警告：窗口大小{window}过大，实际只有{total_records}期数据")
            return pd.DataFrame(), 0.0
        
        print(f"开始回测策略（自适应窗口{window}期，共{test_count}次测试）...")
        
        results = []
        error_patterns = defaultdict(lambda: defaultdict(int))  # 记录错误模式
        
        # 使用单个策略管理器处理整个回测过程
        strategy_manager = StrategyManager()
        
        # 添加交叉验证
        n_splits = 5
        if test_count >= n_splits * 10:  # 确保有足够数据
            print(f"启用 {n_splits}-折交叉验证")
            tscv = TimeSeriesSplit(n_splits=n_splits)
            split_results = []
            
            # 进行时间序列交叉验证
            for train_index, test_index in tscv.split(self.df):
                if len(train_index) < min_window or len(test_index) == 0:
                    continue
                
                # 使用训练数据更新策略
                train = self.df.iloc[train_index].copy()
                self.add_features_to_data(train)
                strategy_manager.update_combo_probs(train)
                strategy_manager.evaluate_factor_validity(train)
                
                # 测试数据
                test = self.df.iloc[test_index]
                
                # 获取特征数据
                feature_row = train.iloc[-1]
                last_zodiac = feature_row['zodiac']
                
                # 预测
                prediction, _ = strategy_manager.generate_prediction(
                    feature_row, last_zodiac
                )
                actual = test['zodiac'].values[0]
                
                # 记录结果
                is_hit = 1 if actual in prediction else 0
                split_results.append(is_hit)
            
            # 计算交叉验证准确率
            cv_accuracy = np.mean(split_results) if split_results else 0.0
            print(f"交叉验证准确率: {cv_accuracy:.2%} ({len(split_results)}次测试)")
        else:
            cv_accuracy = 0.0
            print(f"数据不足{min_window*5}期，跳过交叉验证")
        
        # 主回测循环（使用自适应窗口）
        for i in range(window, total_records - 1):
            # 训练数据：从 i-window 到 i-1 (共window期)
            train = self.df.iloc[i-window:i].copy()
            
            # 为训练数据添加特征
            self.add_features_to_data(train)
            
            # 测试数据：下一期 (i)
            test = self.df.iloc[i:i+1]
            
            # 更新策略管理器的数据
            strategy_manager.update_combo_probs(train)
            strategy_manager.evaluate_factor_validity(train)
            
            # 获取特征数据
            feature_row = train.iloc[-1]
            last_zodiac = feature_row['zodiac']
            
            # 预测
            prediction, _ = strategy_manager.generate_prediction(
                feature_row, last_zodiac
            )
            actual = test['zodiac'].values[0]
            
            # 记录结果
            is_hit = 1 if actual in prediction else 0
            results.append({
                '期号': test['expect'].values[0],
                '上期生肖': last_zodiac,
                '实际生肖': actual,
                '预测生肖': ", ".join(prediction),
                '是否命中': is_hit
            })
            
            # 保存预测记录到CSV文件（新增历史复盘功能） - 修复：使用内部方法
            self.save_prediction_record(
                expect=test['expect'].values[0],
                date=test['date'].dt.strftime('%Y-%m-%d').values[0],
                last_zodiac=last_zodiac,
                predicted_zodiacs=prediction,
                actual_zodiac=actual,
                is_hit=is_hit
            )
            
            # 记录错误模式
            if not is_hit:
                error_data = {
                    'draw_number': test['expect'].values[0],
                    'date': test['date'].dt.strftime('%Y-%m-%d').values[0],
                    'actual_zodiac': actual,
                    'predicted_zodiacs': ",".join(prediction),
                    'last_zodiac': last_zodiac,
                    'weekday': test['weekday'].values[0] if 'weekday' in test.columns else None,
                    'month': test['month'].values[0] if 'month' in test.columns else None,
                    'season': test['season'].values[0] if 'season' in test.columns else None,
                    'festival': test['festival'].values[0] if 'festival' in test.columns else None
                }
                log_error(error_data)
                
                # 记录错误模式：上期生肖 -> 实际生肖
                error_patterns[last_zodiac][actual] += 1
            
            # 每50次迭代输出一次进度
            if len(results) % 50 == 0:
                print(f"回测进度: {len(results)}/{test_count} ({len(results)/test_count*100:.1f}%)")
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            accuracy = result_df['是否命中'].mean()
            hit_count = result_df['是否命中'].sum()
            print(f"回测完成: 准确率={accuracy:.2%}, 命中次数={hit_count}/{len(result_df)}")
            
            # 综合交叉验证结果
            if cv_accuracy > 0:
                combined_accuracy = (accuracy + cv_accuracy) / 2
                print(f"综合准确率 (主回测+交叉验证): {combined_accuracy:.2%}")
            else:
                combined_accuracy = accuracy
            
            # 根据回测结果调整策略，并传入错误模式
            self.strategy_manager.adjust(combined_accuracy, error_patterns)
            
            # 保存错误模式分析
            self.save_error_analysis(error_patterns)
            
            # 检测过拟合
            self.detect_overfitting(result_df, cv_accuracy)
        else:
            combined_accuracy = 0.0
            print("回测完成: 无有效结果")
        
        return result_df, combined_accuracy
    
    def detect_overfitting(self, result_df, cv_accuracy):
        """检测并防止过拟合"""
        if len(result_df) < 50 or cv_accuracy <= 0:
            return
        
        # 计算训练集和验证集的准确率差异
        train_accuracy = result_df['是否命中'].mean()
        accuracy_diff = abs(train_accuracy - cv_accuracy)
        
        print(f"过拟合检测: 训练准确率={train_accuracy:.2%}, 验证准确率={cv_accuracy:.2%}, 差异={accuracy_diff:.2%}")
        
        if accuracy_diff > 0.15:  # 差异超过15%可能过拟合
            print("警告: 检测到可能过拟合（训练集与验证集差异过大）")
            
            # 触发ML模型重新训练
            print("触发ML模型重新训练以防止过拟合...")
            self.ml_predictor.train_model(self.df, retrain=True)
            
            # 调整策略管理器权重
            self.strategy_manager.weights['ml_model'] = max(0.15, self.strategy_manager.weights['ml_model'] * 0.8)
            print(f"降低ML模型权重至 {self.strategy_manager.weights['ml_model']:.3f}")
    
    def kfold_cross_validation(self, n_splits=5):
        """执行K-fold交叉验证"""
        if self.df.empty or len(self.df) < 100:
            print("数据不足，无法进行K-fold交叉验证")
            return 0.0
        
        print(f"开始K-fold交叉验证 ({n_splits}折)...")
        
        # 确保数据量足够
        if len(self.df) < n_splits * 20:
            n_splits = max(2, len(self.df) // 20)
            print(f"数据量不足，调整为{n_splits}折交叉验证")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        overfit_warnings = 0
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(self.df), 1):
            print(f"\n--- 折 {fold}/{n_splits} ---")
            
            # 分割数据
            train_data = self.df.iloc[train_idx].copy()
            test_data = self.df.iloc[test_idx].copy()
            
            # 为训练数据添加特征
            self.add_features_to_data(train_data)
            
            # 初始化策略管理器
            strategy_manager = StrategyManager()
            strategy_manager.update_combo_probs(train_data)
            strategy_manager.evaluate_factor_validity(train_data)
            
            fold_hits = 0
            fold_total = 0
            
            for i in range(len(test_data) - 1):
                # 获取特征数据
                feature_row = test_data.iloc[i]
                last_zodiac = feature_row['zodiac']
                
                # 预测下一期
                prediction, _ = strategy_manager.generate_prediction(
                    feature_row, last_zodiac
                )
                
                # 实际结果
                actual = test_data.iloc[i+1]['zodiac']
                
                # 检查命中
                if actual in prediction:
                    fold_hits += 1
                fold_total += 1
            
            fold_accuracy = fold_hits / fold_total if fold_total > 0 else 0.0
            accuracies.append(fold_accuracy)
            print(f"折 {fold} 准确率: {fold_accuracy:.2%} ({fold_hits}/{fold_total})")
            
            # 检测过拟合迹象
            train_results = []
            for i in range(len(train_data) - 1):
                feature_row = train_data.iloc[i]
                last_zodiac = feature_row['zodiac']
                
                prediction, _ = strategy_manager.generate_prediction(
                    feature_row, last_zodiac
                )
                
                actual = train_data.iloc[i+1]['zodiac']
                
                if actual in prediction:
                    train_results.append(1)
                else:
                    train_results.append(0)
            
            train_accuracy = np.mean(train_results) if train_results else 0.0
            accuracy_diff = abs(train_accuracy - fold_accuracy)
            
            if accuracy_diff > 0.15:
                print(f"警告: 折 {fold} 可能过拟合 (训练准确率={train_accuracy:.2%}, 测试准确率={fold_accuracy:.2%})")
                overfit_warnings += 1
        
        avg_accuracy = np.mean(accuracies)
        print(f"\nK-fold交叉验证平均准确率: {avg_accuracy:.2%}")
        
        if overfit_warnings > n_splits // 2:
            print(f"严重警告: {overfit_warnings}/{n_splits} 折检测到过拟合迹象")
            # 触发模型重新训练
            print("触发ML模型重新训练以防止过拟合...")
            self.ml_predictor.train_model(self.df, retrain=True)
        
        return avg_accuracy
    
    def _calculate_data_volatility(self):
        """计算数据波动性（基于最近200期生肖频率变化）"""
        if len(self.df) < 200:
            # 数据不足时使用全部数据
            recent = self.df
        else:
            recent = self.df.tail(200)
        
        # 计算每个生肖的出现频率变化
        zodiac_volatilities = []
        for zodiac in self.zodiacs:
            # 获取该生肖的出现序列
            occurrences = (recent['zodiac'] == zodiac).astype(int).values
            
            # 计算变异系数（标准差/均值）
            if np.mean(occurrences) > 0:
                cv = variation(occurrences) if len(occurrences) > 1 else 0
                zodiac_volatilities.append(cv)
        
        # 返回平均波动性
        return np.mean(zodiac_volatilities) if zodiac_volatilities else 0.0
    
    def save_error_analysis(self, error_patterns):
        """保存错误模式分析结果"""
        # 转换错误模式为DataFrame
        error_list = []
        for last_zodiac, patterns in error_patterns.items():
            for actual, count in patterns.items():
                error_list.append({
                    '上期生肖': last_zodiac,
                    '实际出现生肖': actual,
                    '错误次数': count
                })
        
        error_df = pd.DataFrame(error_list)
        
        if not error_df.empty:
            # 按错误次数排序
            error_df = error_df.sort_values('错误次数', ascending=False)
            
            # 保存到文件
            error_file = f"error_analysis/error_patterns_{datetime.now().strftime('%Y%m%d')}.csv"
            error_df.to_csv(error_file, index=False)
            print(f"错误模式分析已保存到: {error_file}")
            
            # 提取最常见错误模式
            common_errors = error_df.head(3).to_dict('records')
            
            # 根据常见错误模式调整预测策略
            self.adjust_based_on_errors(common_errors)
    
    def adjust_based_on_errors(self, common_errors):
        """根据常见错误模式调整预测策略"""
        print("\n根据常见错误模式调整预测策略:")
        for error in common_errors:
            last_zodiac = error['上期生肖']
            actual = error['实际出现生肖']
            count = error['错误次数']
            
            print(f"常见错误模式: 上期是 {last_zodiac} 时, 实际出现 {actual} (发生{count}次)")
            
            # 1. 增加特定转移概率的权重
            if f"{last_zodiac}-{actual}" in self.strategy_manager.combo_probs:
                # 增加这个特定转移的权重
                prob = self.strategy_manager.combo_probs[f"{last_zodiac}-{actual}"]
                print(f"  - 增加转移概率权重: {last_zodiac} -> {actual} (概率={prob:.2%})")
                
                # 在策略管理器中标记这个模式需要更多关注
                self.strategy_manager.special_attention_patterns[f"{last_zodiac}-{actual}"] = {
                    'weight_multiplier': 1.5 + (count * 0.05),  # 错误次数越多，权重倍数越高
                    'last_occurrence': datetime.now(),
                    'error_count': count
                }
            
            # 2. 调整特定生肖的权重
            print(f"  - 增加 {actual} 在类似情况下的预测优先级")
            if actual in self.strategy_manager.zodiac_attention:
                self.strategy_manager.zodiac_attention[actual] += count
            else:
                self.strategy_manager.zodiac_attention[actual] = count
    
    def add_features_to_data(self, df):
        """为数据添加必要的特征 - 完整实现"""
        # 确保 is_festival 列存在
        if 'is_festival' not in df.columns and 'festival' in df.columns:
            df['is_festival'] = (df['festival'] != "无").astype(int)
        
        # 1. 添加生肖频率特征
        zodiac_counts = df['zodiac'].value_counts(normalize=True)
        for zodiac in self.zodiacs:
            df[f'freq_{zodiac}'] = zodiac_counts.get(zodiac, 0.0)
        
        # 2. 添加季节生肖特征
        for zodiac in self.zodiacs:
            # 初始化季节特征列
            df[f'season_{zodiac}'] = 0.0
        
        # 计算每个季节中每个生肖的频率
        for season in ['春', '夏', '秋', '冬']:
            # 使用 .loc 进行显式行选择（关键修复）
            season_mask = (df['season'] == season)
            season_data = df.loc[season_mask]
            if not season_data.empty:
                season_counts = season_data['zodiac'].value_counts(normalize=True)
                for zodiac in self.zodiacs:
                    # 更新季节特征值
                    df.loc[season_mask, f'season_{zodiac}'] = season_counts.get(zodiac, 0.0)
    
    def apply_pattern_enhancement(self, prediction, last_zodiac, target_date, data):
        """应用历史模式增强预测 - 强化节日效应和连续/间隔处理"""
        festival = self.detect_festival(target_date)
        
        # 1. 节日效应增强 - 量化处理
        if festival in self.patterns['festival_boost']:
            boost_zodiac = self.patterns['festival_boost'][festival]
            strength = self.patterns['festival_strength'].get(festival, 0.5)  # 默认强度50%
            
            if boost_zodiac not in prediction:
                # 根据节日效应强度决定替换位置
                if strength > 0.7:  # 强效应（70%以上）
                    # 替换预测中的最后一位
                    prediction = prediction[:-1] + [boost_zodiac]
                    print(f"节日效应增强(强): {festival}节日常见生肖 {boost_zodiac} 加入预测 (强度={strength:.2%})")
                elif strength > 0.5:  # 中等效应（50-70%）
                    # 替换预测中的最后一位，并提升到中间位置
                    prediction = prediction[:-1] + [boost_zodiac]
                    prediction.insert(len(prediction)//2, prediction.pop())
                    print(f"节日效应增强(中): {festival}节日常见生肖 {boost_zodiac} 加入预测 (强度={strength:.2%})")
                else:  # 弱效应（50%以下）
                    # 替换预测中的最后一位
                    prediction = prediction[:-1] + [boost_zodiac]
                    print(f"节日效应增强(弱): {festival}节日常见生肖 {boost_zodiac} 加入预测 (强度={strength:.2%})")
            else:
                # 如果已在预测中，提升其优先级
                current_index = prediction.index(boost_zodiac)
                if current_index > 0:
                    # 根据强度决定提升幅度
                    if strength > 0.7:
                        prediction.insert(0, prediction.pop(current_index))
                        print(f"节日效应提升(强): {boost_zodiac} 提升至首位 (强度={strength:.2%})")
                    elif strength > 0.5:
                        new_index = max(0, current_index - 1)
                        prediction.insert(new_index, prediction.pop(current_index))
                        print(f"节日效应提升(中): {boost_zodiac} 提升一位 (强度={strength:.2%})")
        
        # 2. 间隔模式增强 - 严格干预
        for zodiac in self.zodiacs:
            if zodiac in self.patterns['intervals']:
                interval_data = self.patterns['intervals'][zodiac]
                avg_interval = interval_data['mean']
                std_dev = interval_data['std']
                
                last_idx = data[data['zodiac'] == zodiac].index[-1] if not data.empty else -1
                current_interval = len(data) - last_idx
                
                # 计算与平均间隔的偏差（标准差单位）
                if std_dev > 0:
                    z_score = (current_interval - avg_interval) / std_dev
                else:
                    z_score = 0
                
                # 如果接近或超过平均间隔（考虑标准差）
                if z_score >= -0.5:  # 在平均间隔的0.5个标准差范围内
                    # 强制加入或提升优先级
                    if zodiac not in prediction:
                        # 替换最后一位
                        prediction = prediction[:-1] + [zodiac]
                        print(f"间隔模式强制加入: {zodiac} 已间隔 {current_interval}期 (平均 {avg_interval:.1f}±{std_dev:.1f}期)")
                    else:
                        # 提升到首位
                        prediction.remove(zodiac)
                        prediction.insert(0, zodiac)
                        print(f"间隔模式提升首位: {zodiac} 已间隔 {current_interval}期 (平均 {avg_interval:.1f}±{std_dev:.1f}期)")
        
        # 3. 连续出现模式处理 - 严格干预
        if last_zodiac in self.patterns['consecutive']:
            max_consecutive = self.patterns['consecutive'][last_zodiac]
            current_consecutive = 1
            idx = len(data) - 1
            while idx > 0 and data.iloc[idx]['zodiac'] == last_zodiac:
                current_consecutive += 1
                idx -= 1
            
            # 如果连续出现次数接近历史最大值，强制移出预测
            if current_consecutive >= max(2, max_consecutive * 0.8):
                if last_zodiac in prediction:
                    prediction.remove(last_zodiac)
                    print(f"连续模式强制移出: {last_zodiac} 已连续出现 {current_consecutive}次 (历史最高 {max_consecutive}次)")
        
        # 4. 错误学习增强
        # 检查是否有针对当前上期生肖的特殊关注模式
        special_pattern = f"{last_zodiac}-"
        for pattern, info in self.strategy_manager.special_attention_patterns.items():
            if pattern.startswith(special_pattern):
                zodiac_to_boost = pattern.split('-')[1]
                if zodiac_to_boost not in prediction:
                    # 如果这个生肖不在预测中，替换掉得分最低的生肖
                    prediction = prediction[:-1] + [zodiac_to_boost]
                    print(f"错误学习增强: 根据历史错误模式，增加 {zodiac_to_boost} 的优先级")
                    break
        
        # 5. 生肖关注度增强
        # 增加高关注度生肖的优先级
        for zodiac in prediction.copy():
            if zodiac in self.strategy_manager.zodiac_attention:
                # 如果这个生肖有高关注度，提升优先级
                if zodiac in prediction:
                    current_index = prediction.index(zodiac)
                    new_index = max(0, current_index - 1)
                    prediction.insert(new_index, prediction.pop(current_index))
                    print(f"关注度增强: {zodiac} 有高关注度，提升优先级")
        
        return prediction
    
    def predict_next(self):
        """预测下期生肖（使用自适应策略）"""
        if self.df.empty:
            print("无法预测 - 数据为空")
            return {
                'next_number': "未知",
                'prediction': ["无数据"],
                'last_zodiac': "无数据"
            }
        
        # 获取最新数据
        latest = self.df.iloc[-1]
        last_zodiac = latest['zodiac']
        print(f"开始预测下期: 最新生肖={last_zodiac}")
        
        # 获取策略报告
        strategy_report = self.strategy_manager.generate_factor_report()
        print(strategy_report)
        
        # 预测目标日期（下一天）
        target_date = latest['date'] + timedelta(days=1)
        
        # 基于最近200期数据
        if len(self.df) < BACKTEST_WINDOW:
            print(f"数据不足{BACKTEST_WINDOW}期，使用全部数据进行预测")
            recent = self.df
        else:
            print(f"使用最近{BACKTEST_WINDOW}期数据预测")
            recent = self.df.tail(BACKTEST_WINDOW)
        
        # 获取特征数据（确保包含所有必要特征）
        feature_row = recent.iloc[-1]
        
        # 使用策略管理器生成预测
        prediction, factor_predictions = self.strategy_manager.generate_prediction(
            feature_row, last_zodiac
        )
        
        # 应用模式增强
        prediction = self.apply_pattern_enhancement(
            prediction, last_zodiac, target_date, recent
        )
        
        # 确保预测结果4-5个
        prediction = prediction[:5]
        
        # 下期期号
        try:
            last_expect = latest['expect']
            if re.match(r'^\d+$', last_expect):
                next_num = str(int(last_expect) + 1)
            else:
                next_num = "未知"
        except:
            next_num = "未知"
        
        print(f"预测结果: 下期期号={next_num}, 推荐生肖={', '.join(prediction)}")
        print("各因子预测详情:")
        for factor, preds in factor_predictions.items():
            print(f"- {factor}: {', '.join(preds)}")
        
        return {
            'next_number': next_num,
            'prediction': prediction,
            'last_zodiac': last_zodiac,
            'factor_predictions': factor_predictions
        }
    
    def analyze_history(self):
        """历史预测复盘分析 - 新增核心功能"""
        print("开始历史预测复盘分析...")
        
        # 检查预测记录文件是否存在
        prediction_file = 'data/predictions.csv'
        if not os.path.exists(prediction_file):
            print("警告: 未找到预测记录文件，将创建新文件")
            self.initialize_prediction_file()
        
        # 加载预测记录
        try:
            records = pd.read_csv(prediction_file)
            if records.empty:
                print("预测记录为空，无法分析")
                return {}
        except Exception as e:
            print(f"加载预测记录失败: {e}")
            return {}
        
        # 提取错误记录
        errors = records[records['是否命中'] == 0]
        total_errors = len(errors)
        
        if total_errors == 0:
            print("未发现预测错误记录")
            return {}
        
        print(f"发现 {total_errors} 条错误预测记录")
        
        # 分析错误模式
        pattern_counts = errors.groupby(['上期生肖', '实际生肖']).size().reset_index(name='错误次数')
        top_patterns = pattern_counts.sort_values('错误次数', ascending=False).head(3)
        
        # 生成调整建议
        adjustments = []
        for idx, row in top_patterns.iterrows():
            last_zodiac = row['上期生肖']
            actual = row['实际生肖']
            count = row['错误次数']
            
            adjustments.append({
                'pattern': f"{last_zodiac}-{actual}",
                'weight_adjustment': min(1.5, 1 + count / 10)  # 动态权重调整
            })
        
        print("TOP 3 高频错误模式:")
        for adj in adjustments:
            print(f"- {adj['pattern']}: 权重调整系数={adj['weight_adjustment']:.2f}")
        
        # 应用调整建议
        self.strategy_manager.apply_review_results(adjustments)
        
        return top_patterns.to_dict('records')
    
    def generate_report(self):
        """生成符合要求的分析报告 - 包含动态回测信息"""
        if self.df.empty:
            report = "===== 彩票分析报告 =====\n错误：没有获取到有效数据，请检查API"
            print(report)
            return report
        
        # 获取最新开奖信息
        latest = self.df.iloc[-1]
        last_expect = latest['expect']
        last_zodiac = latest['zodiac']
        last_date = latest['date'].strftime('%Y-%m-%d')
        
        # 执行历史复盘分析
        history_review = self.analyze_history()
        
        # 生肖分析
        analysis = self.analyze_zodiac_patterns()
        
        # 回测结果
        backtest_df, accuracy = self.backtest_strategy()
        
        # 执行K-fold交叉验证
        kfold_accuracy = self.kfold_cross_validation()
        
        # 预测下期
        prediction = self.predict_next()
        
        # 因子表现报告
        factor_report = self.strategy_manager.generate_factor_report()
        
        # 针对最新生肖的转移分析
        transition_details = {}
        if 'transition_matrix' in analysis:
            transition_matrix = analysis['transition_matrix']
            if last_zodiac in transition_matrix.columns:
                next_zodiacs = transition_matrix[last_zodiac].nlargest(4)
                transition_details[last_zodiac] = [
                    f"{zodiac}({prob:.1f}%)" for zodiac, prob in next_zodiacs.items()
                ]
        
        # 生成详细报告
        report = f"""
        ===== 每日彩报 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        
        数据统计：
        - 总期数：{len(self.df)}
        - 数据范围：{self.df['date'].min().date()} 至 {last_date}
        - 最新期号：{last_expect}
        - 最新开奖生肖：{last_zodiac}
        
        历史模式：
        - 最长连续出现: {', '.join([f'{z}({c})' for z, c in self.patterns.get('consecutive', {}).items()])}
        - 节日效应: {', '.join([f'{f}→{z}(强度:{self.patterns["festival_strength"].get(f,0):.1%})' for f, z in self.patterns.get('festival_boost', {}).items()])}
        
        生肖频率分析（全部历史数据）：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if 'frequency' in analysis else "无数据"}
        
        最新开奖生肖分析：
        - 生肖: {last_zodiac}
        - 出现后下期最可能出现的4个生肖: {", ".join(transition_details.get(last_zodiac, ["无数据"]))}
        
        回测结果：
        - 主回测准确率：{accuracy:.2%}
        - K-fold交叉验证准确率：{kfold_accuracy:.2%}
        - 综合准确率：{(accuracy * 0.7 + kfold_accuracy * 0.3):.2%}
        
        历史复盘分析：
        {self.format_history_review(history_review)}
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        - 预测依据：综合历史模式与多因素分析
        
        === 多因子预测详情 ===
        """
        
        # 添加因子预测详情
        if 'factor_predictions' in prediction:
            for factor, preds in prediction['factor_predictions'].items():
                report += f"- {factor}: {', '.join(preds)}\n"
        
        # 添加因子表现报告
        report += f"\n{factor_report}\n"
        report += "============================================="
        
        print("分析报告生成完成")
        return report
    
    def format_history_review(self, review_data):
        """格式化历史复盘结果"""
        if not review_data:
            return "未发现显著错误模式"
        
        formatted = []
        for item in review_data:
            last_zodiac = item['上期生肖']
            actual = item['实际生肖']
            count = item['错误次数']
            formatted.append(f"- {last_zodiac}→{actual}: {count}次错误")
        
        return "\n".join(formatted)

# 测试函数
if __name__ == "__main__":
    print("===== 测试彩票分析器 =====")
    
    # 创建测试分析器
    analyzer = LotteryAnalyzer()
    
    # 测试预测记录初始化
    print("\n测试预测记录文件初始化:")
    analyzer.initialize_prediction_file()
    
    # 测试保存预测记录
    print("\n测试保存预测记录:")
    analyzer.save_prediction_record(
        expect="2023001",
        date="2023-01-01",
        last_zodiac="鼠",
        predicted_zodiacs=["牛", "虎", "兔"],
        actual_zodiac="龙",
        is_hit=0
    )
    
    # 测试历史复盘分析
    print("\n测试历史复盘分析:")
    review = analyzer.analyze_history()
    print(f"复盘结果: {review}")
    
    # 测试报告生成
    print("\n测试报告生成:")
    report = analyzer.generate_report()
    print(report[:500] + "...")  # 打印部分报告
    
    print("\n===== 测试完成 =====")
