import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping, log_error
from config import BACKTEST_WINDOW, ACCURACY_THRESHOLD, ML_MODEL_PATH
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar
from collections import defaultdict
from src.strategy_manager import StrategyManager
import warnings
import os
from sklearn.model_selection import TimeSeriesSplit, KFold
from scipy.stats import variation
from src.ml_predictor import MLPredictor
import logging
import joblib
from typing import Dict, List, Tuple, Optional, Union

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(ML_MODEL_PATH, 'analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LotteryAnalyzer')

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

class LotteryAnalyzer:
    def __init__(self):
        """初始化彩票分析系统"""
        logger.info("Initializing Lottery Analyzer")
        self.df = self._load_and_preprocess_data()
        self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        self.strategy_manager = StrategyManager()
        self.ml_predictor = MLPredictor()
        self.performance_metrics = self._init_performance_metrics()
        self.patterns = self.detect_patterns() if not self.df.empty else {}

    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """加载并预处理原始数据"""
        logger.info("Loading historical data...")
        df = fetch_historical_data()
        if df.empty:
            logger.warning("No valid data fetched")
            return df

        # 数据清洗
        df = self._clean_data(df)
        
        # 添加基础特征
        df = self._add_basic_features(df)
        
        # 添加高级特征
        df = self._add_advanced_features(df)
        
        logger.info(f"Data loaded successfully. Total records: {len(df)}")
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据清洗"""
        # 生肖映射
        df['zodiac'] = df.apply(
            lambda row: zodiac_mapping(row['special'], row['year']), axis=1
        )
        
        # 过滤无效生肖
        valid_mask = df['zodiac'].isin(self.zodiacs)
        if not valid_mask.all():
            invalid_count = len(df) - valid_mask.sum()
            logger.warning(f"Found {invalid_count} invalid zodiac records")
            df = df[valid_mask].copy()
        
        # 日期处理
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加基础时间特征"""
        # 农历日期
        df['lunar'] = df['date'].apply(self._convert_to_lunar)
        
        # 节日标记
        df['festival'] = df['date'].apply(self._detect_festival)
        df['is_festival'] = df['festival'] != "无"
        
        # 季节
        df['season'] = df['date'].apply(self._determine_season)
        
        # 时间特征
        df['weekday'] = df['date'].dt.weekday
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加高级分析特征"""
        # 生肖特征
        df = self._add_zodiac_features(df)
        
        # 滚动窗口特征
        df = self._add_rolling_features(df)
        
        return df

    def _convert_to_lunar(self, dt: datetime) -> Lunar:
        """公历转农历（带异常处理）"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            if not hasattr(lunar, 'year'):  # 验证转换结果
                raise ValueError("Invalid lunar date conversion")
            return lunar
        except Exception as e:
            logger.error(f"Lunar conversion failed for {dt}: {str(e)}")
            # 返回标记值
            invalid_lunar = Lunar(0, 0, 0, False)
            invalid_lunar.year = 0  # 确保有year属性
            return invalid_lunar

    def _detect_festival(self, dt: datetime) -> str:
        """检测传统节日（完整实现）"""
        lunar = self._convert_to_lunar(dt)
        if not hasattr(lunar, 'month') or lunar.year == 0:
            return "无"

        # 农历节日（月份, 日）：节日名称
        lunar_festivals = {
            (1, 1): "春节",
            (1, 15): "元宵",
            (5, 5): "端午",
            (7, 7): "七夕",
            (7, 15): "中元",
            (8, 15): "中秋",
            (9, 9): "重阳",
            (12, 8): "腊八"
        }

        # 公历节日（月份, 日）：节日名称
        solar_festivals = {
            (1, 1): "元旦",
            (2, 14): "情人节",
            (3, 8): "妇女节",
            (4, 5): "清明",
            (5, 1): "劳动节",
            (6, 1): "儿童节",
            (9, 10): "教师节",
            (10, 1): "国庆节",
            (12, 25): "圣诞节"
        }

        # 特殊节日规则
        def is_new_years_eve(l: Lunar) -> bool:
            """判断是否为除夕"""
            if l.month != 12:
                return False
            # 农历12月29（非闰月）或30
            return (l.day == 29 and not l.isleap) or l.day == 30

        # 1. 检查除夕（最高优先级）
        if is_new_years_eve(lunar):
            return "除夕"

        # 2. 检查农历节日
        lunar_key = (lunar.month, lunar.day)
        if lunar_key in lunar_festivals:
            return lunar_festivals[lunar_key]

        # 3. 检查公历节日
        solar_key = (dt.month, dt.day)
        if solar_key in solar_festivals:
            return solar_festivals[solar_key]

        return "无"

    def _determine_season(self, dt: datetime) -> str:
        """精确季节判断（基于节气近似）"""
        month = dt.month
        day = dt.day
        
        # 春季：3月20日(春分) ~ 6月21日(夏至)
        if (month == 3 and day >= 20) or (4 <= month <= 5) or (month == 6 and day < 21):
            return "春"
        # 夏季：6月21日 ~ 9月23日(秋分)
        elif (month == 6 and day >= 21) or (7 <= month <= 8) or (month == 9 and day < 23):
            return "夏"
        # 秋季：9月23日 ~ 12月21日(冬至)
        elif (month == 9 and day >= 23) or (10 <= month <= 11) or (month == 12 and day < 21):
            return "秋"
        # 冬季：12月21日 ~ 3月20日
        else:
            return "冬"

    def _add_zodiac_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的生肖特征工程"""
        # 1. 全局频率特征
        total_counts = df['zodiac'].value_counts(normalize=True)
        for zodiac in self.zodiacs:
            df[f'global_freq_{zodiac}'] = total_counts.get(zodiac, 0)
        
        # 2. 移动窗口频率（近期100期）
        for zodiac in self.zodiacs:
            df[f'recent_100_{zodiac}'] = (df['zodiac'] == zodiac).rolling(100, min_periods=1).mean()
        
        # 3. 季节特征
        season_groups = df.groupby('season')['zodiac'].value_counts(normalize=True).unstack()
        for season in ['春', '夏', '秋', '冬']:
            if season in season_groups:
                for zodiac in self.zodiacs:
                    col_name = f'season_{season}_{zodiac}'
                    df[col_name] = season_groups.loc[season].get(zodiac, 0)
        
        # 4. 节日特征
        festival_data = df[df['is_festival']]
        if not festival_data.empty:
            festival_counts = festival_data.groupby('festival')['zodiac'].value_counts(normalize=True).unstack()
            for festival in festival_counts.index:
                for zodiac in self.zodiacs:
                    col_name = f'festival_{festival}_{zodiac}'
                    df[col_name] = festival_counts.loc[festival].get(zodiac, 0)
        
        logger.info(f"Added {len(self.zodiacs)*4} zodiac features")
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """完整的滚动特征工程"""
        windows = [3, 7, 15, 30, 60, 100]  # 多种窗口大小
        
        # 1. 基础出现标记
        for zodiac in self.zodiacs:
            df[f'occur_{zodiac}'] = (df['zodiac'] == zodiac).astype(int)
        
        # 2. 滚动窗口特征
        for window in windows:
            for zodiac in self.zodiacs:
                # 滚动频率
                df[f'rolling_{window}_freq_{zodiac}'] = (
                    df[f'occur_{zodiac}'].rolling(window=window, min_periods=1).mean()
                )
                # 滚动标准差
                df[f'rolling_{window}_std_{zodiac}'] = (
                    df[f'occur_{zodiac}'].rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        # 3. 特殊滚动特征
        for zodiac in self.zodiacs:
            # 近期是否出现过
            df[f'recent_5_occur_{zodiac}'] = (
                df[f'occur_{zodiac}'].rolling(5, min_periods=1).max()
            )
            # 最大连续出现
            df[f'max_streak_{zodiac}'] = (
                df[f'occur_{zodiac}'] * 
                (df[f'occur_{zodiac}'].groupby(
                    (df[f'occur_{zodiac}'] != df[f'occur_{zodiac}'].shift()).cumsum()
                ).cumcount() + 1)
            )
        
        logger.info(f"Added {len(windows)*len(self.zodiacs)*2 + len(self.zodiacs)*2} rolling features")
        return df

    def detect_patterns(self) -> Dict[str, Dict]:
        """完整的历史模式检测"""
        patterns = {
            'consecutive': defaultdict(int),
            'intervals': defaultdict(dict),
            'festival_boost': defaultdict(dict),
            'transition_matrix': pd.DataFrame(),
            'strength_metrics': {
                'consecutive': defaultdict(float),
                'interval': defaultdict(float),
                'festival': defaultdict(float)
            }
        }
        
        if self.df.empty:
            return patterns

        # 1. 连续出现模式
        current_streak = 1
        current_zodiac = self.df.iloc[0]['zodiac']
        
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['zodiac'] == current_zodiac:
                current_streak += 1
            else:
                if current_streak > 1:
                    patterns['consecutive'][current_zodiac] = max(
                        patterns['consecutive'][current_zodiac],
                        current_streak
                    )
                    patterns['strength_metrics']['consecutive'][current_zodiac] += current_streak
                current_streak = 1
                current_zodiac = self.df.iloc[i]['zodiac']
        
        # 标准化连续强度
        total = len(self.df)
        for zodiac in patterns['strength_metrics']['consecutive']:
            patterns['strength_metrics']['consecutive'][zodiac] /= total

        # 2. 间隔模式
        last_occurrence = {zodiac: -1 for zodiac in self.zodiacs}
        interval_records = {zodiac: [] for zodiac in self.zodiacs}
        
        for idx, row in self.df.iterrows():
            zodiac = row['zodiac']
            if last_occurrence[zodiac] != -1:
                interval = idx - last_occurrence[zodiac]
                interval_records[zodiac].append(interval)
            last_occurrence[zodiac] = idx
        
        for zodiac in self.zodiacs:
            if interval_records[zodiac]:
                intervals = interval_records[zodiac]
                patterns['intervals'][zodiac] = {
                    'mean': np.mean(intervals),
                    'median': np.median(intervals),
                    'std': np.std(intervals),
                    'min': min(intervals),
                    'max': max(intervals),
                    'count': len(intervals)
                }
                # 计算间隔稳定性指标
                if np.mean(intervals) > 0:
                    patterns['strength_metrics']['interval'][zodiac] = (
                        np.mean(intervals) / np.std(intervals) if np.std(intervals) > 0 else 10
                else:
                    patterns['strength_metrics']['interval'][zodiac] = 0

        # 3. 节日效应
        festival_data = self.df[self.df['is_festival']]
        if not festival_data.empty:
            for festival, group in festival_data.groupby('festival'):
                zodiac_counts = group['zodiac'].value_counts(normalize=True)
                if not zodiac_counts.empty:
                    top_zodiac = zodiac_counts.idxmax()
                    prob = zodiac_counts.max()
                    patterns['festival_boost'][festival] = {
                        'zodiac': top_zodiac,
                        'probability': prob,
                        'occurrences': zodiac_counts[top_zodiac]
                    }
                    patterns['strength_metrics']['festival'][festival] = prob

        # 4. 转移矩阵
        transition_counts = pd.crosstab(
            self.df['zodiac'].shift(-1),
            self.df['zodiac'],
            dropna=False
        ).fillna(0)
        
        if not transition_counts.empty:
            patterns['transition_matrix'] = (
                transition_counts / transition_counts.sum()
            ).round(4)

        logger.info("Completed pattern detection with:")
        logger.info(f"- {len(patterns['consecutive'])} consecutive patterns")
        logger.info(f"- {len(patterns['intervals'])} interval patterns")
        logger.info(f"- {len(patterns['festival_boost'])} festival effects")
        
        return patterns

    def _init_performance_metrics(self) -> Dict:
        """初始化性能监控指标"""
        return {
            'accuracy_history': [],
            'feature_stability': 0.0,
            'model_drift': 0.0,
            'last_cv_scores': [],
            'error_rates': defaultdict(list)
        }

    def backtest_strategy(self) -> Tuple[pd.DataFrame, float]:
        """完整的策略回测实现"""
        if len(self.df) < BACKTEST_WINDOW + 10:
            logger.warning("Insufficient data for backtesting")
            return pd.DataFrame(), 0.0

        # 动态确定窗口大小
        window_size = self._calculate_dynamic_window()
        logger.info(f"Starting backtest with window size: {window_size}")

        results = []
        error_patterns = defaultdict(int)
        
        for i in range(window_size, len(self.df) - 1):
            train = self.df.iloc[i-window_size:i]
            test = self.df.iloc[i:i+1]
            
            # 更新策略
            self.strategy_manager.update_combo_probs(train)
            
            # 生成预测
            feature_row = train.iloc[-1]
            prediction, _ = self.strategy_manager.generate_prediction(
                feature_row, 
                feature_row['zodiac']
            )
            actual = test.iloc[0]['zodiac']
            
            # 记录结果
            is_correct = int(actual in prediction)
            results.append({
                'period': test['expect'].values[0],
                'date': test['date'].values[0],
                'actual': actual,
                'prediction': ",".join(prediction),
                'is_correct': is_correct,
                'window_size': window_size
            })
            
            # 记录错误模式
            if not is_correct:
                error_key = f"{feature_row['zodiac']}->{actual}"
                error_patterns[error_key] += 1
                self.performance_metrics['error_rates'][error_key].append(1)
            else:
                for key in self.performance_metrics['error_rates']:
                    self.performance_metrics['error_rates'][key].append(0)
            
            # 动态调整窗口
            if i % 50 == 0:
                window_size = self._calculate_dynamic_window()

        # 转换为DataFrame
        result_df = pd.DataFrame(results)
        
        # 计算准确率
        accuracy = result_df['is_correct'].mean()
        self.performance_metrics['accuracy_history'].append(accuracy)
        
        # 交叉验证
        cv_accuracy = self.cross_validate()
        
        # 综合评估
        combined_accuracy = self._evaluate_results(accuracy, cv_accuracy, error_patterns)
        
        logger.info(f"Backtest completed. Final accuracy: {combined_accuracy:.2%}")
        return result_df, combined_accuracy

    def _calculate_dynamic_window(self) -> int:
        """动态计算回测窗口大小"""
        volatility = self._calculate_data_volatility()
        base_window = BACKTEST_WINDOW
        
        if volatility > 0.6:  # 高波动性
            return max(50, int(base_window * 0.6))
        elif volatility > 0.4:  # 中波动性
            return base_window
        else:  # 低波动性
            return min(300, int(base_window * 1.5))

    def _calculate_data_volatility(self) -> float:
        """计算数据波动性指标"""
        recent = self.df.tail(200) if len(self.df) >= 200 else self.df
        volatilities = []
        
        for zodiac in self.zodiacs:
            occurrences = (recent['zodiac'] == zodiac).astype(int).values
            if np.mean(occurrences) > 0:
                cv = variation(occurrences) if len(occurrences) > 1 else 0
                volatilities.append(cv)
        
        return np.mean(volatilities) if volatilities else 0.5

    def cross_validate(self, n_splits: int = 5) -> float:
        """完整的时间序列交叉验证"""
        if len(self.df) < 100:
            logger.warning("Insufficient data for cross validation")
            return 0.0
        
        # 动态确定折数
        n_splits = min(n_splits, len(self.df) // 20)
        n_splits = max(2, n_splits)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"Starting {n_splits}-fold time series cross validation")
        accuracies = []
        overfit_flags = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.df), 1):
            train_data = self.df.iloc[train_idx]
            test_data = self.df.iloc[test_idx]
            
            # 使用副本避免污染原始策略管理器
            temp_manager = StrategyManager()
            temp_manager.update_combo_probs(train_data)
            
            # 验证该折数据
            fold_acc, is_overfit = self._validate_fold(temp_manager, train_data, test_data)
            accuracies.append(fold_acc)
            overfit_flags.append(is_overfit)
            
            logger.info(f"Fold {fold}/{n_splits} - Accuracy: {fold_acc:.2%} | Overfit: {is_overfit}")
        
        avg_accuracy = np.mean(accuracies)
        self.performance_metrics['last_cv_scores'] = accuracies
        
        # 过拟合处理
        if sum(overfit_flags) > n_splits // 2:
            self._handle_overfitting()
        
        logger.info(f"Cross validation completed. Average accuracy: {avg_accuracy:.2%}")
        return avg_accuracy

    def _validate_fold(self, manager: StrategyManager, 
                     train_data: pd.DataFrame, 
                     test_data: pd.DataFrame) -> Tuple[float, bool]:
        """验证单个数据折"""
        test_correct = 0
        train_correct = 0
        total_test = max(1, len(test_data) - 1)
        
        for i in range(total_test):
            # 测试集验证
            feature_row = test_data.iloc[i]
            actual = test_data.iloc[i+1]['zodiac']
            prediction, _ = manager.generate_prediction(feature_row, feature_row['zodiac'])
            test_correct += int(actual in prediction)
            
            # 训练集验证（用于过拟合检测）
            train_row = train_data.iloc[i % len(train_data)]
            train_actual = train_data.iloc[(i+1) % len(train_data)]['zodiac']
            train_pred, _ = manager.generate_prediction(train_row, train_row['zodiac'])
            train_correct += int(train_actual in train_pred)
        
        test_acc = test_correct / total_test
        train_acc = train_correct / total_test
        is_overfit = (train_acc - test_acc) > 0.15  # 差异超过15%视为过拟合
        
        return test_acc, is_overfit

    def _handle_overfitting(self):
        """处理过拟合情况"""
        logger.warning("Handling potential overfitting...")
        
        # 1. 降低ML模型权重
        old_weight = self.strategy_manager.weights.get('ml_model', 0)
        new_weight = max(0.1, old_weight * 0.7)
        self.strategy_manager.weights['ml_model'] = new_weight
        logger.info(f"Reduced ML model weight from {old_weight:.2f} to {new_weight:.2f}")
        
        # 2. 重新训练ML模型
        self.ml_predictor.train_model(self.df, retrain=True)
        
        # 3. 增加正则化
        self.strategy_manager.adjust(ACCURACY_THRESHOLD * 0.9, {})

    def _evaluate_results(self, 
                         base_accuracy: float, 
                         cv_accuracy: float, 
                         error_patterns: Dict) -> float:
        """综合评估回测结果"""
        # 加权平均（主回测70% + 交叉验证30%）
        combined_accuracy = 0.7 * base_accuracy + 0.3 * cv_accuracy
        
        # 如果差异过大则保守处理
        if abs(base_accuracy - cv_accuracy) > 0.2:
            logger.warning(f"Large discrepancy: base={base_accuracy:.2%} vs cv={cv_accuracy:.2%}")
            combined_accuracy = min(base_accuracy, cv_accuracy)
        
        # 更新策略权重
        self.strategy_manager.adjust(combined_accuracy, error_patterns)
        
        # 更新监控指标
        self._update_performance_metrics(combined_accuracy)
        
        return combined_accuracy

    def _update_performance_metrics(self, accuracy: float):
        """更新性能监控指标"""
        self.performance_metrics['accuracy_history'].append(accuracy)
        if len(self.performance_metrics['accuracy_history']) > 10:
            self.performance_metrics['accuracy_history'].pop(0)
        
        # 计算特征稳定性
        if hasattr(self.strategy_manager, 'feature_importance'):
            fi = self.strategy_manager.feature_importance
            self.performance_metrics['feature_stability'] = (
                fi.std() / fi.mean() if fi.mean() > 0 else 0
            )
        
        # 计算模型漂移
        if len(self.performance_metrics['last_cv_scores']) >= 2:
            self.performance_metrics['model_drift'] = abs(
                self.performance_metrics['last_cv_scores'][-1] - 
                self.performance_metrics['last_cv_scores'][-2]
            )

    def predict_next(self) -> Dict[str, Union[str, List[str], float]]:
        """生成下期预测（完整实现）"""
        if self.df.empty:
            logger.warning("No data available for prediction")
            return {
                'next_period': '未知',
                'prediction': [],
                'confidence': 0.0,
                'factors': {}
            }
        
        latest = self.df.iloc[-1]
        feature_row = self.df.tail(BACKTEST_WINDOW).iloc[-1] if len(self.df) >= BACKTEST_WINDOW else latest
        
        # 生成基础预测
        base_prediction, factor_predictions = self.strategy_manager.generate_prediction(
            feature_row,
            latest['zodiac']
        )
        
        # 应用模式增强
        enhanced_prediction = self._apply_pattern_enhancement(
            base_prediction.copy(),
            latest['zodiac'],
            latest['date'] + timedelta(days=1)
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(enhanced_prediction, factor_predictions)
        
        # 生成下期期号
        next_period = self._generate_next_period_number(latest['expect'])
        
        return {
            'next_period': next_period,
            'prediction': enhanced_prediction[:3],  # 返回top3预测
            'confidence': confidence,
            'factors': factor_predictions
        }

    def _apply_pattern_enhancement(self, 
                                  prediction: List[str], 
                                  last_zodiac: str, 
                                  target_date: datetime) -> List[str]:
        """应用历史模式增强预测"""
        if not prediction:
            return prediction
        
        # 1. 节日效应增强
        festival = self._detect_festival(target_date)
        if festival in self.patterns['festival_boost']:
            boost_info = self.patterns['festival_boost'][festival]
            boost_zodiac = boost_info['zodiac']
            prob = boost_info['probability']
            
            if boost_zodiac not in prediction:
                # 根据概率决定替换位置
                replace_pos = -1 if prob < 0.7 else -2
                prediction[replace_pos] = boost_zodiac
                logger.info(f"Applied festival boost: {festival}->{boost_zodiac} (prob={prob:.2%})")
            else:
                # 提升优先级
                current_pos = prediction.index(boost_zodiac)
                new_pos = max(0, current_pos - 1) if prob > 0.6 else current_pos
                prediction.insert(new_pos, prediction.pop(current_pos))
        
        # 2. 间隔模式增强
        for zodiac, stats in self.patterns['intervals'].items():
            if zodiac not in prediction and stats.get('mean', 0) > 0:
                last_idx = self.df[self.df['zodiac'] == zodiac].index[-1]
                current_interval = len(self.df) - last_idx
                
                # 如果当前间隔接近历史平均值
                if current_interval >= stats['mean'] - stats['std']:
                    prediction[-1] = zodiac
                    logger.info(f"Applied interval pattern: {zodiac} (interval={current_interval})")
                    break
        
        # 3. 连续出现抑制
        if last_zodiac in self.patterns['consecutive']:
            current_streak = 1
            idx = len(self.df) - 2
            while idx >= 0 and self.df.iloc[idx]['zodiac'] == last_zodiac:
                current_streak += 1
                idx -= 1
            
            max_streak = self.patterns['consecutive'][last_zodiac]
            if current_streak >= max(2, max_streak * 0.8) and last_zodiac in prediction:
                prediction.remove(last_zodiac)
                logger.info(f"Suppressed consecutive zodiac: {last_zodiac} (streak={current_streak})")
        
        return prediction

    def _calculate_confidence(self, 
                             prediction: List[str], 
                             factors: Dict[str, List[str]]) -> float:
        """计算预测置信度"""
        if not prediction:
            return 0.0
        
        # 1. 因子一致性
        consensus = sum(
            1 for factor_pred in factors.values() 
            if prediction[0] in factor_pred[:2]
        ) / max(1, len(factors))
        
        # 2. 历史准确率
        hist_acc = np.mean(self.performance_metrics['accuracy_history']) \
            if self.performance_metrics['accuracy_history'] else 0.5
        
        # 3. 模式强度
        pattern_strength = 0.0
        if hasattr(self, 'patterns'):
            # 检查是否有强模式支持
            for zodiac in prediction[:2]:
                # 连续模式强度
                pattern_strength += self.patterns['strength_metrics']['consecutive'].get(zodiac, 0)
                # 间隔模式强度
                pattern_strength += self.patterns['strength_metrics']['interval'].get(zodiac, 0)
            pattern_strength = min(1.0, pattern_strength / 2)  # 归一化
        
        # 综合置信度
        confidence = min(0.99, 
            0.5 * consensus + 
            0.3 * hist_acc + 
            0.2 * pattern_strength
        )
        
        return confidence

    def _generate_next_period_number(self, current: str) -> str:
        """生成下期期号（支持多种格式）"""
        try:
            # 处理纯数字
            if current.isdigit():
                return str(int(current) + 1)
            
            # 处理带字母的期号（如"123A"）
            match = re.match(r"^(\d+)([A-Za-z]*)$", current)
            if match:
                num_part = match.group(1)
                suffix = match.group(2)
                
                if not suffix:  # 无后缀
                    return str(int(num_part) + 1)
                elif len(suffix) == 1:  # 单字母后缀
                    if suffix.isupper() and suffix < 'Z':
                        return f"{num_part}{chr(ord(suffix)+1)}"
                    elif suffix.islower() and suffix < 'z':
                        return f"{num_part}{chr(ord(suffix)+1)}"
                    else:
                        return f"{int(num_part)+1}A"
                else:  # 多字母后缀
                    return f"{int(num_part)+1}{suffix}"
            
            # 处理其他格式
            return f"{current}+1"
        except:
            return "未知"

    def generate_report(self) -> str:
        """生成完整分析报告"""
        if self.df.empty:
            return "===== 分析报告 =====\n错误：无有效数据可用"
        
        # 获取基础信息
        latest = self.df.iloc[-1]
        zodiac_dist = self.df['zodiac'].value_counts(normalize=True).sort_values(ascending=False)
        
        # 获取回测结果
        backtest_results, accuracy = self.backtest_strategy()
        
        # 获取预测结果
        prediction = self.predict_next()
        
        # 生成报告内容
        report_lines = [
            "===== 彩票分析报告 =====",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n[数据概览]",
            f"- 总期数: {len(self.df)}",
            f"- 数据范围: {self.df['date'].min().date()} 至 {self.df['date'].max().date()}",
            f"- 最新开奖: 期号 {latest['expect']} | 生肖 {latest['zodiac']} | 日期 {latest['date'].date()}",
            
            "\n[生肖分布]",
            zodiac_dist.to_string(),
            
            "\n[历史模式]",
            f"- 最长连续出现: {', '.join(f'{z}({c}次)' for z,c in self.patterns['consecutive'].items())}",
            f"- 节日效应: {', '.join(f'{f}→{d['zodiac']}({d['probability']:.0%})' for f,d in self.patterns['festival_boost'].items())}",
            
            "\n[模型表现]",
            f"- 回测准确率: {accuracy:.2%}",
            f"- 近期平均准确率: {np.mean(self.performance_metrics['accuracy_history']):.2%}",
            f"- 特征稳定性: {self.performance_metrics['feature_stability']:.3f}",
            f"- 模型漂移指数: {self.performance_metrics['model_drift']:.3f}",
            
            "\n[下期预测]",
            f"- 预测期号: {prediction['next_period']}",
            f"- 推荐生肖: {', '.join(prediction['prediction'])}",
            f"- 预测置信度: {prediction['confidence']:.0%}",
            
            "\n[预测因子详情]"
        ]
        
        # 添加因子详情
        for factor, preds in prediction['factors'].items():
            report_lines.append(f"- {factor}: {', '.join(preds)}")
        
        # 添加错误模式
        if self.performance_metrics['error_rates']:
            report_lines.append("\n[常见错误模式]")
            for pattern, rates in sorted(
                self.performance_metrics['error_rates'].items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]:
                error_rate = np.mean(rates) if rates else 0
                report_lines.append(f"- {pattern}: {error_rate:.1%}")
        
        report_lines.append("\n===== 报告结束 =====")
        
        return "\n".join(report_lines)

# 示例用法
if __name__ == "__main__":
    analyzer = LotteryAnalyzer()
    print(analyzer.generate_report())
