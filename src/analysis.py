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
            return Converter.Solar2Lunar(solar)
        except Exception as e:
            logger.error(f"Lunar conversion failed for {dt}: {str(e)}")
            return Lunar(0, 0, 0, False)

    def _detect_festival(self, dt: datetime) -> str:
        """检测传统节日"""
        lunar = self._convert_to_lunar(dt)
        if not hasattr(lunar, 'month'):
            return "无"

        # 农历节日
        lunar_festivals = {
            (1, 1): "春节",
            (1, 15): "元宵",
            (5, 5): "端午",
            (8, 15): "中秋"
        }
        
        # 公历节日
        solar_festivals = {
            (1, 1): "元旦",
            (4, 5): "清明",
            (10, 1): "国庆"
        }

        # 检查农历节日
        lunar_key = (lunar.month, lunar.day)
        if lunar_key in lunar_festivals:
            return lunar_festivals[lunar_key]

        # 检查公历节日
        solar_key = (dt.month, dt.day)
        if solar_key in solar_festivals:
            return solar_festivals[solar_key]

        # 特殊处理除夕
        if (lunar.month == 12 and lunar.day == 30) or \
           (lunar.month == 12 and lunar.day == 29 and not lunar.isleap):
            return "除夕"

        return "无"

    def _determine_season(self, dt: datetime) -> str:
        """确定季节（精确到日）"""
        month_day = (dt.month, dt.day)
        
        # 基于节气近似划分
        if (3, 20) <= month_day < (6, 21):
            return "春"
        elif (6, 21) <= month_day < (9, 23):
            return "夏"
        elif (9, 23) <= month_day < (12, 21):
            return "秋"
        else:
            return "冬"

    def _add_zodiac_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加生肖相关特征"""
        # 全局频率
        zodiac_counts = df['zodiac'].value_counts(normalize=True)
        for zodiac in self.zodiacs:
            df[f'global_freq_{zodiac}'] = zodiac_counts.get(zodiac, 0)
        
        # 季节频率
        for season in df['season'].unique():
            season_mask = df['season'] == season
            season_counts = df[season_mask]['zodiac'].value_counts(normalize=True)
            for zodiac in self.zodiacs:
                df.loc[season_mask, f'season_{season}_{zodiac}'] = season_counts.get(zodiac, 0)
        
        # 节日频率
        festival_mask = df['is_festival']
        if festival_mask.any():
            festival_counts = df[festival_mask]['zodiac'].value_counts(normalize=True)
            for zodiac in self.zodiacs:
                df.loc[festival_mask, f'festival_{zodiac}'] = festival_counts.get(zodiac, 0)
        
        logger.info(f"Added {len(self.zodiacs)*3} zodiac features")
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滚动窗口特征"""
        windows = [7, 30, 100]  # 7天、30天和100天窗口
        
        # 预计算出现标记
        occur_cols = {}
        for zodiac in self.zodiacs:
            col_name = f'occur_{zodiac}'
            df[col_name] = (df['zodiac'] == zodiac).astype(int)
            occur_cols[zodiac] = col_name
        
        # 计算滚动特征
        for window in windows:
            for zodiac in self.zodiacs:
                df[f'rolling_{window}_{zodiac}'] = (
                    df[occur_cols[zodiac]]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .fillna(0)
                )
        
        logger.info(f"Added {len(windows)*len(self.zodiacs)} rolling features")
        return df

    def detect_patterns(self) -> Dict:
        """检测历史模式"""
        patterns = {
            'consecutive': defaultdict(int),
            'intervals': defaultdict(dict),
            'festival_boost': defaultdict(dict),
            'strength_metrics': defaultdict(dict)
        }
        
        # 连续出现模式
        self._detect_consecutive_patterns(patterns)
        
        # 间隔模式
        self._detect_interval_patterns(patterns)
        
        # 节日效应
        self._detect_festival_patterns(patterns)
        
        logger.info(
            f"Patterns detected: "
            f"{len(patterns['consecutive'])} consecutive, "
            f"{len(patterns['intervals'])} interval, "
            f"{len(patterns['festival_boost'])} festival"
        )
        return patterns

    def _detect_consecutive_patterns(self, patterns: Dict):
        """检测连续出现模式"""
        current_streak = 1
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['zodiac'] == self.df.iloc[i-1]['zodiac']:
                current_streak += 1
            else:
                if current_streak > 1:
                    zodiac = self.df.iloc[i-1]['zodiac']
                    patterns['consecutive'][zodiac] = max(
                        patterns['consecutive'][zodiac], 
                        current_streak
                    )
                    patterns['strength_metrics']['consecutive_strength'][zodiac] = \
                        patterns['consecutive'][zodiac] / len(self.df)
                current_streak = 1

    def _detect_interval_patterns(self, patterns: Dict):
        """检测间隔模式"""
        last_occurrence = {}
        for idx, row in self.df.iterrows():
            zodiac = row['zodiac']
            if zodiac in last_occurrence:
                interval = idx - last_occurrence[zodiac]
                patterns['intervals'][zodiac].setdefault('intervals', []).append(interval)
            last_occurrence[zodiac] = idx
        
        # 计算统计量
        for zodiac in patterns['intervals']:
            intervals = patterns['intervals'][zodiac]['intervals']
            if intervals:
                patterns['intervals'][zodiac].update({
                    'mean': np.mean(intervals),
                    'std': np.std(intervals),
                    'max': max(intervals)
                })
                patterns['strength_metrics']['interval_consistency'][zodiac] = (
                    np.mean(intervals) / max(1, np.std(intervals))
            else:
                patterns['intervals'][zodiac].update({
                    'mean': 0,
                    'std': 0,
                    'max': 0
                })

    def _detect_festival_patterns(self, patterns: Dict):
        """检测节日效应"""
        festival_data = self.df[self.df['is_festival']]
        if not festival_data.empty:
            for festival, group in festival_data.groupby('festival'):
                zodiac_counts = group['zodiac'].value_counts(normalize=True)
                if not zodiac_counts.empty:
                    top_zodiac = zodiac_counts.idxmax()
                    patterns['festival_boost'][festival] = {
                        'zodiac': top_zodiac,
                        'probability': zodiac_counts.max(),
                        'occurrences': zodiac_counts[top_zodiac]
                    }

    def _init_performance_metrics(self) -> Dict:
        """初始化性能指标"""
        return {
            'accuracy_history': [],
            'feature_stability': 0,
            'model_drift': 0,
            'last_cv_scores': []
        }

    def backtest_strategy(self) -> Tuple[pd.DataFrame, float]:
        """策略回测"""
        if len(self.df) < BACKTEST_WINDOW + 10:
            logger.warning("Insufficient data for backtesting")
            return pd.DataFrame(), 0.0
        
        window = self._determine_window_size()
        logger.info(f"Starting backtest with {window}-period window")
        
        results = []
        error_patterns = defaultdict(int)
        
        for i in range(window, len(self.df) - 1):
            train = self.df.iloc[i-window:i]
            test = self.df.iloc[i:i+1]
            
            # 更新策略
            self.strategy_manager.update_combo_probs(train)
            
            # 预测验证
            feature_row = train.iloc[-1]
            pred, _ = self.strategy_manager.generate_prediction(
                feature_row, 
                feature_row['zodiac']
            )
            actual = test.iloc[0]['zodiac']
            
            # 记录结果
            results.append({
                'period': test['expect'].values[0],
                'actual': actual,
                'predicted': pred,
                'is_correct': int(actual in pred)
            })
            
            # 记录错误模式
            if actual not in pred:
                error_patterns[f"{feature_row['zodiac']}->{actual}"] += 1
        
        # 计算准确率
        result_df = pd.DataFrame(results)
        accuracy = result_df['is_correct'].mean()
        
        # 交叉验证
        cv_accuracy = self.cross_validate()
        
        # 综合评估
        final_accuracy = self._evaluate_results(accuracy, cv_accuracy, error_patterns)
        
        logger.info(f"Backtest completed. Final accuracy: {final_accuracy:.2%}")
        return result_df, final_accuracy

    def _determine_window_size(self) -> int:
        """动态确定回测窗口大小"""
        volatility = self._calculate_volatility()
        base_window = BACKTEST_WINDOW
        
        if volatility > 0.6:
            return max(50, int(base_window * 0.7))
        elif volatility > 0.4:
            return base_window
        else:
            return min(300, int(base_window * 1.3))

    def _calculate_volatility(self) -> float:
        """计算数据波动性"""
        recent = self.df.tail(200) if len(self.df) >= 200 else self.df
        volatilities = []
        
        for zodiac in self.zodiacs:
            occurrences = (recent['zodiac'] == zodiac).astype(int).values
            if np.mean(occurrences) > 0:
                cv = variation(occurrences) if len(occurrences) > 1 else 0
                volatilities.append(cv)
        
        return np.mean(volatilities) if volatilities else 0.5

    def cross_validate(self, n_splits: int = 5) -> float:
        """时间序列交叉验证"""
        if len(self.df) < 100:
            logger.warning("Insufficient data for cross-validation")
            return 0.0
        
        n_splits = min(n_splits, len(self.df) // 20)
        n_splits = max(2, n_splits)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"Starting {n_splits}-fold cross validation")
        accuracies = []
        overfit_flags = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(self.df), 1):
            train_data = self.df.iloc[train_idx]
            test_data = self.df.iloc[test_idx]
            
            # 训练临时模型
            temp_manager = StrategyManager()
            temp_manager.update_combo_probs(train_data)
            
            # 验证
            fold_acc, is_overfit = self._validate_fold(temp_manager, train_data, test_data)
            accuracies.append(fold_acc)
            overfit_flags.append(is_overfit)
            
            logger.info(f"Fold {fold}: Accuracy={fold_acc:.2%} | Overfit={is_overfit}")
        
        avg_accuracy = np.mean(accuracies)
        self.performance_metrics['last_cv_scores'] = accuracies
        
        # 过拟合处理
        if sum(overfit_flags) > n_splits // 2:
            self._handle_overfitting()
        
        logger.info(f"Cross validation completed. Avg accuracy: {avg_accuracy:.2%}")
        return avg_accuracy

    def _validate_fold(self, manager: StrategyManager, 
                      train_data: pd.DataFrame, 
                      test_data: pd.DataFrame) -> Tuple[float, bool]:
        """验证单折数据"""
        test_hits = 0
        train_hits = 0
        
        for i in range(len(test_data) - 1):
            # 测试集预测
            feature_row = test_data.iloc[i]
            actual = test_data.iloc[i+1]['zodiac']
            pred, _ = manager.generate_prediction(feature_row, feature_row['zodiac'])
            test_hits += int(actual in pred)
            
            # 训练集预测（用于过拟合检测）
            train_row = train_data.iloc[i % len(train_data)]
            train_actual = train_data.iloc[(i+1) % len(train_data)]['zodiac']
            train_pred, _ = manager.generate_prediction(train_row, train_row['zodiac'])
            train_hits += int(train_actual in train_pred)
        
        test_acc = test_hits / max(1, (len(test_data) - 1))
        train_acc = train_hits / max(1, (len(test_data) - 1))
        is_overfit = (train_acc - test_acc) > 0.15
        
        return test_acc, is_overfit

    def _handle_overfitting(self):
        """处理过拟合情况"""
        logger.warning("Handling overfitting...")
        
        # 调整ML模型权重
        old_weight = self.strategy_manager.weights.get('ml_model', 0)
        new_weight = max(0.1, old_weight * 0.7)
        self.strategy_manager.weights['ml_model'] = new_weight
        
        # 重新训练模型
        self.ml_predictor.train_model(self.df, retrain=True)
        
        logger.info(f"Adjusted ML model weight from {old_weight:.2f} to {new_weight:.2f}")

    def _evaluate_results(self, 
                         base_accuracy: float, 
                         cv_accuracy: float, 
                         error_patterns: Dict) -> float:
        """综合评估结果"""
        combined_acc = 0.7 * base_accuracy + 0.3 * cv_accuracy
        
        # 差异过大时保守处理
        if abs(base_accuracy - cv_accuracy) > 0.2:
            logger.warning("Large discrepancy between backtest and CV")
            combined_acc = min(base_accuracy, cv_accuracy)
        
        # 更新策略
        self.strategy_manager.adjust(combined_acc, error_patterns)
        
        # 更新监控指标
        self.performance_metrics['accuracy_history'].append(combined_acc)
        if len(self.performance_metrics['accuracy_history']) > 10:
            self.performance_metrics['accuracy_history'].pop(0)
        
        return combined_acc

    def predict_next(self) -> Dict:
        """预测下期结果"""
        if self.df.empty:
            logger.warning("No data available for prediction")
            return {"prediction": [], "confidence": 0}
        
        latest = self.df.iloc[-1]
        feature_row = self.df.tail(BACKTEST_WINDOW).iloc[-1] if len(self.df) >= BACKTEST_WINDOW else latest
        
        # 生成预测
        prediction, factors = self.strategy_manager.generate_prediction(
            feature_row, 
            latest['zodiac']
        )
        
        # 应用模式增强
        prediction = self._apply_pattern_enhancement(
            prediction, 
            latest['zodiac'], 
            latest['date'] + timedelta(days=1)
        )
        
        # 计算置信度
        confidence = self._calculate_confidence(prediction, factors)
        
        return {
            "next_period": self._generate_next_period_number(latest['expect']),
            "prediction": prediction[:3],  # 返回top3预测
            "confidence": confidence,
            "factors": factors
        }

    def _apply_pattern_enhancement(self, 
                                  prediction: List[str], 
                                  last_zodiac: str, 
                                  target_date: datetime) -> List[str]:
        """应用历史模式增强预测"""
        if not prediction:
            return prediction
        
        # 节日效应
        festival = self._detect_festival(target_date)
        if festival in self.patterns['festival_boost']:
            boost_zodiac = self.patterns['festival_boost'][festival]['zodiac']
            if boost_zodiac not in prediction:
                prediction[-1] = boost_zodiac
                logger.info(f"Applied festival boost: {festival}->{boost_zodiac}")
        
        # 间隔模式
        for zodiac, stats in self.patterns['intervals'].items():
            if zodiac not in prediction and stats.get('mean', 0) > 0:
                last_idx = self.df[self.df['zodiac'] == zodiac].index[-1]
                current_interval = len(self.df) - last_idx
                if current_interval >= stats['mean'] - stats['std']:
                    prediction[-1] = zodiac
                    logger.info(f"Applied interval pattern: {zodiac} (interval={current_interval})")
        
        return prediction

    def _calculate_confidence(self, prediction: List[str], factors: Dict) -> float:
        """计算预测置信度"""
        if not prediction:
            return 0.0
        
        # 因子一致性
        consensus = sum(
            1 for f in factors.values() 
            if prediction[0] in f[:2]
        ) / max(1, len(factors))
        
        # 历史准确率
        hist_acc = np.mean(self.performance_metrics['accuracy_history']) \
            if self.performance_metrics['accuracy_history'] else 0.5
        
        return min(0.99, (0.6 * consensus + 0.4 * hist_acc))

    def _generate_next_period_number(self, current: str) -> str:
        """生成下期期号"""
        try:
            if current.isdigit():
                return str(int(current) + 1)
            # 处理特殊期号格式
            match = re.match(r"(\d+)([A-Z]*)", current)
            if match:
                num, suffix = match.groups()
                if not suffix:
                    return str(int(num) + 1)
                if suffix == "A":
                    return f"{num}B"
                return f"{int(num)+1}A"
        except:
            pass
        return "未知"

    def generate_report(self) -> str:
        """生成分析报告"""
        if self.df.empty:
            return "===== 分析报告 =====\n错误：无有效数据"
        
        # 基础信息
        latest = self.df.iloc[-1]
        zodiac_dist = self.df['zodiac'].value_counts(normalize=True).sort_values(ascending=False)
        
        # 回测结果
        _, accuracy = self.backtest_strategy()
        
        # 预测结果
        prediction = self.predict_next()
        
        # 生成报告
        report = f"""
===== 彩票分析报告 =====
时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
数据统计:
- 总期数: {len(self.df)}
- 最新期号: {latest['expect']}
- 最新开奖: {latest['zodiac']}

生肖分布:
{zodiac_dist.to_string()}

模型表现:
- 回测准确率: {accuracy:.2%}
- 近期平均准确率: {np.mean(self.performance_metrics['accuracy_history']):.2%}
- 特征稳定性: {self.performance_metrics['feature_stability']:.2f}

下期预测:
- 期号: {prediction['next_period']}
- 推荐生肖: {", ".join(prediction['prediction'])}
- 置信度: {prediction['confidence']:.0%}
"""
        logger.info("Report generated")
        return report

# 示例用法
if __name__ == "__main__":
    analyzer = LotteryAnalyzer()
    print(analyzer.generate_report())
