import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils import zodiac_mapping
from src.strategy_manager import StrategyManager
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar
from collections import defaultdict
import json

class Backtester:
    def __init__(self, analyzer):
        """初始化回测器"""
        self.analyzer = analyzer
        self.df = analyzer.df
        self.strategy_manager = analyzer.strategy_manager
        self.results = []
        self.accuracy_history = []
        self.error_cases = []
        print(f"初始化回测器，使用窗口大小: {BACKTEST_WINDOW}期")
    
    def run_backtest(self):
        """执行严格回测"""
        if self.df.empty:
            print("无法回测 - 数据为空")
            return pd.DataFrame(), 0.0
        
        if len(self.df) < BACKTEST_WINDOW:
            print(f"警告：数据不足{BACKTEST_WINDOW}期，实际只有{len(self.df)}期")
            return pd.DataFrame(), 0.0
        
        print(f"开始回测策略（最近{BACKTEST_WINDOW}期）...")
        recent = self.df.tail(BACKTEST_WINDOW).copy().reset_index(drop=True)
        
        # 使用进度条
        for i in tqdm(range(len(recent)-1), desc="回测进度"):
            # 使用历史数据预测
            train = recent.iloc[:i+1]
            actual = recent.iloc[i+1]['zodiac']
            last_zodiac = train.iloc[-1]['zodiac']
            target_date = recent.iloc[i+1]['date']
            
            # 生成预测
            try:
                prediction = self._generate_prediction(train, last_zodiac, target_date)
                prediction = self.apply_pattern_enhancement(prediction, last_zodiac, target_date, train)
                
                # 打印最新一期的预测详情
                if i == len(recent)-2:
                    print(f"最新回测预测: 上期生肖={last_zodiac}, 预测生肖={prediction}, 实际生肖={actual}")
            except Exception as e:
                print(f"回测过程中出错: {e}")
                prediction = []
            
            # 检查是否命中
            hit = actual in prediction
            self.accuracy_history.append(hit)
            
            # 记录结果
            result = {
                'date': recent.iloc[i+1]['date'],
                'draw_number': recent.iloc[i+1]['expect'],
                'last_zodiac': last_zodiac,
                'actual_zodiac': actual,
                'predicted_zodiacs': prediction,
                'hit': hit
            }
            self.results.append(result)
            
            # 记录错误案例
            if not hit:
                self.error_cases.append({
                    'date': recent.iloc[i+1]['date'],
                    'draw_number': recent.iloc[i+1]['expect'],
                    'actual_zodiac': actual,
                    'predicted_zodiacs': prediction,
                    'last_zodiac': last_zodiac
                })
        
        # 计算整体准确率
        result_df = pd.DataFrame(self.results)
        accuracy = result_df['hit'].mean()
        hit_count = result_df['hit'].sum()
        print(f"回测完成: 准确率={accuracy:.2%}, 命中次数={hit_count}/{len(result_df)}")
        
        # 根据回测结果调整策略
        self.strategy_manager.adjust(accuracy)
        
        return result_df, accuracy
    
    def _generate_prediction(self, data, last_zodiac, target_date):
        """生成预测结果（核心逻辑）"""
        # 1. 频率分析（基于最近50期）
        freq_window = data.tail(50) if len(data) >= 50 else data
        freq_counts = freq_window['zodiac'].value_counts()
        
        # 2. 转移概率分析
        transition = pd.crosstab(
            data['zodiac'].shift(-1), 
            data['zodiac'], 
            normalize=1
        )
        
        # 3. 季节/节日效应
        season = self.analyzer.get_season(target_date)
        festival = self.analyzer.detect_festival(target_date)
        
        season_data = data[data['season'] == season]
        festival_data = data[data['festival'] == festival]
        
        # 组合预测分数
        scores = {}
        weights = self.strategy_manager.weights
        
        # 应用频率权重
        for zodiac, count in freq_counts.items():
            scores[zodiac] = scores.get(zodiac, 0) + count * weights['frequency']
        
        # 应用转移概率权重
        if last_zodiac in transition.columns:
            for zodiac, prob in transition[last_zodiac].items():
                scores[zodiac] = scores.get(zodiac, 0) + prob * weights['transition'] * 100
        
        # 应用季节权重
        if not season_data.empty:
            season_counts = season_data['zodiac'].value_counts()
            for zodiac, count in season_counts.items():
                scores[zodiac] = scores.get(zodiac, 0) + count * weights['season']
        
        # 应用节日权重
        if festival != "无" and not festival_data.empty:
            festival_counts = festival_data['zodiac'].value_counts()
            for zodiac, count in festival_counts.items():
                scores[zodiac] = scores.get(zodiac, 0) + count * weights['festival']
        
        # 获取得分最高的5个生肖
        if scores:
            sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            prediction = [z for z, _ in sorted_zodiacs[:5]]
        else:
            # 备用策略：使用近期高频生肖
            prediction = freq_counts.head(5).index.tolist()
        
        return prediction
    
    def apply_pattern_enhancement(self, prediction, last_zodiac, target_date, data):
        """应用历史模式增强预测"""
        festival = self.analyzer.detect_festival(target_date)
        patterns = self.analyzer.patterns
        
        # 1. 节日效应增强
        if festival in patterns['festival_boost']:
            boost_zodiac = patterns['festival_boost'][festival]
            if boost_zodiac not in prediction:
                # 如果节日生肖不在预测中，替换掉得分最低的生肖
                prediction = prediction[:-1] + [boost_zodiac]
                print(f"节日效应增强: {festival}节日常见生肖 {boost_zodiac} 加入预测")
        
        # 2. 间隔模式增强
        for zodiac in prediction:
            if zodiac in patterns['intervals']:
                avg_interval = patterns['intervals'][zodiac]
                last_idx = data[data['zodiac'] == zodiac].index[-1] if not data.empty else -1
                current_interval = len(data) - last_idx
                
                # 如果接近平均间隔，提升优先级
                if current_interval >= avg_interval * 0.9:
                    if zodiac in prediction:
                        prediction.remove(zodiac)
                        prediction.insert(0, zodiac)
                    print(f"间隔模式增强: {zodiac} 已间隔 {current_interval}期 (平均 {avg_interval:.1f}期), 提升优先级")
        
        # 3. 连续出现模式处理
        if last_zodiac in patterns['consecutive']:
            max_consecutive = patterns['consecutive'][last_zodiac]
            current_consecutive = 1
            idx = len(data) - 1
            while idx > 0 and data.iloc[idx]['zodiac'] == last_zodiac:
                current_consecutive += 1
                idx -= 1
            
            # 如果连续出现次数接近历史最大值，降低该生肖优先级
            if current_consecutive >= max_consecutive * 0.8:
                if last_zodiac in prediction:
                    prediction.remove(last_zodiac)
                    prediction.append(last_zodiac)
                    print(f"连续模式处理: {last_zodiac} 已连续出现 {current_consecutive}次 (历史最高 {max_consecutive}次), 降低优先级")
        
        return prediction
    
    def generate_backtest_report(self, backtest_df):
        """生成详细的回测报告"""
        if backtest_df.empty:
            return "无回测结果"
        
        # 计算整体准确率
        accuracy = backtest_df['hit'].mean()
        
        # 计算各策略的准确率
        strategy_accuracy = {
            'frequency': 0,
            'transition': 0,
            'season': 0,
            'festival': 0
        }
        
        # 分析错误案例
        error_analysis = defaultdict(int)
        for case in self.error_cases:
            error_analysis[case['actual_zodiac']] += 1
        
        # 生成报告
        report = f"""
        ==== 回测分析报告 ====
        回测窗口大小: {BACKTEST_WINDOW}期
        总测试期数: {len(backtest_df)}
        整体准确率: {accuracy:.2%}
        命中次数: {backtest_df['hit'].sum()}/{len(backtest_df)}
        
        策略权重:
        - 频率权重: {self.strategy_manager.weights['frequency']}
        - 转移概率权重: {self.strategy_manager.weights['transition']}
        - 季节权重: {self.strategy_manager.weights['season']}
        - 节日权重: {self.strategy_manager.weights['festival']}
        
        错误分析:
        {self._format_error_analysis(error_analysis)}
        
        预测改进建议:
        {self._generate_improvement_suggestions()}
        """
        return report
    
    def _format_error_analysis(self, error_analysis):
        """格式化错误分析结果"""
        if not error_analysis:
            return "无错误案例"
        
        total_errors = sum(error_analysis.values())
        sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1], reverse=True)
        
        report = "生肖错误分布:\n"
        for zodiac, count in sorted_errors:
            percentage = count / total_errors * 100
            report += f"- {zodiac}: {count}次 ({percentage:.1f}%)\n"
        
        return report
    
    def _generate_improvement_suggestions(self):
        """生成改进建议"""
        suggestions = []
        weights = self.strategy_manager.weights
        
        # 基于准确率历史的建议
        if self.accuracy_history:
            recent_accuracy = np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else self.accuracy_history[-1]
            
            if recent_accuracy < 0.35:
                suggestions.append("当前准确率较低，建议增加季节和节日权重的比例")
            elif recent_accuracy > 0.45:
                suggestions.append("当前准确率较高，建议增加转移概率权重的比例")
        
        # 基于错误案例的建议
        if self.error_cases:
            # 找出最常出错的生肖
            error_zodiacs = [case['actual_zodiac'] for case in self.error_cases]
            common_error = max(set(error_zodiacs), key=error_zodiacs.count)
            suggestions.append(f"生肖 '{common_error}' 的预测错误最多，建议加强其历史模式分析")
        
        # 基于权重的建议
        if weights['festival'] < 0.2:
            suggestions.append("节日权重较低，建议在重要节日前增加节日效应分析")
        if weights['transition'] < 0.25:
            suggestions.append("转移概率权重较低，建议加强生肖转移模式分析")
        
        return "\n".join(suggestions) if suggestions else "无具体改进建议"
    
    def save_results(self, filename="backtest_results.csv"):
        """保存回测结果到CSV文件"""
        if not self.results:
            print("无回测结果可保存")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"回测结果已保存到 {filename}")
    
    def save_error_cases(self, filename="error_cases.json"):
        """保存错误案例到JSON文件"""
        if not self.error_cases:
            print("无错误案例可保存")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.error_cases, f, ensure_ascii=False, indent=2)
        print(f"错误案例已保存到 {filename}")
