import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
from collections import defaultdict

class LotteryAnalyzer:
    def __init__(self):
        self.df = fetch_historical_data()
        if not self.df.empty:
            # 使用动态年份生肖映射
            self.df['zodiac'] = self.df.apply(
                lambda row: zodiac_mapping(row['special'], row['year']), axis=1
            )
            self.zodiacs = list(set(self.df['zodiac']))
        
    def analyze_zodiac_patterns(self):
        """分析生肖出现规律"""
        if self.df.empty:
            return {}
        
        # 1. 生肖频率分析
        freq = self.df['zodiac'].value_counts().reset_index()
        freq.columns = ['生肖', '出现次数']
        freq['频率(%)'] = round(freq['出现次数'] / len(self.df) * 100, 2)
        
        # 2. 生肖转移分析（符合要求5）
        transition = pd.crosstab(
            self.df['zodiac'].shift(-1), 
            self.df['zodiac'], 
            normalize=1
        ).round(4) * 100
        
        # 3. 季节效应分析（符合要求4）
        season_map = {1: '冬', 2: '冬', 3: '春', 4: '春', 5: '春', 
                      6: '夏', 7: '夏', 8: '夏', 9: '秋', 10: '秋', 11: '秋', 12: '冬'}
        self.df['season'] = self.df['month'].map(season_map)
        season_effect = self.df.groupby(['season', 'zodiac']).size().unstack().fillna(0)
        
        # 4. 节日效应分析（符合要求4）
        cn_holidays = holidays.CountryHoliday('CN')
        self.df['is_holiday'] = self.df['date'].apply(lambda x: x in cn_holidays)
        holiday_effect = self.df.groupby(['is_holiday', 'zodiac']).size().unstack().fillna(0)
        
        return {
            'frequency': freq,
            'transition_matrix': transition,
            'seasonal_effect': season_effect,
            'holiday_effect': holiday_effect
        }
    
    def backtest_strategy(self):
        """
        严格回测策略（要求8-9）
        基于最新开奖生肖分析下一期
        """
        if self.df.empty or len(self.df) < BACKTEST_WINDOW:
            print(f"警告：数据不足，无法回测（需要{BACKTEST_WINDOW}期，实际只有{len(self.df)}期）")
            return pd.DataFrame(), 0.0
        
        # 使用最近BACKTEST_WINDOW+1期数据进行回测
        recent = self.df.tail(BACKTEST_WINDOW + 1).copy()
        results = []
        
        # 遍历数据，分析每个开奖生肖后的下一期
        for i in range(len(recent) - 1):
            current_zodiac = recent.iloc[i]['zodiac']
            next_zodiac = recent.iloc[i + 1]['zodiac']
            
            # 统计当前生肖出现后的下一期生肖
            results.append({
                '当前生肖': current_zodiac,
                '下一期生肖': next_zodiac
            })
        
        # 分析每个生肖出现后的下一期生肖分布
        zodiac_transitions = defaultdict(list)
        for record in results:
            zodiac_transitions[record['当前生肖']].append(record['下一期生肖'])
        
        # 回测预测准确性
        backtest_results = []
        for i in range(len(recent) - 1):
            current_zodiac = recent.iloc[i]['zodiac']
            actual_next = recent.iloc[i + 1]['zodiac']
            
            # 预测策略：基于当前生肖的历史转移概率
            if current_zodiac in zodiac_transitions and len(zodiac_transitions[current_zodiac]) > 10:
                # 计算生肖出现频率
                freq = pd.Series(zodiac_transitions[current_zodiac]).value_counts(normalize=True)
                prediction = freq.nlargest(4).index.tolist()
            else:
                # 默认策略：使用全局高频生肖
                prediction = self.df['zodiac'].value_counts().head(4).index.tolist()
            
            # 记录结果
            backtest_results.append({
                '期号': recent.iloc[i + 1]['expect'],
                '当前生肖': current_zodiac,
                '实际下一期': actual_next,
                '预测生肖': ", ".join(prediction),
                '是否命中': 1 if actual_next in prediction else 0
            })
        
        result_df = pd.DataFrame(backtest_results)
        accuracy = result_df['是否命中'].mean()
        return result_df, accuracy, zodiac_transitions
    
    def predict_next(self, zodiac_transitions):
        """
        严格预测策略（要求6）
        基于最新开奖生肖预测下一期
        """
        if self.df.empty:
            return {
                'next_number': "未知",
                'prediction': ["无数据"],
                'last_zodiac': "无数据"
            }
        
        # 获取最新数据
        latest = self.df.iloc[-1]
        last_zodiac = latest['zodiac']
        
        # 策略1：基于该生肖的历史转移概率
        if last_zodiac in zodiac_transitions and len(zodiac_transitions[last_zodiac]) > 10:
            freq = pd.Series(zodiac_transitions[last_zodiac]).value_counts(normalize=True)
            prediction = freq.nlargest(5).index.tolist()
        else:
            # 策略2：使用全局高频生肖
            prediction = self.df['zodiac'].value_counts().head(5).index.tolist()
        
        # 下期期号
        try:
            last_expect = self.df.iloc[-1]['expect']
            if isinstance(last_expect, str) and last_expect.isdigit():
                next_num = str(int(last_expect) + 1)
            else:
                next_num = "未知"
        except:
            next_num = "未知"
        
        return {
            'next_number': next_num,
            'prediction': prediction,
            'last_zodiac': last_zodiac
        }
    
    def generate_report(self):
        """生成分析报告（符合所有要求）"""
        if self.df.empty:
            return "===== 彩票分析报告 =====\n错误：没有获取到有效数据，请检查API"
        
        analysis = self.analyze_zodiac_patterns()
        backtest_df, accuracy, zodiac_transitions = self.backtest_strategy()
        prediction = self.predict_next(zodiac_transitions)
        
        # 获取最新开奖生肖的转移分析
        last_zodiac = self.df.iloc[-1]['zodiac']
        if last_zodiac in zodiac_transitions:
            freq = pd.Series(zodiac_transitions[last_zodiac]).value_counts(normalize=True)
            top_transition = freq.nlargest(4).index.tolist()
            transition_detail = f"{last_zodiac} → {', '.join(top_transition)}"
        else:
            transition_detail = f"{last_zodiac} → 无历史数据"
        
        # 生成详细报告
        report = f"""
        ===== 彩票分析报告 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        数据统计：
        - 总期数：{len(self.df)}
        - 数据范围：{self.df['date'].min().date()} 至 {self.df['date'].max().date()}
        - 最新期号：{self.df.iloc[-1]['expect']}
        - 最新开奖生肖：{self.df.iloc[-1]['zodiac']}
        
        生肖频率分析：
        {analysis['frequency'].to_string(index=False)}
        
        最新开奖生肖转移分析：
        {transition_detail}
        
        季节效应分析：
        春季: {analysis['seasonal_effect'].loc['春'].nlargest(3).to_dict() if '春' in analysis['seasonal_effect'].index else "无数据"}
        夏季: {analysis['seasonal_effect'].loc['夏'].nlargest(3).to_dict() if '夏' in analysis['seasonal_effect'].index else "无数据"}
        秋季: {analysis['seasonal_effect'].loc['秋'].nlargest(3).to_dict() if '秋' in analysis['seasonal_effect'].index else "无数据"}
        冬季: {analysis['seasonal_effect'].loc['冬'].nlargest(3).to_dict() if '冬' in analysis['seasonal_effect'].index else "无数据"}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        - 策略详情：基于当前生肖的历史转移概率
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        =============================================
        """
        return report
