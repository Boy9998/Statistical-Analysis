import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping
from config import BACKTEST_WINDOW
from datetime import datetime

class LotteryAnalyzer:
    def __init__(self):
        self.df = fetch_historical_data()
        if not self.df.empty:
            self.df['zodiac'] = self.df['special'].apply(zodiac_mapping)
            self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        
    def analyze_zodiac_patterns(self):
        """分析生肖出现规律"""
        if self.df.empty:
            print("警告：没有可用数据进行分析")
            return {
                'frequency': pd.DataFrame(),
                'transition_matrix': pd.DataFrame(),
                'seasonal_effect': pd.DataFrame()
            }
        
        # 生肖频率分析
        freq = self.df['zodiac'].value_counts().reset_index()
        freq.columns = ['生肖', '出现次数']
        freq['频率(%)'] = round(freq['出现次数'] / len(self.df) * 100, 2)
        
        # 生肖转移矩阵
        transition = pd.crosstab(
            self.df['zodiac'].shift(-1), 
            self.df['zodiac'], 
            normalize=1
        ).round(3) * 100
        
        # 季节效应
        season_map = {1: '冬', 2: '冬', 3: '春', 4: '春', 5: '春', 
                      6: '夏', 7: '夏', 8: '夏', 9: '秋', 10: '秋', 11: '秋', 12: '冬'}
        self.df['season'] = self.df['month'].map(season_map)
        season_freq = self.df.groupby(['season', 'zodiac']).size().unstack().fillna(0)
        
        return {
            'frequency': freq,
            'transition_matrix': transition,
            'seasonal_effect': season_freq
        }
    
    def backtest_strategy(self, window=BACKTEST_WINDOW):
        """回测预测策略"""
        if self.df.empty or len(self.df) < window:
            print(f"警告：数据不足，无法进行回测（需要{window}期，实际只有{len(self.df)}期）")
            return pd.DataFrame(), 0.0
        
        recent = self.df.tail(window).copy()
        results = []
        
        for i in range(len(recent)-1):
            # 使用历史数据预测
            train = recent.iloc[:i+1]
            actual = recent.iloc[i+1]['zodiac']
            
            # 预测策略：近期高频生肖 + 转移概率高的生肖
            top_freq = train['zodiac'].value_counts().head(2).index.tolist()
            last_zodiac = train.iloc[-1]['zodiac']
            top_transition = self.transition_matrix[last_zodiac].nlargest(2).index.tolist()
            
            prediction = list(set(top_freq + top_transition))[:4]
            
            # 记录结果
            results.append({
                '期号': recent.iloc[i+1]['expect'],
                '实际生肖': actual,
                '预测生肖': ", ".join(prediction),
                '是否命中': 1 if actual in prediction else 0
            })
        
        result_df = pd.DataFrame(results)
        accuracy = result_df['是否命中'].mean()
        return result_df, accuracy
    
    def predict_next(self):
        """预测下期生肖"""
        if self.df.empty:
            return {
                'next_number': "未知",
                'prediction': ["无数据"],
                'last_zodiac': "无数据"
            }
        
        # 获取最新数据
        latest = self.df.iloc[-1]
        last_zodiac = latest['zodiac']
        
        # 策略1：近期高频生肖
        top_freq = self.df['zodiac'].tail(100).value_counts().head(2).index.tolist()
        
        # 策略2：转移概率高的生肖
        top_transition = self.transition_matrix[last_zodiac].nlargest(2).index.tolist()
        
        # 组合预测结果
        prediction = list(set(top_freq + top_transition))[:4]
        
        # 下期期号
        if 'expect' in self.df.columns:
            last_expect = self.df.iloc[-1]['expect']
            if isinstance(last_expect, str) and last_expect.isdigit():
                next_num = str(int(last_expect) + 1)
            else:
                next_num = "未知"
        else:
            next_num = "未知"
        
        return {
            'next_number': next_num,
            'prediction': prediction,
            'last_zodiac': last_zodiac
        }
    
    def generate_report(self):
        """生成分析报告"""
        if self.df.empty:
            return "===== 彩票分析报告 =====\n错误：没有获取到有效数据，请检查API"
        
        analysis = self.analyze_zodiac_patterns()
        self.transition_matrix = analysis['transition_matrix']  # 保存用于预测
        backtest_df, accuracy = self.backtest_strategy()
        prediction = self.predict_next()
        
        report = f"""
        ===== 彩票分析报告 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        数据统计：
        - 总期数：{len(self.df)}
        - 数据范围：{self.df['date'].min().date()} 至 {self.df['date'].max().date()}
        
        生肖频率分析：
        {analysis['frequency'].to_string(index=False) if not analysis['frequency'].empty else "无数据"}
        
        最近一期生肖：{self.df.iloc[-1]['zodiac']}
        转移概率最高生肖：{', '.join(self.transition_matrix[self.df.iloc[-1]['zodiac']].nlargest(3).index.tolist())}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        =============================================
        """
        return report
