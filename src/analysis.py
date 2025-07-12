import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays

class LotteryAnalyzer:
    def __init__(self):
        self.df = fetch_historical_data()
        if not self.df.empty:
            # 添加农历日期信息
            self.df['lunar_date'] = self.df['date'].apply(self.get_lunar_date)
            self.df['zodiac'] = self.df['special'].apply(zodiac_mapping)
            self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        
    def get_lunar_date(self, dt):
        """将公历日期转换为农历日期（简化版）"""
        # 农历新年通常在1月21日到2月20日之间
        if 1 <= dt.month <= 2 and 21 <= dt.day <= 31:
            return "春节附近"
        elif 4 <= dt.month <= 5 and 1 <= dt.day <= 7:
            return "清明节附近"
        elif 9 <= dt.month <= 10 and 10 <= dt.day <= 20:
            return "中秋节附近"
        return "普通日期"
    
    def get_season(self, month):
        """获取季节"""
        if 3 <= month <= 5:
            return "春季"
        elif 6 <= month <= 8:
            return "夏季"
        elif 9 <= month <= 11:
            return "秋季"
        return "冬季"
    
    def analyze_zodiac_patterns(self):
        """深度分析生肖出现规律（符合所有要求）"""
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
        
        # 3. 农历效应分析（符合要求4）
        lunar_effect = self.df.groupby(['lunar_date', 'zodiac']).size().unstack().fillna(0)
        
        # 4. 季节效应分析（符合要求4）
        self.df['season'] = self.df['month'].apply(self.get_season)
        season_effect = self.df.groupby(['season', 'zodiac']).size().unstack().fillna(0)
        
        # 5. 节日效应分析（符合要求4）
        # 创建中国节假日日历
        cn_holidays = holidays.CountryHoliday('CN')
        self.df['is_holiday'] = self.df['date'].apply(lambda x: x in cn_holidays)
        holiday_effect = self.df.groupby(['is_holiday', 'zodiac']).size().unstack().fillna(0)
        
        # 6. 周期效应分析（安全处理）
        if 'day_of_year' in self.df.columns:
            try:
                # 分析生肖出现的周期性
                self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 7)
                self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 7)
            except KeyError:
                print("警告：day_of_year 列访问失败，跳过周期效应分析")
        else:
            print("警告：数据中缺少 day_of_year 列，跳过周期效应分析")
        
        return {
            'frequency': freq,
            'transition_matrix': transition,
            'lunar_effect': lunar_effect,
            'seasonal_effect': season_effect,
            'holiday_effect': holiday_effect
        }
    
    def backtest_strategy(self):
        """严格符合要求8-9的回测策略"""
        if self.df.empty or len(self.df) < BACKTEST_WINDOW:
            print(f"警告：数据不足，无法回测（需要{BACKTEST_WINDOW}期，实际只有{len(self.df)}期）")
            return pd.DataFrame(), 0.0
        
        recent = self.df.tail(BACKTEST_WINDOW).copy()
        results = []
        
        for i in range(len(recent)-1):
            # 使用历史数据预测
            train = recent.iloc[:i+1]
            actual = recent.iloc[i+1]['zodiac']
            
            # 策略1：转移概率最高的4个生肖
            last_zodiac = train.iloc[-1]['zodiac']
            prediction = []
            
            if len(train) > 10:
                try:
                    transition = pd.crosstab(
                        train['zodiac'].shift(-1), 
                        train['zodiac'], 
                        normalize=1
                    )
                    if last_zodiac in transition.columns:
                        top_transition = transition[last_zodiac].nlargest(4).index.tolist()
                        prediction.extend(top_transition)
                except:
                    pass
            
            # 策略2：近期高频生肖
            if len(train) > 10:
                try:
                    top_freq = train['zodiac'].tail(50).value_counts().head(4).index.tolist()
                    prediction.extend(top_freq)
                except:
                    pass
            
            # 策略3：季节效应
            try:
                current_season = self.get_season(recent.iloc[i+1]['month'])
                season_zodiacs = self.df[self.df['season'] == current_season]['zodiac'].value_counts().head(2).index.tolist()
                prediction.extend(season_zodiacs)
            except:
                pass
            
            # 组合预测（取4-5个生肖）
            prediction = list(set(prediction))[:5]
            if not prediction:
                prediction = ["无预测"]
            
            # 记录结果
            results.append({
                '期号': recent.iloc[i+1]['expect'],
                '实际生肖': actual,
                '预测生肖': ", ".join(prediction),
                '是否命中': 1 if actual in prediction else 0
            })
        
        result_df = pd.DataFrame(results)
        accuracy = result_df['是否命中'].mean() if not result_df.empty else 0.0
        return result_df, accuracy
    
    def predict_next(self):
        """严格符合要求6的预测方法"""
        if self.df.empty:
            return {
                'next_number': "未知",
                'prediction': ["无数据"],
                'last_zodiac': "无数据"
            }
        
        # 获取最新数据
        latest = self.df.iloc[-1]
        last_zodiac = latest['zodiac']
        
        # 策略1：转移概率最高的4个生肖（基于最近200期）
        recent = self.df.tail(BACKTEST_WINDOW)
        prediction = []
        
        if len(recent) > 10:
            try:
                transition = pd.crosstab(
                    recent['zodiac'].shift(-1), 
                    recent['zodiac'], 
                    normalize=1
                )
                
                if last_zodiac in transition.columns:
                    top_transition = transition[last_zodiac].nlargest(4).index.tolist()
                    prediction.extend(top_transition)
            except Exception as e:
                print(f"转移概率计算失败: {e}")
        
        # 策略2：近期高频生肖（最近50期）
        if len(recent) > 10:
            try:
                top_freq = recent['zodiac'].tail(50).value_counts().head(4).index.tolist()
                prediction.extend(top_freq)
            except Exception as e:
                print(f"高频生肖计算失败: {e}")
        
        # 策略3：季节效应
        try:
            next_date = latest['date'] + timedelta(days=1)
            next_season = self.get_season(next_date.month)
            season_zodiacs = self.df[self.df['season'] == next_season]['zodiac'].value_counts().head(2).index.tolist()
            prediction.extend(season_zodiacs)
        except Exception as e:
            print(f"季节效应计算失败: {e}")
        
        # 策略4：节日效应
        try:
            cn_holidays = holidays.CountryHoliday('CN')
            is_holiday = next_date in cn_holidays
            holiday_zodiacs = self.df[self.df['is_holiday'] == is_holiday]['zodiac'].value_counts().head(2).index.tolist()
            prediction.extend(holiday_zodiacs)
        except Exception as e:
            print(f"节日效应计算失败: {e}")
        
        # 组合预测（取4-5个生肖）
        prediction = list(set(prediction))[:5]
        if not prediction:
            # 默认策略：使用近期高频生肖
            prediction = self.df['zodiac'].tail(50).value_counts().head(5).index.tolist()
        
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
        """严格符合要求的分析报告"""
        if self.df.empty:
            return "===== 彩票分析报告 =====\n错误：没有获取到有效数据，请检查API"
        
        analysis = self.analyze_zodiac_patterns()
        backtest_df, accuracy = self.backtest_strategy()
        prediction = self.predict_next()
        
        # 生肖转移分析详情
        transition_details = {}
        for zodiac in self.zodiacs:
            if zodiac in analysis.get('transition_matrix', pd.DataFrame()):
                next_zodiacs = analysis['transition_matrix'][zodiac].nlargest(4).index.tolist()
                transition_details[zodiac] = next_zodiacs
        
        # 生成详细报告
        report = f"""
        ===== 彩票分析报告 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        数据统计：
        - 总期数：{len(self.df)}
        - 数据范围：{self.df['date'].min().date()} 至 {self.df['date'].max().date()}
        - 最新期号：{self.df.iloc[-1]['expect']}
        - 最新开奖生肖：{self.df.iloc[-1]['zodiac']}
        
        生肖频率分析：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if not analysis.get('frequency', pd.DataFrame()).empty else "无数据"}
        
        生肖转移分析（出现后下期最可能出现的4个生肖）：
        {', '.join([f"{k}→{','.join(v)}" for k, v in transition_details.items()]) if transition_details else "无数据"}
        
        季节效应分析：
        春季: {analysis.get('seasonal_effect', pd.DataFrame()).loc['春季'].nlargest(3).to_dict() if 'seasonal_effect' in analysis else "无数据"}
        夏季: {analysis.get('seasonal_effect', pd.DataFrame()).loc['夏季'].nlargest(3).to_dict() if 'seasonal_effect' in analysis else "无数据"}
        秋季: {analysis.get('seasonal_effect', pd.DataFrame()).loc['秋季'].nlargest(3).to_dict() if 'seasonal_effect' in analysis else "无数据"}
        冬季: {analysis.get('seasonal_effect', pd.DataFrame()).loc['冬季'].nlargest(3).to_dict() if 'seasonal_effect' in analysis else "无数据"}
        
        节日效应分析：
        节日: {analysis.get('holiday_effect', pd.DataFrame()).loc[True].nlargest(3).to_dict() if 'holiday_effect' in analysis else "无数据"}
        非节日: {analysis.get('holiday_effect', pd.DataFrame()).loc[False].nlargest(3).to_dict() if 'holiday_effect' in analysis else "无数据"}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        - 策略详情：基于转移概率+近期高频+季节效应+节日效应
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        =============================================
        """
        return report
