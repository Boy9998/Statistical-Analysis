import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar  # 精确农历计算

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
            valid_zodiacs = self.df['zodiac'].isin(["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"])
            if not valid_zodiacs.all():
                invalid_count = len(self.df) - valid_zodiacs.sum()
                print(f"警告：发现 {invalid_count} 条记录的生肖映射无效")
                self.df = self.df[valid_zodiacs]
            
            self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
            
            # ==== 新增代码：添加农历和节日信息 ====
            print("添加农历和节日信息...")
            # 添加农历日期列
            self.df['lunar'] = self.df['date'].apply(self.get_lunar_date)
            # 添加节日列
            self.df['festival'] = self.df['date'].apply(self.detect_festival)
            # 添加季节列
            self.df['season'] = self.df['date'].apply(self.get_season)
            # ================================
            
            # 打印最新开奖信息
            latest = self.df.iloc[-1]
            print(f"最新开奖记录: 期号 {latest['expect']}, 日期 {latest['date'].date()}, 生肖 {latest['zodiac']}")
            # 打印农历和节日测试
            print(f"农历测试: {latest['date'].date()} -> {self.get_lunar_date(latest['date'])}")
            print(f"节日测试: {self.detect_festival(latest['date'])}")
            
            # ==== 添加节日识别测试 ====
            test_dates = [
                datetime(2023, 1, 22),  # 春节
                datetime(2023, 9, 29),  # 中秋
                datetime(2023, 4, 5)    # 清明
            ]
            print("\n节日识别测试:")
            for date in test_dates:
                festival = self.detect_festival(date)
                print(f"{date.strftime('%Y-%m-%d')} -> {festival}")
        else:
            print("警告：未获取到任何有效数据")
    
    # ==== 新增函数：精确农历计算 ====
    def get_lunar_date(self, dt):
        """精确转换公历到农历"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
    # ==== 新增函数：节日检测 ====
    def detect_festival(self, dt):
        """识别传统节日"""
        lunar = self.get_lunar_date(dt)
        if not lunar:
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
        
        # 春节范围：农历正月初一至十五
        if lunar.month == 1 and 1 <= lunar.day <= 15:
            return "春节"
        
        # 精确匹配
        if lunar_date in lunar_festivals:
            return lunar_festivals[lunar_date]
        if solar_date in solar_festivals:
            return solar_festivals[solar_date]
        
        return "无"
    
    # ==== 新增函数：季节检测 ====
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
        """严格回测预测策略（基于最近200期）"""
        if self.df.empty:
            print("无法回测 - 数据为空")
            return pd.DataFrame(), 0.0
        
        if len(self.df) < BACKTEST_WINDOW:
            print(f"警告：数据不足{BACKTEST_WINDOW}期，实际只有{len(self.df)}期")
            return pd.DataFrame(), 0.0
        
        print(f"开始回测策略（最近{BACKTEST_WINDOW}期）...")
        recent = self.df.tail(BACKTEST_WINDOW).copy().reset_index(drop=True)
        results = []
        
        for i in range(len(recent)-1):
            # 使用历史数据预测
            train = recent.iloc[:i+1]
            actual = recent.iloc[i+1]['zodiac']
            last_zodiac = train.iloc[-1]['zodiac']
            
            # 策略：转移概率最高的4个生肖
            try:
                # 创建转移矩阵
                transition = pd.crosstab(
                    train['zodiac'].shift(-1), 
                    train['zodiac'], 
                    normalize=1
                )
                
                prediction = []
                if last_zodiac in transition.columns:
                    # 获取转移概率最高的4个生肖
                    top_zodiacs = transition[last_zodiac].nlargest(4)
                    prediction = top_zodiacs.index.tolist()
                    
                    # 打印调试信息
                    if i == len(recent)-2:  # 最新一期
                        print(f"最新回测预测: 上期生肖={last_zodiac}, 预测生肖={prediction}, 实际生肖={actual}")
            except Exception as e:
                print(f"回测过程中出错: {e}")
                prediction = []
            
            # 记录结果
            results.append({
                '期号': recent.iloc[i+1]['expect'],
                '上期生肖': last_zodiac,
                '实际生肖': actual,
                '预测生肖': ", ".join(prediction) if prediction else "无预测",
                '是否命中': 1 if actual in prediction else 0
            })
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            accuracy = result_df['是否命中'].mean()
            hit_count = result_df['是否命中'].sum()
            print(f"回测完成: 准确率={accuracy:.2%}, 命中次数={hit_count}/{len(result_df)}")
        else:
            accuracy = 0.0
            print("回测完成: 无有效结果")
        
        return result_df, accuracy
    
    def predict_next(self):
        """预测下期生肖（严格基于最近200期）"""
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
        
        # 基于最近200期数据
        if len(self.df) < BACKTEST_WINDOW:
            print(f"数据不足{BACKTEST_WINDOW}期，使用全部数据进行预测")
            recent = self.df
        else:
            print(f"使用最近{BACKTEST_WINDOW}期数据预测")
            recent = self.df.tail(BACKTEST_WINDOW)
        
        # 策略：转移概率最高的4个生肖
        try:
            # 创建转移矩阵
            transition = pd.crosstab(
                recent['zodiac'].shift(-1), 
                recent['zodiac'], 
                normalize=1
            )
            
            prediction = []
            if last_zodiac in transition.columns:
                # 获取转移概率最高的4个生肖
                top_zodiacs = transition[last_zodiac].nlargest(4)
                prediction = top_zodiacs.index.tolist()
                
                # 打印转移概率详情
                print(f"转移概率分析: {last_zodiac} → {', '.join([f'{z}({p:.1%})' for z, p in top_zodiacs.items()])}")
            else:
                print(f"警告：生肖 '{last_zodiac}' 在转移矩阵中无数据")
                
                # 备用策略：使用近期高频生肖
                top_freq = recent['zodiac'].value_counts().head(4).index.tolist()
                prediction = top_freq
                print(f"使用备用策略: 近期高频生肖 - {', '.join(top_freq)}")
        except Exception as e:
            print(f"预测过程中出错: {e}")
            # 默认策略：使用近期高频生肖
            top_freq = self.df['zodiac'].tail(50).value_counts().head(4).index.tolist()
            prediction = top_freq
            print(f"使用备用策略: 近期高频生肖 - {', '.join(top_freq)}")
        
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
        return {
            'next_number': next_num,
            'prediction': prediction,
            'last_zodiac': last_zodiac
        }
    
    def generate_report(self):
        """生成符合要求的分析报告"""
        if self.df.empty:
            report = "===== 彩票分析报告 =====\n错误：没有获取到有效数据，请检查API"
            print(report)
            return report
        
        # 获取最新开奖信息
        latest = self.df.iloc[-1]
        last_expect = latest['expect']
        last_zodiac = latest['zodiac']
        last_date = latest['date'].strftime('%Y-%m-%d')
        
        # 生肖分析
        analysis = self.analyze_zodiac_patterns()
        
        # 回测结果
        backtest_df, accuracy = self.backtest_strategy()
        
        # 预测下期
        prediction = self.predict_next()
        
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
        ===== 彩票分析报告 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        
        数据统计：
        - 总期数：{len(self.df)}
        - 数据范围：{self.df['date'].min().date()} 至 {last_date}
        - 最新期号：{last_expect}
        - 最新开奖生肖：{last_zodiac}
        
        生肖频率分析（全部历史数据）：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if 'frequency' in analysis else "无数据"}
        
        最新开奖生肖分析：
        - 生肖: {last_zodiac}
        - 出现后下期最可能出现的4个生肖: {", ".join(transition_details.get(last_zodiac, ["无数据"]))}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        - 策略详情：基于上期生肖的转移概率预测
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        - 预测依据：基于上期生肖 '{last_zodiac}' 的转移概率分析
        
        =============================================
        """
        print("分析报告生成完成")
        return report
