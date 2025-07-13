import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar  # 精确农历计算

class StrategyManager:
    """策略管理器，根据回测准确率动态调整预测权重"""
    def __init__(self):
        # 初始权重分配
        self.weights = {
            'frequency': 0.35,  # 频率权重
            'transition': 0.35,  # 转移概率权重
            'season': 0.15,     # 季节权重
            'festival': 0.15    # 节日权重
        }
        self.accuracy_history = []
        print(f"初始化策略管理器: 权重={self.weights}")
    
    def adjust(self, accuracy):
        """根据准确率动态调整权重"""
        self.accuracy_history.append(accuracy)
        
        # 计算近期准确率趋势 (最近10次)
        trend = np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else accuracy
        
        # 根据趋势调整权重
        if trend < 0.35:
            # 准确率低时增加节日和季节权重
            self.weights['festival'] = min(0.25, self.weights['festival'] + 0.05)
            self.weights['season'] = min(0.25, self.weights['season'] + 0.05)
            self.weights['frequency'] = max(0.25, self.weights['frequency'] - 0.05)
            self.weights['transition'] = max(0.25, self.weights['transition'] - 0.05)
            print(f"策略调整: 准确率低({trend:.2f})，增加季节/节日权重")
        elif trend > 0.45:
            # 准确率高时增加转移概率权重
            self.weights['transition'] = min(0.45, self.weights['transition'] + 0.05)
            self.weights['frequency'] = max(0.25, self.weights['frequency'] - 0.05)
            print(f"策略调整: 准确率高({trend:.2f})，增加转移概率权重")
        
        # 归一化权重
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] = round(self.weights[key] / total, 2)
        
        print(f"调整后权重: {self.weights}")

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
            
            # 添加农历和节日信息
            print("添加农历和节日信息...")
            self.df['lunar'] = self.df['date'].apply(self.get_lunar_date)
            self.df['festival'] = self.df['date'].apply(self.detect_festival)
            self.df['season'] = self.df['date'].apply(self.get_season)
            
            # 初始化策略管理器
            self.strategy_manager = StrategyManager()
            
            # 打印最新开奖信息
            latest = self.df.iloc[-1]
            print(f"最新开奖记录: 期号 {latest['expect']}, 日期 {latest['date'].date()}, 生肖 {latest['zodiac']}")
        else:
            print("警告：未获取到任何有效数据")
            self.strategy_manager = StrategyManager()
    
    def get_lunar_date(self, dt):
        """精确转换公历到农历"""
        try:
            solar = Solar(dt.year, dt.month, dt.day)
            lunar = Converter.Solar2Lunar(solar)
            return lunar
        except Exception as e:
            print(f"农历转换错误: {e}, 日期: {dt}")
            return None
    
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
            
            # 策略：使用加权组合预测
            try:
                # 获取预测
                prediction = self._generate_prediction(train, last_zodiac, recent.iloc[i+1]['date'])
                
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
            
            # 根据回测结果调整策略
            self.strategy_manager.adjust(accuracy)
        else:
            accuracy = 0.0
            print("回测完成: 无有效结果")
        
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
        season = self.get_season(target_date)
        festival = self.detect_festival(target_date)
        
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
        
        # 获取得分最高的4-5个生肖
        if scores:
            sorted_zodiacs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            prediction = [z for z, _ in sorted_zodiacs[:5]]
        else:
            # 备用策略：使用近期高频生肖
            prediction = freq_counts.head(4).index.tolist()
        
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
        print(f"当前策略权重: {self.strategy_manager.weights}")
        
        # 预测目标日期（下一天）
        target_date = latest['date'] + timedelta(days=1)
        
        # 基于最近200期数据
        if len(self.df) < BACKTEST_WINDOW:
            print(f"数据不足{BACKTEST_WINDOW}期，使用全部数据进行预测")
            recent = self.df
        else:
            print(f"使用最近{BACKTEST_WINDOW}期数据预测")
            recent = self.df.tail(BACKTEST_WINDOW)
        
        # 生成预测
        prediction = self._generate_prediction(recent, last_zodiac, target_date)
        
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
        
        策略权重：
        - 频率权重: {self.strategy_manager.weights['frequency']}
        - 转移概率权重: {self.strategy_manager.weights['transition']}
        - 季节权重: {self.strategy_manager.weights['season']}
        - 节日权重: {self.strategy_manager.weights['festival']}
        
        生肖频率分析（全部历史数据）：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if 'frequency' in analysis else "无数据"}
        
        最新开奖生肖分析：
        - 生肖: {last_zodiac}
        - 出现后下期最可能出现的4个生肖: {", ".join(transition_details.get(last_zodiac, ["无数据"]))}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        - 策略详情：自适应权重预测
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        - 预测依据：基于当前策略权重组合多个因素
        
        =============================================
        """
        print("分析报告生成完成")
        return report
