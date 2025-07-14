import pandas as pd
import numpy as np
from src.utils import fetch_historical_data, zodiac_mapping, log_error  # 添加log_error导入
from config import BACKTEST_WINDOW
from datetime import datetime, timedelta
import holidays
import re
from lunarcalendar import Converter, Solar, Lunar
from collections import defaultdict

class StrategyManager:
    """策略管理器，根据回测准确率动态调整预测权重"""
    def __init__(self):
        # 初始权重分配
        self.weights = {
            'frequency': 0.30,  # 频率权重
            'transition': 0.30,  # 转移概率权重
            'season': 0.20,     # 季节权重
            'festival': 0.20    # 节日权重
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
    def __init__(self, df=None):
        """初始化分析器，获取历史数据并处理生肖映射"""
        if df is not None:
            self.df = df
            print(f"使用提供的 {len(self.df)} 条历史开奖记录")
        else:
            print("开始获取历史数据...")
            self.df = fetch_historical_data()
            if not self.df.empty:
                print(f"成功获取 {len(self.df)} 条历史开奖记录")
            else:
                print("警告：未获取到任何有效数据")
        
        if not self.df.empty:
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
            
            # 添加基础时序特征（修复添加）
            print("添加基础时序特征...")
            self.df = self.add_temporal_features(self.df)
            
            # 初始化策略管理器
            self.strategy_manager = StrategyManager()
            
            # 检测历史模式
            print("检测历史模式...")
            self.patterns = self.detect_patterns()
            
            # 打印最新开奖信息
            latest = self.df.iloc[-1]
            print(f"最新开奖记录: 期号 {latest['expect']}, 日期 {latest['date'].date()}, 生肖 {latest['zodiac']}")
        else:
            print("警告：未获取到任何有效数据")
            self.strategy_manager = StrategyManager()
            self.patterns = {}
    
    def add_temporal_features(self, df):
        """添加星期几、月份等基础时序特征"""
        # 确保日期是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # 添加星期几特征 (1-7 表示周一到周日)
        df['weekday'] = df['date'].dt.dayofweek + 1
        
        # 添加月份特征
        df['month'] = df['date'].dt.month
        
        return df
    
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
    
    def detect_patterns(self):
        """检测历史模式（连续出现、间隔模式、节日效应）"""
        patterns = {
            'consecutive': defaultdict(int),
            'intervals': defaultdict(list),
            'festival_boost': defaultdict(lambda: defaultdict(int))
        }
        
        # 1. 连续出现模式
        consecutive_count = 1
        last_zodiac = None
        for idx, row in self.df.iterrows():
            if row['zodiac'] == last_zodiac:
                consecutive_count += 1
            else:
                if consecutive_count > 1:
                    patterns['consecutive'][last_zodiac] = max(patterns['consecutive'][last_zodiac], consecutive_count)
                consecutive_count = 1
                last_zodiac = row['zodiac']
        
        # 2. 间隔出现模式
        last_occurrence = {}
        for idx, row in self.df.iterrows():
            zodiac = row['zodiac']
            if zodiac in last_occurrence:
                interval = idx - last_occurrence[zodiac]
                patterns['intervals'][zodiac].append(interval)
            last_occurrence[zodiac] = idx
        
        # 计算平均间隔
        for zodiac, intervals in patterns['intervals'].items():
            if intervals:
                patterns['intervals'][zodiac] = sum(intervals) / len(intervals)
        
        # 3. 节日效应模式
        festival_zodiacs = defaultdict(lambda: defaultdict(int))
        for idx, row in self.df.iterrows():
            if row['festival'] != "无":
                festival_zodiacs[row['festival']][row['zodiac']] += 1
        
        # 找出每个节日出现频率最高的生肖
        for festival, zodiac_counts in festival_zodiacs.items():
            if zodiac_counts:
                most_common = max(zodiac_counts.items(), key=lambda x: x[1])
                patterns['festival_boost'][festival] = most_common[0]
        
        # 修复的打印语句 - 移除了多余的括号
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
        """严格回测预测策略（基于最近200期） - 修复版本"""
        if self.df.empty:
            print("无法回测 - 数据为空")
            return pd.DataFrame(), 0.0
        
        if len(self.df) < BACKTEST_WINDOW + 1:  # 需要至少201期数据
            print(f"警告：数据不足{BACKTEST_WINDOW + 1}期，实际只有{len(self.df)}期")
            return pd.DataFrame(), 0.0
        
        print(f"开始回测策略（固定{BACKTEST_WINDOW}期窗口）...")
        results = []
        
        # 使用固定窗口大小的滑动窗口回测
        for i in range(BACKTEST_WINDOW, len(self.df) - 1):
            # 获取训练数据 (固定200期窗口)
            train = self.df.iloc[i - BACKTEST_WINDOW:i].copy()
            
            # 获取测试数据 (下一期)
            test = self.df.iloc[i + 1:i + 2]
            
            # 创建分析器实例并传入训练数据
            analyzer = LotteryAnalyzer(df=train)
            
            # 进行预测
            prediction = analyzer.predict_next()
            actual = test['zodiac'].values[0]
            predicted_zodiacs = prediction.get('prediction', [])
            
            # 检查是否命中
            is_hit = 1 if actual in predicted_zodiacs else 0
            
            # 记录结果
            results.append({
                '期号': test['expect'].values[0],
                '日期': test['date'].values[0],
                '上期生肖': train.iloc[-1]['zodiac'],
                '实际生肖': actual,
                '预测生肖': ", ".join(predicted_zodiacs) if predicted_zodiacs else "无预测",
                '是否命中': is_hit
            })
            
            # 记录错误日志
            if not is_hit:
                error_data = {
                    'draw_number': test['expect'].values[0],
                    'date': test['date'].values[0],
                    'actual_zodiac': actual,
                    'predicted_zodiacs': ",".join(predicted_zodiacs) if predicted_zodiacs else "无预测",
                    'last_zodiac': train.iloc[-1]['zodiac'],
                    'weekday': test['weekday'].values[0],
                    'month': test['month'].values[0],
                    'lunar_month': test['lunar'].apply(lambda x: x.month).values[0] if 'lunar' in test else None
                }
                log_error(error_data)
        
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
        festival = self.detect_festival(target_date)
        
        # 1. 节日效应增强
        if festival in self.patterns['festival_boost']:
            boost_zodiac = self.patterns['festival_boost'][festival]
            if boost_zodiac not in prediction:
                # 如果节日生肖不在预测中，替换掉得分最低的生肖
                prediction = prediction[:-1] + [boost_zodiac]
                print(f"节日效应增强: {festival}节日常见生肖 {boost_zodiac} 加入预测")
        
        # 2. 间隔模式增强
        for zodiac in prediction:
            if zodiac in self.patterns['intervals']:
                avg_interval = self.patterns['intervals'][zodiac]
                last_idx = data[data['zodiac'] == zodiac].index[-1] if not data.empty else -1
                current_interval = len(data) - last_idx
                
                # 如果接近平均间隔，提升优先级
                if current_interval >= avg_interval * 0.9:
                    if zodiac in prediction:
                        prediction.remove(zodiac)
                        prediction.insert(0, zodiac)
                    print(f"间隔模式增强: {zodiac} 已间隔 {current_interval}期 (平均 {avg_interval:.1f}期), 提升优先级")
        
        # 3. 连续出现模式处理
        if last_zodiac in self.patterns['consecutive']:
            max_consecutive = self.patterns['consecutive'][last_zodiac]
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
        
        # 应用模式增强
        prediction = self.apply_pattern_enhancement(prediction, last_zodiac, target_date, recent)
        
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
        ===== 每日彩报 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====
        
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
        
        历史模式：
        - 最长连续出现: {', '.join([f'{z}({c})' for z, c in self.patterns.get('consecutive', {}).items()])}
        - 节日效应: {', '.join([f'{f}→{z}' for f, z in self.patterns.get('festival_boost', {}).items()])}
        
        生肖频率分析（全部历史数据）：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if 'frequency' in analysis else "无数据"}
        
        最新开奖生肖分析：
        - 生肖: {last_zodiac}
        - 出现后下期最可能出现的4个生肖: {", ".join(transition_details.get(last_zodiac, ["无数据"]))}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * BACKTEST_WINDOW)}次
        
        下期预测：
        - 预测期号：{prediction['next_number']}
        - 推荐生肖：{", ".join(prediction['prediction'])}
        - 预测依据：综合历史模式与多因素分析
        
        =============================================
        """
        print("分析报告生成完成")
        return report
