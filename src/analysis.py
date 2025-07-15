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
            self.df['is_festival'] = self.df['festival'] != "无"
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
            
            # 检测历史模式
            print("检测历史模式...")
            self.patterns = self.detect_patterns()
            
            # 更新组合概率和特征重要性
            self.strategy_manager.update_combo_probs(self.df)
            self.strategy_manager.evaluate_feature_importance(self.df)
            
            # 打印最新开奖信息
            latest = self.df.iloc[-1]
            print(f"最新开奖记录: 期号 {latest['expect']}, 日期 {latest['date'].date()}, 生肖 {latest['zodiac']}")
        else:
            print("警告：未获取到任何有效数据")
            self.strategy_manager = StrategyManager()
            self.patterns = {}
    
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
            season_data = self.df[self.df['season'] == season]
            if not season_data.empty:
                season_counts = season_data['zodiac'].value_counts(normalize=True)
                for zodiac in self.zodiacs:
                    # 更新季节特征值
                    self.df.loc[self.df['season'] == season, f'season_{zodiac}'] = season_counts.get(zodiac, 0.0)
        
        # 3. 节日生肖特征
        for zodiac in self.zodiacs:
            # 为每个生肖创建节日特征列
            self.df[f'festival_{zodiac}'] = 0.0
        
        # 计算节日中每个生肖的频率
        festival_data = self.df[self.df['is_festival']]
        if not festival_data.empty:
            festival_counts = festival_data['zodiac'].value_counts(normalize=True)
            for zodiac in self.zodiacs:
                # 更新节日特征值
                self.df.loc[self.df['is_festival'], f'festival_{zodiac}'] = festival_counts.get(zodiac, 0.0)
        
        print(f"已添加生肖特征: {len(self.zodiacs)*3}个新特征")
    
    def add_rolling_features(self):
        """添加滚动窗口特征 - 关键修复"""
        print("添加滚动窗口特征...")
        windows = [7, 30]  # 7天和30天窗口
        
        # 创建生肖出现标志
        for zodiac in self.zodiacs:
            self.df[f'occur_{zodiac}'] = (self.df['zodiac'] == zodiac).astype(int)
        
        for window in windows:
            # 为每个生肖添加滚动窗口特征
            for zodiac in self.zodiacs:
                # 计算滚动频率 - 使用expanding min_periods确保有足够数据
                self.df[f'rolling_{window}d_{zodiac}'] = (
                    self.df[f'occur_{zodiac}'].rolling(window=window, min_periods=1).mean()
                )
        
        print(f"已添加滚动窗口特征: {len(windows)*len(self.zodiacs)}个新特征")
    
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
        
        # 计算平均间隔
        for zodiac, intervals in patterns['intervals'].items():
            if intervals:
                patterns['intervals'][zodiac] = sum(intervals) / len(intervals)
        
        # 3. 节日效应模式
        festival_zodiacs = defaultdict(lambda: defaultdict(int))
        for idx, row in self.df.iterrows():
            if row['is_festival']:
                festival_zodiacs[row['festival']][row['zodiac']] += 1
        
        # 找出每个节日出现频率最高的生肖
        for festival, zodiac_counts in festival_zodiacs.items():
            if zodiac_counts:
                most_common = max(zodiac_counts.items(), key=lambda x: x[1])
                patterns['festival_boost'][festival] = most_common[0]
        
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
        """严格回测预测策略（使用固定窗口滑动） - 性能优化"""
        if self.df.empty:
            print("无法回测 - 数据为空")
            return pd.DataFrame(), 0.0
        
        window = BACKTEST_WINDOW
        if len(self.df) < window + 1:
            print(f"警告：数据不足{window+1}期，实际只有{len(self.df)}期")
            return pd.DataFrame(), 0.0
        
        print(f"开始回测策略（固定窗口{window}期）...")
        results = []
        
        # 使用单个策略管理器处理整个回测过程 - 性能优化
        strategy_manager = StrategyManager()
        
        # 预先计算整个数据集的索引位置
        indices = range(window, len(self.df)-1)
        total = len(indices)
        
        for i, idx in enumerate(indices):
            if (i+1) % 50 == 0:
                print(f"回测进度: {i+1}/{total} ({((i+1)/total*100):.1f}%)")
            
            # 训练数据：从 i-window 到 i-1 (共window期)
            train = self.df.iloc[idx-window:idx].copy()
            
            # 为训练数据添加特征（关键修复）
            self.add_features_to_data(train)
            
            # 测试数据：下一期 (i)
            test = self.df.iloc[idx:idx+1]
            
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
            
            # 记录结果 - 修复预测生肖格式（去掉空格）
            is_hit = 1 if actual in prediction else 0
            results.append({
                '期号': test['expect'].values[0],
                '上期生肖': last_zodiac,
                '实际生肖': actual,
                '预测生肖': ",".join(prediction),  # 确保无空格
                '是否命中': is_hit
            })
            
            # 记录预测错误
            if not is_hit:
                error_data = {
                    'draw_number': test['expect'].values[0],
                    'date': test['date'].dt.strftime('%Y-%m-%d').values[0],
                    'actual_zodiac': actual,
                    'predicted_zodiacs': ",".join(prediction),
                    'last_zodiac': last_zodiac,
                    'weekday': test['weekday'].values[0] if 'weekday' in test.columns else None,
                    'month': test['month'].values[0] if 'month' in test.columns else None
                }
                log_error(error_data)
        
        result_df = pd.DataFrame(results)
        if not result_df.empty:
            # 转换期号为数值类型用于排序
            result_df['期号数值'] = result_df['期号'].astype(int)
            
            # 去重处理 - 保留每个期号的最新结果
            result_df = result_df.sort_values('期号数值').drop_duplicates(subset=['期号'], keep='last')
            
            # 按数值期号倒序排序
            result_df = result_df.sort_values('期号数值', ascending=False)
            
            accuracy = result_df['是否命中'].mean()
            hit_count = result_df['是否命中'].sum()
            print(f"回测完成: 准确率={accuracy:.2%}, 命中次数={hit_count}/{len(result_df)}")
            
            # 根据回测结果调整策略
            self.strategy_manager.adjust(accuracy)
        else:
            accuracy = 0.0
            print("回测完成: 无有效结果")
        
        return result_df, accuracy
    
    def add_features_to_data(self, df):
        """为数据添加必要的特征 - 完整实现"""
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
            season_data = df[df['season'] == season]
            if not season_data.empty:
                season_counts = season_data['zodiac'].value_counts(normalize=True)
                for zodiac in self.zodiacs:
                    # 更新季节特征值
                    df.loc[df['season'] == season, f'season_{zodiac}'] = season_counts.get(zodiac, 0.0)
        
        # 3. 添加节日生肖特征
        for zodiac in self.zodiacs:
            # 初始化节日特征列
            df[f'festival_{zodiac}'] = 0.0
        
        # 计算节日中每个生肖的频率
        festival_data = df[df['is_festival']]
        if not festival_data.empty:
            festival_counts = festival_data['zodiac'].value_counts(normalize=True)
            for zodiac in self.zodiacs:
                # 更新节日特征值
                df.loc[df['is_festival'], f'festival_{zodiac}'] = festival_counts.get(zodiac, 0.0)
    
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
        for zodiac in prediction.copy():  # 使用副本避免修改迭代中的列表
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
    
    def _generate_recent_table(self):
        """生成最近10期预测结果表 - 修复排序和重复问题"""
        if not hasattr(self, 'backtest_results') or self.backtest_results.empty:
            return "\n无历史预测数据"
        
        # 创建副本避免修改原始数据
        df = self.backtest_results.copy()
        
        # 确保期号为数值类型
        if not pd.api.types.is_numeric_dtype(df['期号']):
            df['期号数值'] = df['期号'].astype(int)
        else:
            df['期号数值'] = df['期号']
        
        # 按数值期号倒序排列获取真正最近10期
        recent = df.sort_values('期号数值', ascending=False).head(10)
        
        # 构建对齐的表格
        table = "\n期号      预测生肖            实际开奖    结果"
        table += "\n============================================"
        
        for _, row in recent.iterrows():
            mark = "✓" if row['是否命中'] else "✗"
            # 格式化对齐
            prediction_str = row['预测生肖']
            actual_str = row['实际生肖']
            table += f"\n{row['期号']}  {prediction_str:<15}  {actual_str:<5}  {mark}"
        
        return table
    
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
        
        # 回测结果 - 存储到实例变量
        self.backtest_results, accuracy = self.backtest_strategy()
        
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
        - 节日效应: {', '.join([f'{f}→{z}' for f, z in self.patterns.get('festival_boost', {}).items()])}
        
        生肖频率分析（全部历史数据）：
        {analysis.get('frequency', pd.DataFrame()).to_string(index=False) if 'frequency' in analysis else "无数据"}
        
        最新开奖生肖分析：
        - 生肖: {last_zodiac}
        - 出现后下期最可能出现的4个生肖: {", ".join(transition_details.get(last_zodiac, ["无数据"]))}
        
        回测结果（最近{BACKTEST_WINDOW}期）：
        - 准确率：{accuracy:.2%}
        - 命中次数：{int(accuracy * len(self.backtest_results)) if not self.backtest_results.empty else 0}/{len(self.backtest_results) if not self.backtest_results.empty else 0}
        
        === 最近10期预测结果追踪 ==={self._generate_recent_table()}
        
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
