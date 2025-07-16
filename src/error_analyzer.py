import pandas as pd
import numpy as np
from collections import defaultdict
import re

class ErrorAnalyzer:
    """错误分析系统，用于自动分类预测错误并提供优化建议"""
    
    ERROR_TYPES = {
        'FESTIVAL': '节日预测错误',
        'CONSECUTIVE': '连续出现错误',
        'SEASONAL': '季节特征失效',
        'ROLLING_WINDOW': '滚动窗口特征失效',
        'TRANSITION': '转移概率错误',
        'ZODIAC_FREQ': '生肖频率偏差'
    }
    
    def __init__(self, error_log_path):
        """
        初始化错误分析器
        :param error_log_path: 错误日志文件路径
        """
        self.error_log_path = error_log_path
        self.df = self._load_error_data()
        self.strategy_suggestions = []
        self.error_distribution = defaultdict(int)
        self.error_patterns = defaultdict(lambda: defaultdict(int))
    
    def _load_error_data(self):
        """加载错误日志数据"""
        try:
            df = pd.read_csv(self.error_log_path)
            
            # 确保必要的列存在
            required_columns = ['date', 'actual_zodiac', 'predicted_zodiacs', 'last_zodiac']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"错误日志缺少必要列: {col}")
            
            # 转换日期列为datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            print(f"成功加载错误日志: {len(df)} 条错误记录")
            return df
        except Exception as e:
            print(f"加载错误日志失败: {e}")
            return pd.DataFrame()
    
    def analyze(self):
        """执行完整错误分析流程"""
        if self.df.empty:
            print("警告: 没有可分析的错误数据")
            return []
        
        print("开始错误分析...")
        
        # 错误分类
        self._classify_errors()
        
        # 生成策略建议
        suggestions = self._generate_suggestions()
        
        # 打印分析摘要
        self._print_summary()
        
        return suggestions
    
    def _classify_errors(self):
        """对错误进行分类"""
        # 1. 节日相关错误
        festival_errors = self.df[self.df['festival'] != '无']
        if not festival_errors.empty:
            self.error_distribution[self.ERROR_TYPES['FESTIVAL']] = len(festival_errors)
            self.strategy_suggestions.append({
                'type': self.ERROR_TYPES['FESTIVAL'],
                'action': 'INCREASE_WEIGHT',
                'target': 'festival_factor',
                'value': 0.05,
                'reason': f"在节日期间发生 {len(festival_errors)} 次错误",
                'priority': 1
            })
        
        # 2. 季节相关错误
        seasonal_errors = self.df.groupby('season').size()
        for season, count in seasonal_errors.items():
            if count > 5:  # 只处理显著的季节错误
                self.error_distribution[self.ERROR_TYPES['SEASONAL']] += count
                self.strategy_suggestions.append({
                    'type': self.ERROR_TYPES['SEASONAL'],
                    'action': 'ADJUST_FEATURE',
                    'target': f'season_{season}',
                    'value': 0.1,
                    'reason': f"在{season}季发生 {count} 次错误",
                    'priority': 2
                })
        
        # 3. 连续出现错误
        consecutive_threshold = 3  # 连续出现3次以上的生肖
        for idx, row in self.df.iterrows():
            last_zodiac = row['last_zodiac']
            actual_zodiac = row['actual_zodiac']
            
            # 记录错误模式
            self.error_patterns[last_zodiac][actual_zodiac] += 1
            
            # 检查是否连续出现
            if last_zodiac == actual_zodiac:
                self.error_distribution[self.ERROR_TYPES['CONSECUTIVE']] += 1
                self.strategy_suggestions.append({
                    'type': self.ERROR_TYPES['CONSECUTIVE'],
                    'action': 'SUPPRESS_ZODIAC',
                    'target': actual_zodiac,
                    'value': 0.15,
                    'reason': f"{actual_zodiac}连续出现时预测错误",
                    'priority': 1
                })
        
        # 4. 滚动窗口特征错误
        rolling_errors = self.df[self.df.apply(
            lambda row: self._is_rolling_feature_error(row), axis=1
        )]
        if not rolling_errors.empty:
            count = len(rolling_errors)
            self.error_distribution[self.ERROR_TYPES['ROLLING_WINDOW']] = count
            self.strategy_suggestions.append({
                'type': self.ERROR_TYPES['ROLLING_WINDOW'],
                'action': 'RECALCULATE_FEATURE',
                'target': 'rolling_features',
                'value': None,
                'reason': f"滚动窗口特征失效导致 {count} 次错误",
                'priority': 1
            })
        
        # 5. 转移概率错误
        transition_errors = self.df[self.df.apply(
            lambda row: self._is_transition_error(row), axis=1
        )]
        if not transition_errors.empty:
            count = len(transition_errors)
            self.error_distribution[self.ERROR_TYPES['TRANSITION']] = count
            self.strategy_suggestions.append({
                'type': self.ERROR_TYPES['TRANSITION'],
                'action': 'ADJUST_PROBABILITY',
                'target': 'combo_probs',
                'value': 0.07,
                'reason': f"转移概率不准确导致 {count} 次错误",
                'priority': 1
            })
    
    def _is_rolling_feature_error(self, row):
        """检查是否是滚动窗口特征错误"""
        # 这里简化处理，实际应根据具体特征计算
        last_zodiac = row['last_zodiac']
        predicted = row['predicted_zodiacs'].split(',')
        actual = row['actual_zodiac']
        
        # 如果实际生肖在近期高频出现但未被预测
        if last_zodiac == actual and actual not in predicted:
            return True
        
        return False
    
    def _is_transition_error(self, row):
        """检查是否是转移概率错误"""
        last_zodiac = row['last_zodiac']
        actual_zodiac = row['actual_zodiac']
        predicted = row['predicted_zodiacs'].split(',')
        
        # 如果实际生肖是上期生肖的常见转移但未被预测
        common_transitions = self._get_common_transitions(last_zodiac)
        if actual_zodiac in common_transitions and actual_zodiac not in predicted:
            return True
        
        return False
    
    def _get_common_transitions(self, zodiac):
        """获取常见转移生肖（简化实现）"""
        # 实际应用中应从历史数据计算
        transitions = {
            '鼠': ['牛', '龙', '猴'],
            '牛': ['鼠', '蛇', '鸡'],
            '虎': ['猪', '马', '狗'],
            '兔': ['狗', '羊', '猪'],
            '龙': ['鸡', '猴', '鼠'],
            '蛇': ['猴', '鸡', '牛'],
            '马': ['羊', '虎', '狗'],
            '羊': ['马', '兔', '猪'],
            '猴': ['蛇', '龙', '鼠'],
            '鸡': ['龙', '蛇', '牛'],
            '狗': ['兔', '虎', '马'],
            '猪': ['虎', '兔', '羊']
        }
        return transitions.get(zodiac, [])
    
    def _generate_suggestions(self):
        """生成策略优化建议"""
        if not self.strategy_suggestions:
            return []
        
        # 按优先级排序
        sorted_suggestions = sorted(
            self.strategy_suggestions, 
            key=lambda x: x['priority']
        )
        
        # 合并重复建议
        final_suggestions = []
        seen = set()
        
        for suggestion in sorted_suggestions:
            key = (suggestion['type'], suggestion['action'], suggestion['target'])
            if key not in seen:
                final_suggestions.append(suggestion)
                seen.add(key)
        
        print(f"生成 {len(final_suggestions)} 条策略优化建议")
        return final_suggestions
    
    def _print_summary(self):
        """打印错误分析摘要"""
        print("\n===== 错误分析摘要 =====")
        print(f"总错误数: {len(self.df)}")
        
        if self.error_distribution:
            print("\n错误类型分布:")
            for error_type, count in self.error_distribution.items():
                print(f"- {error_type}: {count} 次 ({count/len(self.df)*100:.1f}%)")
        
        if self.error_patterns:
            print("\n最常见错误模式:")
            pattern_list = []
            for last_z, patterns in self.error_patterns.items():
                for actual, count in patterns.items():
                    pattern_list.append((last_z, actual, count))
            
            # 取前5个最常见模式
            top_patterns = sorted(pattern_list, key=lambda x: x[2], reverse=True)[:5]
            for last_z, actual, count in top_patterns:
                print(f"- {last_z} -> {actual}: {count} 次")
        
        if self.strategy_suggestions:
            print("\n生成的优化建议:")
            for i, suggestion in enumerate(self.strategy_suggestions, 1):
                action_map = {
                    'INCREASE_WEIGHT': "增加权重",
                    'ADJUST_FEATURE': "调整特征",
                    'SUPPRESS_ZODIAC': "抑制生肖",
                    'RECALCULATE_FEATURE': "重新计算特征",
                    'ADJUST_PROBABILITY': "调整概率"
                }
                action = action_map.get(suggestion['action'], suggestion['action'])
                print(f"{i}. [{suggestion['type']}] {action}: {suggestion['target']} ({suggestion['reason']})")
    
    def get_error_patterns(self):
        """获取错误模式数据"""
        return self.error_patterns
    
    def get_top_error_patterns(self, n=5):
        """获取最常见的n个错误模式"""
        pattern_list = []
        for last_z, patterns in self.error_patterns.items():
            for actual, count in patterns.items():
                pattern_list.append({
                    'last_zodiac': last_z,
                    'actual_zodiac': actual,
                    'count': count
                })
        
        sorted_patterns = sorted(pattern_list, key=lambda x: x['count'], reverse=True)
        return sorted_patterns[:n]
    
    def save_analysis_report(self, output_path):
        """保存分析报告"""
        if self.df.empty:
            print("无分析报告可保存")
            return False
        
        try:
            report = {
                'total_errors': len(self.df),
                'error_distribution': dict(self.error_distribution),
                'top_error_patterns': self.get_top_error_patterns(),
                'strategy_suggestions': self.strategy_suggestions
            }
            
            report_df = pd.DataFrame({
                '指标': list(report.keys()),
                '值': list(report.values())
            })
            
            report_df.to_csv(output_path, index=False)
            print(f"分析报告已保存至: {output_path}")
            return True
        except Exception as e:
            print(f"保存分析报告失败: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    print("===== 错误分析系统测试 =====")
    
    # 创建测试数据
    test_data = {
        'date': ['2023-08-01', '2023-08-02', '2023-08-03', '2023-08-04', '2023-08-05'],
        'actual_zodiac': ['鼠', '牛', '虎', '兔', '龙'],
        'predicted_zodiacs': ['牛,虎', '虎,兔', '兔,龙', '龙,蛇', '蛇,马'],
        'last_zodiac': ['猪', '鼠', '牛', '虎', '兔'],
        'festival': ['无', '春节', '无', '无', '元宵'],
        'season': ['夏', '夏', '秋', '秋', '秋']
    }
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('test_error_log.csv', index=False)
    
    # 实例化分析器
    analyzer = ErrorAnalyzer('test_error_log.csv')
    
    # 执行分析
    suggestions = analyzer.analyze()
    
    # 保存报告
    analyzer.save_analysis_report('test_error_analysis.csv')
    
    print("\n测试完成")
