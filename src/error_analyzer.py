import pandas as pd
import numpy as np
from collections import defaultdict
import re
from datetime import datetime

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
    
    # 错误类型到策略的映射关系
    ERROR_STRATEGY_MAP = {
        'FESTIVAL': {
            'action': 'INCREASE_WEIGHT',
            'target': 'festival_factor',
            'priority': 1
        },
        'CONSECUTIVE': {
            'action': 'SUPPRESS_ZODIAC',
            'target': 'consecutive_suppress_factor',
            'priority': 1
        },
        'SEASONAL': {
            'action': 'ADJUST_FEATURE',
            'target': 'seasonal_adjustment',
            'priority': 2
        },
        'ROLLING_WINDOW': {
            'action': 'RECALCULATE_FEATURE',
            'target': 'rolling_window_features',
            'priority': 1
        },
        'TRANSITION': {
            'action': 'ADJUST_PROBABILITY',
            'target': 'transition_probabilities',
            'priority': 1
        },
        'ZODIAC_FREQ': {
            'action': 'CALIBRATE_FREQUENCY',
            'target': 'zodiac_frequency_model',
            'priority': 2
        }
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
        self.zodiac_freq_errors = defaultdict(int)
    
    def _load_error_data(self):
        """加载错误日志数据，确保包含所有必要字段"""
        try:
            df = pd.read_csv(self.error_log_path)
            
            # 确保必要的列存在（新增festival和season字段）
            required_columns = [
                'date', 'actual_zodiac', 'predicted_zodiacs', 
                'last_zodiac', 'festival', 'season'
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    # 创建缺失字段并填充默认值
                    if col == 'festival':
                        df['festival'] = '无'
                    elif col == 'season':
                        # 根据日期推断季节
                        df['date'] = pd.to_datetime(df['date'])
                        df['season'] = df['date'].apply(self._infer_season)
                    else:
                        raise ValueError(f"错误日志缺少必要列: {col}")
            
            # 转换日期列为datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            print(f"成功加载错误日志: {len(df)} 条错误记录")
            return df
        except Exception as e:
            print(f"加载错误日志失败: {e}")
            return pd.DataFrame()
    
    def _infer_season(self, date):
        """根据日期推断季节（用于处理缺失的季节数据）"""
        month = date.month
        if 3 <= month <= 5:
            return '春'
        elif 6 <= month <= 8:
            return '夏'
        elif 9 <= month <= 11:
            return '秋'
        else:
            return '冬'
    
    def analyze(self):
        """执行完整错误分析流程"""
        if self.df.empty:
            print("警告: 没有可分析的错误数据")
            return []
        
        print("开始错误分析...")
        start_time = datetime.now()
        
        # 错误分类
        self._classify_errors()
        
        # 生成策略建议
        suggestions = self._generate_suggestions()
        
        # 打印分析摘要
        self._print_summary()
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"分析完成，耗时: {duration:.2f}秒")
        
        return suggestions
    
    def _classify_errors(self):
        """对错误进行分类和统计"""
        # 重置统计
        self.error_distribution.clear()
        self.error_patterns.clear()
        self.zodiac_freq_errors.clear()
        self.strategy_suggestions = []
        
        # 1. 节日相关错误
        festival_errors = self.df[self.df['festival'] != '无']
        if not festival_errors.empty:
            self.error_distribution['FESTIVAL'] = len(festival_errors)
        
        # 2. 季节相关错误
        seasonal_errors = self.df.groupby('season').size()
        for season, count in seasonal_errors.items():
            if count > 5:  # 只处理显著的季节错误
                self.error_distribution['SEASONAL'] += count
        
        # 3. 连续出现错误
        consecutive_errors = self.df.apply(
            lambda row: row['last_zodiac'] == row['actual_zodiac'], axis=1
        ).sum()
        if consecutive_errors > 0:
            self.error_distribution['CONSECUTIVE'] = consecutive_errors
        
        # 4. 滚动窗口特征错误
        rolling_errors = self.df[self.df.apply(
            lambda row: self._is_rolling_feature_error(row), axis=1
        )]
        if not rolling_errors.empty:
            self.error_distribution['ROLLING_WINDOW'] = len(rolling_errors)
        
        # 5. 转移概率错误
        transition_errors = self.df[self.df.apply(
            lambda row: self._is_transition_error(row), axis=1
        )]
        if not transition_errors.empty:
            self.error_distribution['TRANSITION'] = len(transition_errors)
        
        # 6. 生肖频率偏差
        self._detect_zodiac_freq_errors()
        
        # 7. 记录错误模式
        for _, row in self.df.iterrows():
            last_zodiac = row['last_zodiac']
            actual_zodiac = row['actual_zodiac']
            self.error_patterns[last_zodiac][actual_zodiac] += 1
    
    def _detect_zodiac_freq_errors(self):
        """检测生肖频率偏差错误"""
        # 计算每个生肖的实际出现频率
        actual_counts = self.df['actual_zodiac'].value_counts().to_dict()
        
        # 计算每个生肖的预测频率
        all_predicted = []
        for preds in self.df['predicted_zodiacs']:
            all_predicted.extend(preds.split(','))
        predicted_counts = pd.Series(all_predicted).value_counts().to_dict()
        
        # 检测显著偏差（>15%）
        for zodiac in actual_counts:
            actual_freq = actual_counts.get(zodiac, 0) / len(self.df)
            predicted_freq = predicted_counts.get(zodiac, 0) / len(self.df)
            
            if abs(actual_freq - predicted_freq) > 0.15:
                self.zodiac_freq_errors[zodiac] = {
                    'actual': actual_freq,
                    'predicted': predicted_freq,
                    'deviation': abs(actual_freq - predicted_freq)
                }
        
        if self.zodiac_freq_errors:
            self.error_distribution['ZODIAC_FREQ'] = len(self.zodiac_freq_errors)
    
    def _is_rolling_feature_error(self, row):
        """检查是否是滚动窗口特征错误"""
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
        """生成策略优化建议，基于错误频率动态计算调整幅度"""
        for error_type, count in self.error_distribution.items():
            if error_type in self.ERROR_STRATEGY_MAP:
                strategy = self.ERROR_STRATEGY_MAP[error_type]
                
                # 动态计算调整值：min(0.1, 错误次数×0.015)
                adjust_value = min(0.1, count * 0.015) if strategy['action'] != 'RECALCULATE_FEATURE' else None
                
                # 特殊处理生肖频率偏差
                if error_type == 'ZODIAC_FREQ':
                    zodiacs = ", ".join(self.zodiac_freq_errors.keys())
                    reason = f"{len(self.zodiac_freq_errors)}个生肖频率偏差显著: {zodiacs}"
                else:
                    reason = f"{self.ERROR_TYPES[error_type]}发生{count}次"
                
                suggestion = {
                    'type': self.ERROR_TYPES[error_type],
                    'action': strategy['action'],
                    'target': strategy['target'],
                    'value': adjust_value,
                    'reason': reason,
                    'priority': strategy['priority']
                }
                self.strategy_suggestions.append(suggestion)
        
        # 按优先级排序
        sorted_suggestions = sorted(
            self.strategy_suggestions, 
            key=lambda x: (x['priority'], -self.error_distribution.get(x['type'], 0))
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
                name = self.ERROR_TYPES.get(error_type, error_type)
                print(f"- {name}: {count} 次 ({count/len(self.df)*100:.1f}%)")
        
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
        
        if self.zodiac_freq_errors:
            print("\n生肖频率偏差:")
            for zodiac, data in self.zodiac_freq_errors.items():
                print(f"- {zodiac}: 实际{data['actual']:.2f} vs 预测{data['predicted']:.2f} (偏差{data['deviation']:.2f})")
        
        if self.strategy_suggestions:
            print("\n生成的优化建议:")
            for i, suggestion in enumerate(self.strategy_suggestions, 1):
                action_map = {
                    'INCREASE_WEIGHT': "增加权重",
                    'ADJUST_FEATURE': "调整特征",
                    'SUPPRESS_ZODIAC': "抑制生肖",
                    'RECALCULATE_FEATURE': "重新计算特征",
                    'ADJUST_PROBABILITY': "调整概率",
                    'CALIBRATE_FREQUENCY': "校准频率"
                }
                action = action_map.get(suggestion['action'], suggestion['action'])
                value = f"{suggestion['value']:.3f}" if suggestion['value'] is not None else "N/A"
                print(f"{i}. [{suggestion['type']}] {action}: {suggestion['target']} (值: {value}, 原因: {suggestion['reason']})")
    
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
                'zodiac_freq_errors': dict(self.zodiac_freq_errors),
                'strategy_suggestions': self.strategy_suggestions
            }
            
            report_df = pd.DataFrame({
                'metric': list(report.keys()),
                'value': list(report.values())
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
