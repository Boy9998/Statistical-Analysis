[file name]: src/monitor.py
[file content begin]
import pandas as pd
import numpy as np
import time
import os
import json
import logging
import psutil
from datetime import datetime
from config import ML_MODEL_PATH, MONITOR_LOG_PATH

# 确保日志目录存在
os.makedirs(os.path.dirname(MONITOR_LOG_PATH), exist_ok=True)

# 配置监控日志
monitor_logger = logging.getLogger('Monitor')
monitor_logger.setLevel(logging.INFO)
handler = logging.FileHandler(MONITOR_LOG_PATH)
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
monitor_logger.addHandler(handler)

class SystemMonitor:
    def __init__(self):
        """初始化实时监控系统"""
        self.feature_history = []
        self.performance_history = []
        self.error_patterns = {}
        self.last_check_time = datetime.now()
        self.start_time = datetime.now()
        
        # 创建监控数据存储
        os.makedirs('monitoring_data', exist_ok=True)
        
        monitor_logger.info("===== 监控系统初始化 =====")
    
    def monitor_features(self, df):
        """
        监控特征稳定性
        包括特征维度、类型和值范围
        """
        try:
            current_time = datetime.now()
            feature_report = {
                'timestamp': current_time.isoformat(),
                'feature_count': len(df.columns),
                'feature_names': sorted(df.columns.tolist()),
                'dtypes': {col: str(df[col].dtype) for col in df.columns},
                'missing_values': df.isnull().sum().to_dict(),
                'feature_ranges': {}
            }
            
            # 计算数值特征的统计信息
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                feature_report['feature_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
            
            # 保存特征报告
            self.feature_history.append(feature_report)
            
            # 检查特征维度变化
            if len(self.feature_history) > 1:
                prev = self.feature_history[-2]
                if prev['feature_count'] != feature_report['feature_count']:
                    msg = f"特征数量变化: {prev['feature_count']} -> {feature_report['feature_count']}"
                    monitor_logger.warning(msg)
                    self.alert_admin("特征维度变化", msg)
            
            # 保存到文件
            with open(f"monitoring_data/features_{current_time.strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(self.feature_history, f, indent=2)
                
            return feature_report
        except Exception as e:
            monitor_logger.error(f"特征监控失败: {e}")
            return {}
    
    def monitor_performance(self, accuracy, prediction, actual):
        """
        监控模型性能
        包括准确率趋势和预测质量
        """
        try:
            current_time = datetime.now()
            performance_report = {
                'timestamp': current_time.isoformat(),
                'accuracy': accuracy,
                'prediction': prediction,
                'actual': actual,
                'is_correct': actual in prediction,
                'prediction_rank': prediction.index(actual) + 1 if actual in prediction else -1
            }
            
            # 保存性能报告
            self.performance_history.append(performance_report)
            
            # 检查准确率趋势
            if len(self.performance_history) > 10:
                last_10 = [p['accuracy'] for p in self.performance_history[-10:]]
                avg_accuracy = np.mean(last_10)
                if avg_accuracy < 0.5:  # 低于50%准确率
                    msg = f"近期准确率偏低: {avg_accuracy:.2%} (最近10次)"
                    monitor_logger.warning(msg)
                    self.alert_admin("准确率下降", msg)
            
            # 保存到文件
            with open(f"monitoring_data/performance_{current_time.strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(self.performance_history, f, indent=2)
                
            return performance_report
        except Exception as e:
            monitor_logger.error(f"性能监控失败: {e}")
            return {}
    
    def monitor_errors(self, last_zodiac, predicted, actual):
        """
        监控错误预测模式
        跟踪高频错误转移模式
        """
        try:
            key = f"{last_zodiac}->{actual}"
            
            # 更新错误计数
            if key not in self.error_patterns:
                self.error_patterns[key] = {
                    'count': 0,
                    'first_occurrence': datetime.now().isoformat(),
                    'last_occurrence': datetime.now().isoformat(),
                    'predicted': predicted
                }
            else:
                self.error_patterns[key]['count'] += 1
                self.error_patterns[key]['last_occurrence'] = datetime.now().isoformat()
            
            # 检查高频错误
            for pattern, info in self.error_patterns.items():
                if info['count'] >= 5:  # 相同错误出现5次以上
                    msg = f"高频错误模式: {pattern} (已出现{info['count']}次)"
                    monitor_logger.warning(msg)
                    self.alert_admin("高频错误", msg)
                    # 重置计数
                    info['count'] = 0
            
            # 保存到文件
            with open(f"monitoring_data/errors_{datetime.now().strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(self.error_patterns, f, indent=2)
                
            return self.error_patterns
        except Exception as e:
            monitor_logger.error(f"错误监控失败: {e}")
            return {}
    
    def monitor_system_health(self):
        """监控系统资源使用情况"""
        try:
            current_time = datetime.now()
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            process = psutil.Process(os.getpid())
            
            health_report = {
                'timestamp': current_time.isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_info.percent,
                'memory_used_gb': memory_info.used / (1024**3),
                'disk_percent': disk_usage.percent,
                'disk_free_gb': disk_usage.free / (1024**3),
                'process_memory_gb': process.memory_info().rss / (1024**3),
                'uptime_minutes': (datetime.now() - self.start_time).total_seconds() / 60
            }
            
            # 检查资源阈值
            if cpu_percent > 80:
                msg = f"CPU使用率过高: {cpu_percent}%"
                monitor_logger.warning(msg)
                self.alert_admin("CPU过载", msg)
                
            if memory_info.percent > 80:
                msg = f"内存使用率过高: {memory_info.percent}%"
                monitor_logger.warning(msg)
                self.alert_admin("内存不足", msg)
                
            if disk_usage.percent > 90:
                msg = f"磁盘空间不足: {disk_usage.percent}%"
                monitor_logger.warning(msg)
                self.alert_admin("磁盘空间不足", msg)
            
            # 保存到文件
            with open(f"monitoring_data/health_{current_time.strftime('%Y%m%d')}.json", 'w') as f:
                json.dump(health_report, f, indent=2)
                
            return health_report
        except Exception as e:
            monitor_logger.error(f"系统健康监控失败: {e}")
            return {}
    
    def alert_admin(self, alert_type, message):
        """
        发送管理员警报
        在实际系统中，这里可以集成邮件、短信或钉钉通知
        """
        # 在实际部署中，这里应该实现通知发送逻辑
        # 目前仅记录日志
        full_message = f"[{alert_type}] {message}"
        monitor_logger.warning(f"管理员警报: {full_message}")
        
        # 这里可以添加实际的通知发送代码
        # 例如：send_email(admin_email, "系统警报", full_message)
    
    def generate_daily_report(self):
        """生成每日监控报告"""
        try:
            report_date = datetime.now().strftime('%Y-%m-%d')
            report = {
                'date': report_date,
                'feature_stability': self._get_feature_stability_report(),
                'performance_summary': self._get_performance_summary(),
                'error_patterns': self._get_top_errors(),
                'system_health': self._get_system_health_summary(),
                'recommendations': []
            }
            
            # 添加建议
            if report['performance_summary']['last_10_accuracy'] < 0.55:
                report['recommendations'].append("模型准确率下降，建议重新训练模型")
                
            if report['system_health']['max_cpu'] > 80:
                report['recommendations'].append("CPU使用率过高，建议优化代码或升级服务器")
                
            if report['error_patterns']:
                top_error = report['error_patterns'][0]
                report['recommendations'].append(
                    f"高频错误模式: {top_error['pattern']} (出现{top_error['count']}次)，建议调整策略权重"
                )
            
            # 保存报告
            report_path = f"monitoring_data/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            monitor_logger.info(f"生成每日监控报告: {report_path}")
            return report
        except Exception as e:
            monitor_logger.error(f"生成报告失败: {e}")
            return {}
    
    def _get_feature_stability_report(self):
        """获取特征稳定性摘要"""
        if not self.feature_history:
            return {}
        
        last_feature = self.feature_history[-1]
        return {
            'feature_count': last_feature['feature_count'],
            'feature_change_count': len(self.feature_history) - 1,
            'last_feature_names': last_feature['feature_names'][:5]  # 只显示前5个
        }
    
    def _get_performance_summary(self):
        """获取性能摘要"""
        if not self.performance_history:
            return {}
        
        last_10 = self.performance_history[-10:]
        accuracies = [p['accuracy'] for p in last_10]
        correct_predictions = sum(1 for p in last_10 if p['is_correct'])
        
        return {
            'last_accuracy': self.performance_history[-1]['accuracy'],
            'last_10_accuracy': np.mean(accuracies),
            'last_10_correct': correct_predictions,
            'avg_prediction_rank': np.mean([p['prediction_rank'] for p in last_10 if p['prediction_rank'] > 0])
        }
    
    def _get_top_errors(self, top_n=3):
        """获取高频错误"""
        if not self.error_patterns:
            return []
        
        sorted_errors = sorted(
            [(k, v) for k, v in self.error_patterns.items()],
            key=lambda x: x[1]['count'],
            reverse=True
        )[:top_n]
        
        return [{
            'pattern': k,
            'count': v['count'],
            'first_occurrence': v['first_occurrence'],
            'last_occurrence': v['last_occurrence']
        } for k, v in sorted_errors]
    
    def _get_system_health_summary(self):
        """获取系统健康摘要"""
        if not hasattr(self, 'health_history'):
            return {}
        
        last_health = self.health_history[-1] if self.health_history else {}
        return {
            'avg_cpu': np.mean([h.get('cpu_percent', 0) for h in self.health_history]),
            'max_cpu': max([h.get('cpu_percent', 0) for h in self.health_history]),
            'avg_memory': np.mean([h.get('memory_percent', 0) for h in self.health_history]),
            'uptime_hours': last_health.get('uptime_minutes', 0) / 60
        }

# 全局监控实例
system_monitor = SystemMonitor()

# 集成点：在预测流程中添加监控
def monitor_prediction(features, prediction, actual, last_zodiac):
    """监控预测流程"""
    # 监控特征
    feature_df = pd.DataFrame([features])
    system_monitor.monitor_features(feature_df)
    
    # 监控性能
    accuracy = 1.0 if actual in prediction else 0.0
    system_monitor.monitor_performance(accuracy, prediction, actual)
    
    # 监控错误
    if actual not in prediction:
        system_monitor.monitor_errors(last_zodiac, prediction, actual)
    
    # 监控系统健康
    system_monitor.monitor_system_health()

# 测试函数
if __name__ == "__main__":
    print("===== 测试监控系统 =====")
    
    # 测试特征监控
    test_df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['a', 'b', 'c']
    })
    print("特征监控:", system_monitor.monitor_features(test_df))
    
    # 测试性能监控
    print("性能监控:", system_monitor.monitor_performance(0.85, ['鼠', '牛', '虎'], '牛'))
    
    # 测试错误监控
    print("错误监控:", system_monitor.monitor_errors('鼠', ['牛', '虎', '兔'], '龙'))
    
    # 测试系统健康监控
    print("系统健康监控:", system_monitor.monitor_system_health())
    
    # 测试报告生成
    print("每日报告:", system_monitor.generate_daily_report())
    
    print("\n===== 监控系统测试完成 =====")
[file content end]
