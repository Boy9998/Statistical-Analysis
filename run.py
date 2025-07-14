import os
import sys
import traceback
from datetime import datetime

# 确保正确导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis import LotteryAnalyzer
from src.utils import send_dingtalk, send_email, log_error
from src.data_processor import add_temporal_features, add_lunar_features, add_festival_features, add_season_features

def main():
    try:
        print("=" * 50)
        print(f"开始执行分析任务 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
        # 创建分析器
        print("初始化分析器...")
        analyzer = LotteryAnalyzer()
        
        # 确保数据不为空
        if analyzer.df.empty:
            print("错误：未获取到有效数据，终止分析")
            return
        
        print(f"原始数据记录数: {len(analyzer.df)}")
        
        # 添加时序特征（关键优化）
        print("添加时序特征...")
        analyzer.df = add_temporal_features(analyzer.df)
        
        # 添加农历特征
        print("添加农历特征...")
        analyzer.df = add_lunar_features(analyzer.df)
        
        # 添加节日特征
        print("添加节日特征...")
        analyzer.df = add_festival_features(analyzer.df)
        
        # 添加季节特征
        print("添加季节特征...")
        analyzer.df = add_season_features(analyzer.df)
        
        print(f"特征工程后数据记录数: {len(analyzer.df)}")
        
        # 生成分析报告
        print("\n生成分析报告...")
        report = analyzer.generate_report()
        
        # 打印报告摘要
        print("\n报告摘要:")
        print(report.split("下期预测")[0].strip())
        
        # 发送通知
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            print("发送钉钉通知...")
            send_dingtalk(report, ding_webhook)
        
        if email_receiver:
            print("发送邮件通知...")
            send_email("每日彩票分析报告", report, email_receiver)
            
        print("=" * 50)
        print(f"分析任务完成 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
    except Exception as e:
        # 记录完整错误信息
        error_msg = f"分析任务失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        # 记录到错误日志
        log_error({
            'draw_number': 'SYSTEM_ERROR',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'actual_zodiac': 'N/A',
            'predicted_zodiacs': 'N/A',
            'last_zodiac': 'N/A',
            'weekday': datetime.now().weekday() + 1,
            'month': datetime.now().month
        })
        
        # 发送错误通知
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            send_dingtalk(f"分析任务失败: {str(e)}", ding_webhook)
        
        if email_receiver:
            send_email("彩票分析系统错误", error_msg, email_receiver)

if __name__ == "__main__":
    main()
