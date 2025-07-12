import os
import sys

# 添加src目录到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import send_dingtalk, send_email
from analysis import LotteryAnalyzer

def main():
    print("开始分析历史数据...")
    analyzer = LotteryAnalyzer()
    report = analyzer.generate_report()
    
    print("\n生成分析报告:")
    print(report)
    
    # 发送通知
    ding_webhook = os.getenv("DINGTALK_WEBHOOK")
    email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
    
    if ding_webhook:
        print("发送钉钉通知...")
        send_dingtalk(report, ding_webhook)
    
    if email_receiver:
        print("发送邮件通知...")
        send_email("每日彩票分析报告", report, email_receiver)

if __name__ == "__main__":
    main()
