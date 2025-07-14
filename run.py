import os
import sys
from datetime import datetime

# 确保正确导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis import LotteryAnalyzer
from src.utils import send_dingtalk, send_email
from src.data_processor import create_features  # 新增导入

def main():
    print(f"\n===== 开始执行分析 [{datetime.now().strftime('%Y-%m-%d %H:%M')}] =====")
    
    try:
        # 1. 初始化分析器（自动获取数据）
        print("\n[阶段1] 初始化分析器...")
        analyzer = LotteryAnalyzer()
        
        # 2. 执行特征工程（新增步骤）
        print("\n[阶段2] 执行特征工程...")
        analyzer.df = create_features(analyzer.df)
        
        # 3. 生成分析报告（包含回测）
        print("\n[阶段3] 生成分析报告...")
        report = analyzer.generate_report()
        
        print("\n分析报告内容:")
        print(report)
        
        # 4. 发送通知
        print("\n[阶段4] 发送通知...")
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            print("发送钉钉通知...")
            send_dingtalk(report, ding_webhook)
        
        if email_receiver:
            print("发送邮件通知...")
            send_email("每日彩票分析报告", report, email_receiver)
            
    except Exception as e:
        error_msg = f"分析过程中出错: {str(e)}"
        print(error_msg)
        # 尝试发送错误通知
        if os.getenv("DINGTALK_WEBHOOK"):
            send_dingtalk(error_msg, os.getenv("DINGTALK_WEBHOOK"))
        if os.getenv("EMAIL_RECEIVER"):
            send_email("彩票分析系统错误", error_msg, os.getenv("EMAIL_RECEIVER"))
        sys.exit(1)
    
    print("\n===== 分析完成 =====")

if __name__ == "__main__":
    main()
