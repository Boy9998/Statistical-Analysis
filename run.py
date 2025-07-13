import os
import sys
from datetime import datetime  # Added missing import
import traceback  # Added for better error handling

# 确保正确导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.analysis import LotteryAnalyzer
from src.utils import send_dingtalk, send_email, fetch_historical_data  # Added fetch_historical_data
from src.data_processor import (
    add_temporal_features, 
    add_lunar_features, 
    add_festival_features, 
    add_season_features
)

def main():
    print("=" * 50)
    print(f"生肖预测系统启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    try:
        # 1. 数据获取
        print("获取历史数据...")
        df = fetch_historical_data()
        
        if df.empty:
            raise ValueError("获取到的数据为空，请检查API连接")
        
        # 2. 数据预处理和特征工程
        print("添加时序特征...")
        df = add_temporal_features(df)
        
        print("添加农历特征...")
        df = add_lunar_features(df)
        
        print("添加节日特征...")
        df = add_festival_features(df)
        
        print("添加季节特征...")
        df = add_season_features(df)
        
        # 3. 创建分析器
        print("初始化分析器...")
        analyzer = LotteryAnalyzer()
        analyzer.df = df  # 传入处理后的数据
        
        # 4. 回测验证
        print("执行回测验证...")
        backtest_results, accuracy = analyzer.backtest_strategy()
        
        # 5. 策略调整
        print("调整策略权重...")
        analyzer.strategy_manager.adjust(accuracy)
        
        # 6. 预测下期
        print("生成预测...")
        prediction = analyzer.predict_next()
        
        # 7. 生成报告
        print("生成分析报告...")
        report = analyzer.generate_report()
        
        # 8. 发送通知
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            print("发送钉钉通知...")
            send_dingtalk(report, ding_webhook)
        
        if email_receiver:
            print("发送邮件通知...")
            send_email("每日彩票分析报告", report, email_receiver)
        
        print("分析完成，报告已发送")
        
    except Exception as e:
        error_msg = f"系统运行失败: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        
        # 记录错误日志
        from src.utils import log_error
        log_error({
            'error_type': '系统运行异常',
            'details': error_msg
        })
        
        sys.exit(1)
    
    sys.exit(0)

if __name__ == "__main__":
    main()
