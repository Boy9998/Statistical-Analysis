[file name]: run.py
[file content begin]
import os
import sys
import traceback
import numpy as np
from datetime import datetime
import subprocess
import importlib.util
import pkg_resources

# 确保正确导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

print(f"当前工作目录: {os.getcwd()}")
print(f"系统路径: {sys.path}")

# === 新增：依赖检查与安装功能 ===
def check_dependencies():
    """检查并安装缺失的依赖项"""
    # 定义依赖文件路径
    req_file = os.path.join(current_dir, "requirements.txt")
    
    # 检查依赖文件是否存在
    if not os.path.exists(req_file):
        print(f"错误: 依赖文件不存在 {req_file}")
        print("请确保requirements.txt文件存在")
        return False
    
    print("=" * 50)
    print("开始依赖检查...")
    print(f"使用依赖文件: {req_file}")
    
    try:
        # 读取依赖文件
        with open(req_file, 'r') as f:
            required_packages = [line.strip() for line in f.readlines() 
                                if line.strip() and not line.startswith('#')]
        
        # 检查已安装的包
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing_packages = []
        
        # 检查每个依赖
        for package in required_packages:
            # 解析包名和版本
            if '==' in package:
                pkg_name, required_version = package.split('==')
            else:
                pkg_name = package
                required_version = None
            
            # 检查是否已安装
            if pkg_name.lower() in installed_packages:
                if required_version:
                    installed_version = installed_packages[pkg_name.lower()]
                    if installed_version != required_version:
                        print(f"版本不匹配: {pkg_name} (需要 {required_version}, 已安装 {installed_version})")
                        missing_packages.append(package)
                else:
                    print(f"已安装: {pkg_name}")
            else:
                print(f"缺失: {package}")
                missing_packages.append(package)
        
        # 如果没有缺失，直接返回
        if not missing_packages:
            print("所有依赖已满足 ✓")
            print("=" * 50)
            return True
        
        print(f"\n检测到 {len(missing_packages)} 个缺失依赖")
        print("=" * 50)
        
        # 尝试安装缺失依赖
        try:
            print("尝试安装缺失依赖...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("依赖安装成功 ✓")
            print("=" * 50)
            print("请重新运行程序")
            return False  # 需要重启
        except subprocess.CalledProcessError as e:
            print(f"依赖安装失败: {e}")
            print("请手动执行以下命令安装依赖:")
            print(f"pip install -r {req_file}")
            return False
    
    except Exception as e:
        print(f"依赖检查失败: {str(e)}")
        print(traceback.format_exc())
        return False

# 在导入其他模块前检查依赖
if not check_dependencies():
    sys.exit(1)

try:
    # 尝试从 src 包导入
    from src.data_processor import (
        add_temporal_features, 
        add_lunar_features, 
        add_festival_features, 
        add_season_features,
        add_rolling_features
    )
    from src.analysis import LotteryAnalyzer
    from src.utils import send_dingtalk, send_email, log_error
    print("成功从 src 包导入模块")
except ImportError as e:
    print(f"从 src 包导入失败: {e}")
    try:
        # 尝试直接导入模块
        from data_processor import (
            add_temporal_features, 
            add_lunar_features, 
            add_festival_features, 
            add_season_features,
            add_rolling_features
        )
        from analysis import LotteryAnalyzer
        from utils import send_dingtalk, send_email, log_error
        print("成功直接导入模块")
    except ImportError as e:
        print(f"直接导入失败: {e}")
        try:
            # 尝试相对导入
            from .data_processor import (
                add_temporal_features, 
                add_lunar_features, 
                add_festival_features, 
                add_season_features,
                add_rolling_features
            )
            from .analysis import LotteryAnalyzer
            from .utils import send_dingtalk, send_email, log_error
            print("成功使用相对导入")
        except ImportError as e:
            print(f"相对导入失败: {e}")
            print("尝试最后手段：动态导入")
            import importlib.util
            
            def load_module(module_path):
                spec = importlib.util.spec_from_file_location("module", module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            
            # 加载各个模块
            data_processor_path = os.path.join(current_dir, "src", "data_processor.py")
            analysis_path = os.path.join(current_dir, "src", "analysis.py")
            utils_path = os.path.join(current_dir, "src", "utils.py")
            
            if os.path.exists(data_processor_path):
                data_processor = load_module(data_processor_path)
                add_temporal_features = data_processor.add_temporal_features
                add_lunar_features = data_processor.add_lunar_features
                add_festival_features = data_processor.add_festival_features
                add_season_features = data_processor.add_season_features
                add_rolling_features = data_processor.add_rolling_features
                print("成功加载 data_processor 模块")
            else:
                print(f"错误: 文件不存在 {data_processor_path}")
                raise ImportError(f"文件不存在 {data_processor_path}")
            
            if os.path.exists(analysis_path):
                analysis = load_module(analysis_path)
                LotteryAnalyzer = analysis.LotteryAnalyzer
                print("成功加载 analysis 模块")
            else:
                print(f"错误: 文件不存在 {analysis_path}")
                raise ImportError(f"文件不存在 {analysis_path}")
            
            if os.path.exists(utils_path):
                utils = load_module(utils_path)
                send_dingtalk = utils.send_dingtalk
                send_email = utils.send_email
                log_error = utils.log_error
                print("成功加载 utils 模块")
            else:
                print(f"错误: 文件不存在 {utils_path}")
                raise ImportError(f"文件不存在 {utils_path}")
            
            print("成功使用动态导入")

def main():
    try:
        print("=" * 50)
        print(f"开始执行分析任务 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
        # 创建分析器
        print("初始化分析器...")
        analyzer = LotteryAnalyzer()
        
        if analyzer.df.empty:
            print("错误：未获取到有效数据，终止分析")
            return
        
        print(f"原始数据记录数: {len(analyzer.df)}")
        
        # 添加基础特征
        print("添加时序特征...")
        analyzer.df = add_temporal_features(analyzer.df)
        
        print("添加农历特征...")
        analyzer.df = add_lunar_features(analyzer.df)
        
        print("添加节日特征...")
        analyzer.df = add_festival_features(analyzer.df)
        
        print("添加季节特征...")
        analyzer.df = add_season_features(analyzer.df)
        
        # 添加滚动窗口特征
        print("添加滚动窗口特征...")
        analyzer.df = add_rolling_features(analyzer.df)
        
        # 检查特征是否添加成功
        rolling_features = [col for col in analyzer.df.columns if col.startswith('rolling_')]
        heat_index_features = [col for col in analyzer.df.columns if col.startswith('heat_index_')]
        
        print(f"特征工程后数据记录数: {len(analyzer.df)}")
        print(f"添加的滚动特征数量: {len(rolling_features)}")
        print(f"添加的热度指数特征数量: {len(heat_index_features)}")
        
        # 生成报告
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
        error_msg = f"分析任务失败: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        log_error({
            'draw_number': 'SYSTEM_ERROR',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'actual_zodiac': 'N/A',
            'predicted_zodiacs': 'N/A',
            'last_zodiac': 'N/A',
            'weekday': datetime.now().weekday() + 1,
            'month': datetime.now().month
        })
        
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            send_dingtalk(f"分析任务失败: {str(e)}", ding_webhook)
        
        if email_receiver:
            send_email("彩票分析系统错误", error_msg, email_receiver)

if __name__ == "__main__":
    main()
[file content end]
