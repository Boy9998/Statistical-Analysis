import os
import sys
import traceback
import numpy as np
from datetime import datetime
import subprocess
import importlib.util
import pkg_resources

def find_requirements_file():
    """查找 requirements.txt 文件路径"""
    search_paths = [
        os.path.dirname(os.path.abspath(__file__)),  # 当前目录
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"),  # src目录
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),  # 上级目录
    ]
    
    for path in search_paths:
        req_path = os.path.join(path, "requirements.txt")
        if os.path.exists(req_path):
            return req_path
    return None

def check_dependencies():
    """检查并安装缺失的依赖项"""
    req_file = find_requirements_file()
    
    if not req_file:
        print("Warning: requirements.txt not found in project directories")
        print("Defaulting to core dependencies check...")
        core_dependencies = [
            'pandas>=2.0.3',
            'numpy>=1.24.4',
            'scikit-learn>=1.3.2',
            'requests>=2.31.0',
            'xgboost>=1.7.0; platform_system!="Windows"',  # Windows系统可选
            'joblib>=1.2.0'
        ]
        required_packages = core_dependencies
        print("Checking core dependencies only:")
    else:
        print(f"Using requirements file: {req_file}")
        try:
            with open(req_file, 'r') as f:
                required_packages = []
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pkg = line.split('#')[0].strip()
                        if pkg:
                            required_packages.append(pkg)
        except Exception as e:
            print(f"Error reading requirements.txt: {e}")
            return False
    
    if not required_packages:
        print("No dependencies to check")
        return True
    
    print("=" * 50)
    print("Checking dependencies...")
    
    try:
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing = []
        version_mismatch = []
        
        for pkg_spec in required_packages:
            # 跳过平台特定依赖检查
            if ';' in pkg_spec:
                pkg_name = pkg_spec.split(';')[0].strip()
                platform_condition = pkg_spec.split(';')[1].strip()
                
                # 简单平台检查
                if 'platform_system' in platform_condition:
                    sys_condition = platform_condition.split('!=')[1].strip('"\'')
                    if sys.platform.lower() == sys_condition.lower():
                        continue  # 跳过不匹配平台的依赖
            else:
                pkg_name = pkg_spec
            
            if '>=' in pkg_name:
                pkg_name, req_version = pkg_name.split('>=', 1)
                op = '>='
            elif '==' in pkg_name:
                pkg_name, req_version = pkg_name.split('==', 1)
                op = '=='
            else:
                pkg_name = pkg_name
                req_version = None
                op = None
            
            pkg_name = pkg_name.lower().strip()
            if req_version:
                req_version = req_version.strip()
            
            if pkg_name in installed:
                if req_version:
                    installed_version = pkg_resources.parse_version(installed[pkg_name])
                    required_version = pkg_resources.parse_version(req_version)
                    
                    if op == '>=' and installed_version < required_version:
                        version_mismatch.append(f"{pkg_name} (installed: {installed[pkg_name]}, required: {pkg_spec})")
                    elif op == '==' and installed_version != required_version:
                        version_mismatch.append(f"{pkg_name} (installed: {installed[pkg_name]}, required: {pkg_spec})")
            else:
                missing.append(pkg_spec.split(';')[0].strip())  # 添加不带平台条件的包名
        
        if not missing and not version_mismatch:
            print("All dependencies are satisfied ✓")
            print("=" * 50)
            return True
        
        if version_mismatch:
            print("Version mismatches detected:")
            for item in version_mismatch:
                print(f" - {item}")
        
        if missing:
            print(f"Missing packages: {', '.join(missing)}")
        
        print("=" * 50)
        
        if missing:
            print("Attempting to install missing packages...")
            try:
                # 安装时不带平台条件
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                print("Installation successful ✓")
                print("Please restart the program")
                return False
            except subprocess.CalledProcessError as e:
                print(f"Installation failed: {e}")
                print("Please install manually with:")
                print(f"pip install {' '.join(missing)}")
                return False
    
    except Exception as e:
        print(f"Dependency check failed: {str(e)}")
        print(traceback.format_exc())
        return False

def import_module(module_name, package=None):
    """动态导入模块，支持绝对导入"""
    try:
        module = importlib.import_module(module_name, package)
        return module
    except ImportError:
        print(f"Failed to import {module_name}, trying alternative path...")
        try:
            # 尝试从src目录导入
            module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", f"{module_name}.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Failed to import module {module_name}: {e}")
            return None

# 设置工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 确保正确导入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "src"))

# 在导入其他模块前检查依赖
if not check_dependencies():
    sys.exit(1)

# 使用绝对导入方式
try:
    from src.data_processor import (
        add_temporal_features,
        add_lunar_features,
        add_festival_features,
        add_season_features,
        add_rolling_features
    )
    from src.analysis import LotteryAnalyzer
    from src.utils import (
        send_dingtalk,
        send_email,
        log_error
    )
except ImportError as e:
    print(f"Absolute import failed: {e}")
    print("Trying dynamic import as fallback...")
    
    # 动态导入作为回退方案
    data_processor = import_module("data_processor", "src")
    analysis = import_module("analysis", "src")
    utils = import_module("utils", "src")
    
    if not all([data_processor, analysis, utils]):
        print("Critical modules missing, exiting...")
        sys.exit(1)
    
    # 获取具体功能
    add_temporal_features = getattr(data_processor, 'add_temporal_features', None)
    add_lunar_features = getattr(data_processor, 'add_lunar_features', None)
    add_festival_features = getattr(data_processor, 'add_festival_features', None)
    add_season_features = getattr(data_processor, 'add_season_features', None)
    add_rolling_features = getattr(data_processor, 'add_rolling_features', None)
    LotteryAnalyzer = getattr(analysis, 'LotteryAnalyzer', None)
    send_dingtalk = getattr(utils, 'send_dingtalk', None)
    send_email = getattr(utils, 'send_email', None)
    log_error = getattr(utils, 'log_error', None)

def main():
    try:
        print("=" * 50)
        print(f"Starting analysis task [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
        if not LotteryAnalyzer:
            print("Error: LotteryAnalyzer not found")
            return
        
        print("Initializing analyzer...")
        analyzer = LotteryAnalyzer()
        
        if analyzer.df.empty:
            print("Error: No valid data obtained, aborting analysis")
            return
        
        print(f"Original data records: {len(analyzer.df)}")
        
        # 添加特征
        features = [
            ('temporal', add_temporal_features),
            ('lunar', add_lunar_features),
            ('festival', add_festival_features),
            ('season', add_season_features),
            ('rolling', add_rolling_features)
        ]
        
        for name, func in features:
            if func:
                print(f"Adding {name} features...")
                analyzer.df = func(analyzer.df)
            else:
                print(f"Warning: {name} feature function not available")
        
        # 特征统计
        rolling_features = [col for col in analyzer.df.columns if col.startswith('rolling_')]
        heat_index_features = [col for col in analyzer.df.columns if col.startswith('heat_index_')]
        
        print(f"Processed data records: {len(analyzer.df)}")
        print(f"Added rolling features: {len(rolling_features)}")
        print(f"Added heat index features: {len(heat_index_features)}")
        
        # 生成报告
        print("\nGenerating analysis report...")
        report = analyzer.generate_report()
        
        print("\nReport summary:")
        print(report.split("下期预测")[0].strip())
        
        # 发送通知
        if send_dingtalk:
            ding_webhook = os.getenv("DINGTALK_WEBHOOK")
            if ding_webhook:
                print("Sending DingTalk notification...")
                send_dingtalk(report, ding_webhook)
        
        if send_email:
            email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
            if email_receiver:
                print("Sending email notification...")
                send_email("Daily Lottery Analysis Report", report, email_receiver)
        
        print("=" * 50)
        print(f"Analysis task completed [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
    except Exception as e:
        error_msg = f"Analysis task failed: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        
        if log_error:
            log_error({
                'draw_number': 'SYSTEM_ERROR',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'actual_zodiac': 'N/A',
                'predicted_zodiacs': 'N/A',
                'last_zodiac': 'N/A',
                'weekday': datetime.now().weekday() + 1,
                'month': datetime.now().month
            })
        
        if send_dingtalk:
            ding_webhook = os.getenv("DINGTALK_WEBHOOK")
            if ding_webhook:
                send_dingtalk(f"Analysis task failed: {str(e)}", ding_webhook)
        
        if send_email:
            email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
            if email_receiver:
                send_email("Lottery Analysis System Error", error_msg, email_receiver)

if __name__ == "__main__":
    main()
