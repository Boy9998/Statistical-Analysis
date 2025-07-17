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

def check_dependencies():
    """检查并安装缺失的依赖项"""
    req_file = os.path.join(current_dir, "requirements.txt")
    
    if not os.path.exists(req_file):
        print(f"Error: Requirements file not found at {req_file}")
        return False
    
    print("=" * 50)
    print("Checking dependencies...")
    print(f"Using requirements file: {req_file}")
    
    try:
        # 读取依赖文件并过滤有效内容
        with open(req_file, 'r') as f:
            required_packages = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg = line.split('#')[0].strip()  # 移除行内注释
                    if pkg:
                        required_packages.append(pkg)
        
        if not required_packages:
            print("Warning: No valid dependencies found in requirements.txt")
            return True
        
        # 检查已安装的包
        installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        missing = []
        version_mismatch = []
        
        for pkg_spec in required_packages:
            if '==' in pkg_spec:
                pkg_name, req_version = pkg_spec.split('==', 1)
                pkg_name = pkg_name.lower().strip()
                req_version = req_version.strip()
            else:
                pkg_name = pkg_spec.lower().strip()
                req_version = None
            
            if pkg_name in installed:
                if req_version and installed[pkg_name] != req_version:
                    version_mismatch.append(f"{pkg_name} (required: {req_version}, installed: {installed[pkg_name]})")
            else:
                missing.append(pkg_spec)
        
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
        
        # 尝试安装缺失依赖
        if missing:
            print("Attempting to install missing packages...")
            try:
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
    print("Successfully imported modules from src package")
except ImportError as e:
    print(f"Import from src package failed: {e}")
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
        print("Successfully imported modules directly")
    except ImportError as e:
        print(f"Direct import failed: {e}")
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
            print("Successfully used relative imports")
        except ImportError as e:
            print(f"Relative import failed: {e}")
            print("Attempting dynamic import...")
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
                print("Successfully loaded data_processor module")
            else:
                raise ImportError(f"File not found: {data_processor_path}")
            
            if os.path.exists(analysis_path):
                analysis = load_module(analysis_path)
                LotteryAnalyzer = analysis.LotteryAnalyzer
                print("Successfully loaded analysis module")
            else:
                raise ImportError(f"File not found: {analysis_path}")
            
            if os.path.exists(utils_path):
                utils = load_module(utils_path)
                send_dingtalk = utils.send_dingtalk
                send_email = utils.send_email
                log_error = utils.log_error
                print("Successfully loaded utils module")
            else:
                raise ImportError(f"File not found: {utils_path}")
            
            print("Dynamic import completed")

def main():
    try:
        print("=" * 50)
        print(f"Starting analysis task [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
        print("Initializing analyzer...")
        analyzer = LotteryAnalyzer()
        
        if analyzer.df.empty:
            print("Error: No valid data obtained, aborting analysis")
            return
        
        print(f"Original data records: {len(analyzer.df)}")
        
        print("Adding temporal features...")
        analyzer.df = add_temporal_features(analyzer.df)
        
        print("Adding lunar features...")
        analyzer.df = add_lunar_features(analyzer.df)
        
        print("Adding festival features...")
        analyzer.df = add_festival_features(analyzer.df)
        
        print("Adding season features...")
        analyzer.df = add_season_features(analyzer.df)
        
        print("Adding rolling window features...")
        analyzer.df = add_rolling_features(analyzer.df)
        
        rolling_features = [col for col in analyzer.df.columns if col.startswith('rolling_')]
        heat_index_features = [col for col in analyzer.df.columns if col.startswith('heat_index_')]
        
        print(f"Processed data records: {len(analyzer.df)}")
        print(f"Added rolling features: {len(rolling_features)}")
        print(f"Added heat index features: {len(heat_index_features)}")
        
        print("\nGenerating analysis report...")
        report = analyzer.generate_report()
        
        print("\nReport summary:")
        print(report.split("下期预测")[0].strip())
        
        ding_webhook = os.getenv("DINGTALK_WEBHOOK")
        email_receiver = os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_USER")
        
        if ding_webhook:
            print("Sending DingTalk notification...")
            send_dingtalk(report, ding_webhook)
        
        if email_receiver:
            print("Sending email notification...")
            send_email("Daily Lottery Analysis Report", report, email_receiver)
            
        print("=" * 50)
        print(f"Analysis task completed [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        print("=" * 50)
        
    except Exception as e:
        error_msg = f"Analysis task failed: {str(e)}\n\n{traceback.format_exc()}"
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
            send_dingtalk(f"Analysis task failed: {str(e)}", ding_webhook)
        
        if email_receiver:
            send_email("Lottery Analysis System Error", error_msg, email_receiver)

if __name__ == "__main__":
    main()
