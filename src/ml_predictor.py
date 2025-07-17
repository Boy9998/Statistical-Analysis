[file name]: src/ml_predictor.py
[file content begin]
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import warnings
import logging

# === 增强降级处理 ===
try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
    logging.info("XGBoost 已成功导入")
except ImportError:
    XGB_INSTALLED = False
    logging.warning("XGBoost 未安装，将使用随机森林作为替代模型")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from config import ML_MODEL_PATH
from src.utils import fetch_historical_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ML_MODEL_PATH, 'ml_predictor.log'))
    ]
)
logger = logging.getLogger('MLPredictor')

# 忽略警告
warnings.filterwarnings('ignore')

# 确保模型目录存在
os.makedirs(ML_MODEL_PATH, exist_ok=True)

class MLPredictor:
    def __init__(self, model_type='xgboost'):
        """
        初始化机器学习预测器
        
        参数:
            model_type: 模型类型，可选 'xgboost', 'randomforest', 'svm', 'gradientboosting'
        """
        self.original_model_type = model_type
        
        # === 增强降级处理 ===
        if model_type == 'xgboost' and not XGB_INSTALLED:
            logger.warning("XGBoost 不可用，自动切换为随机森林模型")
            model_type = 'randomforest'
        
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        
        logger.info(f"初始化预测器 | 请求模型: {self.original_model_type} | 实际使用: {self.model_type}")
        
        # 尝试加载预训练模型
        self.load_model()
    
    def load_model(self):
        """加载预训练的模型和预处理工具"""
        model_path = os.path.join(ML_MODEL_PATH, f'{self.model_type}_model.pkl')
        scaler_path = os.path.join(ML_MODEL_PATH, 'scaler.pkl')
        encoder_path = os.path.join(ML_MODEL_PATH, 'label_encoder.pkl')
        features_path = os.path.join(ML_MODEL_PATH, 'feature_columns.txt')
        
        try:
            # === 增强模型加载回退 ===
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"已加载预训练的{self.model_type}模型")
            else:
                logger.warning(f"模型文件不存在: {model_path}")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("已加载特征缩放器")
            else:
                logger.warning("特征缩放器不存在，将在训练时创建")
                
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                logger.info("已加载标签编码器")
            else:
                logger.warning("标签编码器不存在，将在训练时创建")
                
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = f.read().splitlines()
                logger.info(f"已加载{len(self.feature_columns)}个特征列")
            else:
                logger.warning("特征列文件不存在，将在训练时生成")
                
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            # 重置模型状态确保安全回退
            self.model = None
            self.scaler = None
            self.label_encoder = LabelEncoder()
    
    def save_model(self):
        """保存模型和预处理工具"""
        if self.model is None:
            logger.warning("无法保存模型 - 模型未训练")
            return
        
        model_path = os.path.join(ML_MODEL_PATH, f'{self.model_type}_model.pkl')
        scaler_path = os.path.join(ML_MODEL_PATH, 'scaler.pkl')
        encoder_path = os.path.join(ML_MODEL_PATH, 'label_encoder.pkl')
        features_path = os.path.join(ML_MODEL_PATH, 'feature_columns.txt')
        
        try:
            joblib.dump(self.model, model_path)
            if self.scaler:
                joblib.dump(self.scaler, scaler_path)
            joblib.dump(self.label_encoder, encoder_path)
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_columns))
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def prepare_data(self, df):
        """
        准备训练数据
        
        参数:
            df: 包含历史数据的DataFrame
            
        返回:
            X: 特征数据
            y: 目标标签
        """
        # 确保数据包含必要的列
        required_columns = ['zodiac', 'date', 'season', 'is_festival', 'weekday', 'month']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"数据缺少必要列: {col}")
                raise ValueError(f"数据缺少必要列: {col}")
        
        # 创建副本避免修改原始数据
        data = df.copy()
        
        # 添加时序特征
        data['days_since_start'] = (data['date'] - data['date'].min()).dt.days
        data['year'] = data['date'].dt.year
        
        # 添加生肖特征
        for zodiac in self.zodiacs:
            # 添加生肖出现标志
            data[f'occur_{zodiac}'] = (data['zodiac'] == zodiac).astype(int)
            
            # 添加滚动窗口特征
            data[f'rolling_7p_{zodiac}'] = data[f'occur_{zodiac}'].rolling(window=7, min_periods=1).mean()
            data[f'rolling_30p_{zodiac}'] = data[f'occur_{zodiac}'].rolling(window=30, min_periods=1).mean()
        
        # 添加转移概率特征
        data['prev_zodiac'] = data['zodiac'].shift(1)
        for zodiac in self.zodiacs:
            data[f'trans_from_{zodiac}'] = (data['prev_zodiac'] == zodiac).astype(int)
        
        # 选择特征列
        self.feature_columns = [
            'days_since_start', 'year', 'weekday', 'month', 
            'is_festival', 'prev_zodiac'
        ]
        
        # 添加生肖特征列
        for zodiac in self.zodiacs:
            self.feature_columns.extend([
                f'occur_{zodiac}',
                f'rolling_7p_{zodiac}',
                f'rolling_30p_{zodiac}',
                f'trans_from_{zodiac}'
            ])
        
        # 添加季节特征
        seasons = ['春', '夏', '秋', '冬']
        for season in seasons:
            self.feature_columns.append(f'season_{season}')
            data[f'season_{season}'] = (data['season'] == season).astype(int)
        
        # 处理NaN值
        data = data.dropna(subset=self.feature_columns + ['zodiac'])
        
        # 特征和目标变量
        X = data[self.feature_columns]
        y = data['zodiac']
        
        # 编码目标变量
        y_encoded = self.label_encoder.fit_transform(y)
        
        logger.info(f"数据准备完成，特征维度: {X.shape}")
        return X, y_encoded
    
    def train_model(self, df, test_size=0.2, retrain=False):
        """
        训练机器学习模型
        
        参数:
            df: 包含历史数据的DataFrame
            test_size: 测试集比例
            retrain: 是否强制重新训练
            
        返回:
            训练准确率和测试准确率
        """
        # 检查是否已有模型
        if self.model and not retrain:
            logger.info("使用现有模型，跳过训练")
            return 0.0, 0.0
        
        logger.info("开始训练机器学习模型...")
        
        # 准备数据
        X, y = self.prepare_data(df)
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        logger.info(f"使用时间序列交叉验证 ({tscv.get_n_splits()} 折)")
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 特征缩放
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = self.scaler.transform(X_train)
            
            X_test_scaled = self.scaler.transform(X_test)
            
            # 初始化模型 (已移除冗余的XGB检查)
            if self.model_type == 'randomforest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            elif self.model_type == 'svm':
                model = SVC(probability=True, random_state=42)
            elif self.model_type == 'gradientboosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
            else:  # xgboost
                model = XGBClassifier(n_estimators=100, random_state=42, max_depth=3)
            
            logger.info(f"第 {fold+1} 折 - 使用 {self.model_type} 模型")
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 评估模型
            train_pred = model.predict(X_train_scaled)
            train_acc = accuracy_score(y_train, train_pred)
            
            test_pred = model.predict(X_test_scaled)
            test_acc = accuracy_score(y_test, test_pred)
            
            accuracies.append((train_acc, test_acc))
            logger.info(f"第 {fold+1} 折 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        # 计算平均准确率
        avg_train_acc = np.mean([acc[0] for acc in accuracies])
        avg_test_acc = np.mean([acc[1] for acc in accuracies])
        
        logger.info(f"平均训练准确率: {avg_train_acc:.4f}, 平均测试准确率: {avg_test_acc:.4f}")
        
        # 使用全部数据重新训练最终模型
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        
        # 初始化最终模型
        if self.model_type == 'randomforest':
            self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        elif self.model_type == 'gradientboosting':
            self.model = GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=3)
        else:  # xgboost
            self.model = XGBClassifier(n_estimators=150, random_state=42, max_depth=3)
        
        logger.info(f"最终模型: {self.model_type} (n_estimators=150)")
        
        # 训练最终模型
        self.model.fit(X_scaled, y)
        logger.info("最终模型训练完成")
        
        # 保存模型
        self.save_model()
        
        return avg_train_acc, avg_test_acc
    
    def predict(self, features):
        """
        使用训练好的模型进行预测
        
        参数:
            features: 包含特征的字典或Series
            
        返回:
            预测的生肖列表 (按概率排序)
        """
        if self.model is None:
            logger.warning("模型未训练，返回空预测")
            return []
        
        if not self.feature_columns:
            logger.warning("特征列未定义，返回空预测")
            return []
        
        # 创建特征DataFrame
        X = pd.DataFrame([features])
        
        # 只保留需要的特征
        available_features = [col for col in self.feature_columns if col in X.columns]
        if not available_features:
            logger.warning("没有匹配的特征列，返回空预测")
            return []
        
        X = X[available_features]
        
        # 特征缩放
        try:
            if self.scaler:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
        except Exception as e:
            logger.error(f"特征缩放失败: {e}")
            return []
        
        # 预测概率
        try:
            probabilities = self.model.predict_proba(X_scaled)[0]
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return []
        
        # 获取概率最高的3个生肖
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_zodiacs = self.label_encoder.inverse_transform(top_indices)
        
        logger.info(f"预测完成: {top_zodiacs}")
        return list(top_zodiacs)
    
    def evaluate(self, df):
        """
        评估模型在完整数据集上的表现
        
        参数:
            df: 包含历史数据的DataFrame
            
        返回:
            准确率
        """
        if self.model is None:
            logger.warning("无法评估 - 模型未训练")
            return 0.0
        
        # 准备数据
        X, y = self.prepare_data(df)
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # 预测
        y_pred = self.model.predict(X_scaled)
        
        # 计算准确率
        acc = accuracy_score(y, y_pred)
        logger.info(f"模型准确率: {acc:.4f}")
        
        return acc
    
    def feature_importance(self):
        """
        获取特征重要性
        
        返回:
            特征重要性Series (按重要性排序)
        """
        if self.model is None:
            logger.warning("无法获取特征重要性 - 模型未训练")
            return pd.Series()
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # 随机森林、XGBoost等
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # SVM等
                importances = np.abs(self.model.coef_[0])
            else:
                logger.warning("无法获取特征重要性 - 模型类型不支持")
                return pd.Series()
            
            # 创建Series
            importance_series = pd.Series(importances, index=self.feature_columns)
            logger.info("特征重要性计算完成")
            return importance_series.sort_values(ascending=False)
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return pd.Series()

# 示例用法
if __name__ == "__main__":
    # 获取历史数据
    print("获取历史数据...")
    df = fetch_historical_data()
    
    if df.empty:
        print("无法获取数据，退出")
    else:
        # 创建预测器
        predictor = MLPredictor(model_type='xgboost')
        
        # 训练模型
        train_acc, test_acc = predictor.train_model(df)
        
        # 评估模型
        predictor.evaluate(df)
        
        # 获取特征重要性
        importance = predictor.feature_importance()
        if not importance.empty:
            print("特征重要性 (Top 10):")
            print(importance.head(10))
        
        # 模拟预测
        if not df.empty:
            # 使用最新数据作为特征
            latest = df.iloc[-1]
            features = latest.to_dict()
            
            # 预测
            prediction = predictor.predict(features)
            print(f"预测结果: {prediction}")
[file content end]
