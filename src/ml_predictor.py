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
import hashlib
import json

# === 修改：XGBoost 可选导入与降级方案 ===
XGB_INSTALLED = False
try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
    logging.info("XGBoost 已成功导入")
except ImportError:
    logging.warning("XGBoost 未安装，将使用随机森林作为替代模型")

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from config import ML_MODEL_PATH
from src.utils import fetch_historical_data
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    def __init__(self, model_type='auto'):
        """
        初始化机器学习预测器
        
        参数:
            model_type: 模型类型，可选 'auto', 'xgboost', 'randomforest', 'svm', 'gradientboosting'
                      默认'auto'会自动选择最佳可用模型
        """
        # 自动选择模型类型
        if model_type == 'auto':
            if XGB_INSTALLED:
                model_type = 'xgboost'
                logger.info("自动选择 XGBoost 模型")
            else:
                model_type = 'randomforest'
                logger.info("自动选择 随机森林 模型")
        
        # 检查XGBoost可用性
        if model_type == 'xgboost' and not XGB_INSTALLED:
            logger.warning("XGBoost 不可用，自动切换为随机森林模型")
            model_type = 'randomforest'
        
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.feature_signature = None  # 新增: 特征签名
        self.zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
        
        # 尝试加载预训练模型
        self.load_model()
        logger.info(f"MLPredictor 初始化完成，使用模型: {model_type}")
    
    def load_model(self):
        """加载预训练的模型和预处理工具"""
        model_path = os.path.join(ML_MODEL_PATH, f'{self.model_type}_model.pkl')
        preprocessor_path = os.path.join(ML_MODEL_PATH, 'preprocessor.pkl')
        encoder_path = os.path.join(ML_MODEL_PATH, 'label_encoder.pkl')
        features_path = os.path.join(ML_MODEL_PATH, 'feature_columns.txt')
        signature_path = os.path.join(ML_MODEL_PATH, 'feature_signature.json')  # 修改为JSON格式
        
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info(f"已加载预训练的{self.model_type}模型")
            if os.path.exists(preprocessor_path):
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("已加载特征预处理器")
            if os.path.exists(encoder_path):
                self.label_encoder = joblib.load(encoder_path)
                logger.info("已加载标签编码器")
            if os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    self.feature_columns = f.read().splitlines()
                logger.info(f"已加载{len(self.feature_columns)}个特征列")
            # 加载特征签名
            if os.path.exists(signature_path):
                with open(signature_path, 'r') as f:
                    self.feature_signature = json.load(f)
                logger.info("已加载特征签名")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.model = None
            self.preprocessor = None
            self.feature_signature = None
    
    def save_model(self):
        """保存模型和预处理工具"""
        if self.model is None:
            logger.warning("无法保存模型 - 模型未训练")
            return
        
        model_path = os.path.join(ML_MODEL_PATH, f'{self.model_type}_model.pkl')
        preprocessor_path = os.path.join(ML_MODEL_PATH, 'preprocessor.pkl')
        encoder_path = os.path.join(ML_MODEL_PATH, 'label_encoder.pkl')
        features_path = os.path.join(ML_MODEL_PATH, 'feature_columns.txt')
        signature_path = os.path.join(ML_MODEL_PATH, 'feature_signature.json')  # 修改为JSON格式
        
        try:
            joblib.dump(self.model, model_path)
            if self.preprocessor:
                joblib.dump(self.preprocessor, preprocessor_path)
            joblib.dump(self.label_encoder, encoder_path)
            with open(features_path, 'w') as f:
                f.write('\n'.join(self.feature_columns))
            # 保存特征签名
            if self.feature_signature is not None:
                with open(signature_path, 'w') as f:
                    json.dump(self.feature_signature, f, indent=2)
                logger.info(f"特征签名已保存到: {signature_path}")
            logger.info(f"模型已保存到: {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def create_feature_signature(self, X):
        """创建特征签名用于版本控制"""
        # 计算特征哈希值
        columns_hash = hashlib.sha256(
            ','.join(sorted(X.columns)).encode('utf-8')
        ).hexdigest()
        
        # 创建特征签名
        self.feature_signature = {
            'columns': sorted(X.columns.tolist()),
            'dtypes': {col: str(X[col].dtype) for col in X.columns},
            'shape': (len(X.columns),),
            'hash': columns_hash,
            'created_at': pd.Timestamp.now().isoformat()
        }
        logger.info(f"创建特征签名: {columns_hash[:8]}...")
    
    def validate_feature_signature(self, X):
        """验证输入特征与训练特征签名是否匹配"""
        if self.feature_signature is None:
            logger.error("特征签名不可用，无法验证")
            return False
        
        # 检查特征数量
        if len(X.columns) != self.feature_signature['shape'][0]:
            logger.error(f"特征数量不匹配! 训练: {self.feature_signature['shape'][0]}, 输入: {len(X.columns)}")
            return False
        
        # 检查特征列
        missing_cols = set(self.feature_signature['columns']) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_signature['columns'])
        
        if missing_cols or extra_cols:
            logger.error(f"特征列不匹配! 缺失: {missing_cols}, 多余: {extra_cols}")
            return False
        
        # 检查数据类型
        for col, dtype in self.feature_signature['dtypes'].items():
            if str(X[col].dtype) != dtype:
                logger.error(f"特征 '{col}' 数据类型不匹配! 训练: {dtype}, 输入: {X[col].dtype}")
                return False
        
        logger.info("特征签名验证通过")
        return True
    
    def align_features(self, features):
        """
        对齐特征，确保与训练特征一致
        1. 添加缺失的特征列并用0填充
        2. 移除多余的特征列
        3. 确保特征顺序一致
        """
        aligned_features = {}
        
        # 如果特征签名不可用，直接返回原始特征
        if self.feature_signature is None:
            logger.warning("无特征签名可用，跳过特征对齐")
            return features
        
        # 确保所有特征都存在
        for col in self.feature_signature['columns']:
            if col in features:
                aligned_features[col] = features[col]
            else:
                logger.warning(f"特征 '{col}' 缺失，填充默认值0")
                aligned_features[col] = 0
        
        # 移除多余特征
        extra_cols = set(features.keys()) - set(self.feature_signature['columns'])
        if extra_cols:
            logger.warning(f"移除多余特征: {extra_cols}")
        
        return aligned_features
    
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
        
        # 选择特征列
        self.feature_columns = [
            'days_since_start', 'year', 'weekday', 'month', 
            'is_festival', 'prev_zodiac', 'season'
        ]
        
        # 添加生肖特征列
        for zodiac in self.zodiacs:
            self.feature_columns.extend([
                f'occur_{zodiac}',
                f'rolling_7p_{zodiac}',
                f'rolling_30p_{zodiac}'
            ])
        
        # 处理NaN值
        data = data.dropna(subset=self.feature_columns + ['zodiac'])
        
        # 特征和目标变量
        X = data[self.feature_columns]
        y = data['zodiac']
        
        # 编码目标变量
        y_encoded = self.label_encoder.fit_transform(y)
        
        # 创建特征签名
        self.create_feature_signature(X)
        
        logger.info(f"数据准备完成，特征维度: {X.shape}")
        return X, y_encoded
    
    def create_preprocessor(self, X):
        """创建特征预处理器，正确处理分类特征"""
        # 识别数值特征和分类特征
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 创建数值特征处理器
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 创建分类特征处理器
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # 组合处理器
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
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
        
        # 创建预处理器
        self.preprocessor = self.create_preprocessor(X)
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        accuracies = []
        
        logger.info(f"使用时间序列交叉验证 ({tscv.get_n_splits()} 折)")
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 预处理特征
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # 初始化模型
            if self.model_type == 'randomforest':
                model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
                logger.info(f"第 {fold+1} 折 - 使用随机森林模型")
            elif self.model_type == 'svm':
                model = SVC(probability=True, random_state=42)
                logger.info(f"第 {fold+1} 折 - 使用SVM模型")
            elif self.model_type == 'gradientboosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
                logger.info(f"第 {fold+1} 折 - 使用梯度提升模型")
            else:  # xgboost
                if XGB_INSTALLED:
                    model = XGBClassifier(n_estimators=100, random_state=42, max_depth=3)
                    logger.info(f"第 {fold+1} 折 - 使用XGBoost模型")
                else:
                    logger.warning(f"第 {fold+1} 折 - XGBoost 不可用，使用随机森林替代")
                    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
            
            # 训练模型
            model.fit(X_train_processed, y_train)
            
            # 评估模型
            train_pred = model.predict(X_train_processed)
            train_acc = accuracy_score(y_train, train_pred)
            
            test_pred = model.predict(X_test_processed)
            test_acc = accuracy_score(y_test, test_pred)
            
            accuracies.append((train_acc, test_acc))
            logger.info(f"第 {fold+1} 折 - 训练准确率: {train_acc:.4f}, 测试准确率: {test_acc:.4f}")
        
        # 计算平均准确率
        avg_train_acc = np.mean([acc[0] for acc in accuracies])
        avg_test_acc = np.mean([acc[1] for acc in accuracies])
        
        logger.info(f"平均训练准确率: {avg_train_acc:.4f}, 平均测试准确率: {avg_test_acc:.4f}")
        
        # 使用全部数据重新训练最终模型
        X_processed = self.preprocessor.fit_transform(X)
        
        # 初始化最终模型
        if self.model_type == 'randomforest':
            self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
            logger.info("最终模型: 随机森林 (n_estimators=150)")
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
            logger.info("最终模型: SVM")
        elif self.model_type == 'gradientboosting':
            self.model = GradientBoostingClassifier(n_estimators=150, random_state=42, max_depth=3)
            logger.info("最终模型: 梯度提升 (n_estimators=150)")
        else:  # xgboost
            if XGB_INSTALLED:
                self.model = XGBClassifier(n_estimators=150, random_state=42, max_depth=3)
                logger.info("最终模型: XGBoost (n_estimators=150)")
            else:
                logger.warning("XGBoost 不可用，使用随机森林作为最终模型")
                self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
        
        # 训练最终模型
        self.model.fit(X_processed, y)
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
        
        # 对齐特征
        aligned_features = self.align_features(features)
        
        # 创建特征DataFrame
        X = pd.DataFrame([aligned_features])
        
        # 验证特征签名
        if not self.validate_feature_signature(X):
            logger.error("特征签名验证失败! 请重新训练模型")
            return []
        
        # 特征预处理
        try:
            if self.preprocessor:
                X_processed = self.preprocessor.transform(X)
            else:
                logger.warning("没有预处理器，使用原始特征")
                X_processed = X.values
        except Exception as e:
            logger.error(f"特征预处理失败: {e}")
            return []
        
        # 预测概率
        try:
            probabilities = self.model.predict_proba(X_processed)[0]
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
        
        # 验证特征签名
        if not self.validate_feature_signature(X):
            logger.error("特征签名验证失败! 请重新训练模型")
            return 0.0
        
        if self.preprocessor:
            X_processed = self.preprocessor.transform(X)
        else:
            logger.warning("没有预处理器，使用原始特征")
            X_processed = X.values
        
        # 预测
        y_pred = self.model.predict(X_processed)
        
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
                # 获取特征名称（处理分类特征）
                feature_names = []
                if self.preprocessor:
                    # 获取数值特征名称
                    num_features = self.preprocessor.transformers_[0][2]
                    feature_names.extend(num_features)
                    
                    # 获取分类特征名称
                    cat_transformer = self.preprocessor.transformers_[1][1]
                    if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                        onehot = cat_transformer.named_steps['onehot']
                        cat_features = onehot.get_feature_names_out(self.preprocessor.transformers_[1][2])
                        feature_names.extend(cat_features)
                else:
                    feature_names = self.feature_columns
                
                # 创建Series
                importance_series = pd.Series(self.model.feature_importances_, index=feature_names)
                logger.info("特征重要性计算完成")
                return importance_series.sort_values(ascending=False)
            else:
                logger.warning("无法获取特征重要性 - 模型类型不支持")
                return pd.Series()
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
        predictor = MLPredictor(model_type='auto')
        
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
