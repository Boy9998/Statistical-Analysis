import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import os
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings

# 导入ML_MODEL_PATH配置
try:
    from config import ML_MODEL_PATH
except ImportError:
    # 如果导入失败，使用默认值
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ML_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ml_models')
    print(f"警告: 未找到ML_MODEL_PATH配置，使用默认值'{ML_MODEL_PATH}'")

# 确保目录存在
os.makedirs(ML_MODEL_PATH, exist_ok=True)
os.makedirs('error_analysis', exist_ok=True)

class StrategyManager:
    def __init__(self):
        # 扩展权重维度，增加ML模型权重
        self.weights = {
            'frequency': 0.12,     # 基础频率 (原0.15)
            'transition': 0.12,    # 转移概率 (原0.15)
            'season': 0.08,        # 季节 (原0.10)
            'festival': 0.08,      # 节日 (原0.10)
            'rolling_7d': 0.12,    # 7天滚动特征 (原0.15) - 修复：d取代p
            'rolling_30d': 0.08,   # 30天滚动特征 (原0.10) - 修复：d取代p
            'rolling_100d': 0.08,  # 新增100天滚动特征
            'combo': 0.12,         # 生肖组合概率 (原0.15)
            'feature_imp': 0.08,   # 特征重要性 (原0.10)
            'ml_model': 0.20      # 新增ML模型权重
        }
        self.accuracy_history = []
        self.factor_performance = defaultdict(list)  # 记录每个因子独立表现的历史准确率
        self.feature_importance = None  # 存储特征重要性
        self.combo_probs = {}  # 生肖组合概率
        self.factor_validity = {}  # 因子有效性评分
        self.special_attention_patterns = {}  # 需要特别关注的转移模式
        self.zodiac_attention = defaultdict(int)  # 生肖关注度
        self.error_learning_rate = 0.2  # 错误学习率
        self.ml_model = self._load_ml_model()  # 加载ML模型
        print(f"初始化策略管理器: 权重={self.weights} | 强化学习已启用 | ML模型{'已加载' if self.ml_model else '未找到'}")
    
    def _load_ml_model(self):
        """加载预训练的ML模型"""
        model_path = os.path.join(ML_MODEL_PATH, 'xgboost_model.pkl')
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                print(f"加载ML模型失败: {e}")
        return None
    
    def apply_review_results(self, adjustments):
        """应用历史复盘结果 - 新增方法"""
        if not adjustments:
            print("无复盘结果需要应用")
            return
        
        print("应用历史复盘结果:")
        for adj in adjustments:
            pattern = adj['pattern']
            weight_adj = adj['weight_adjustment']
            
            # 更新或添加特殊关注模式
            if pattern in self.special_attention_patterns:
                # 如果模式已存在，更新权重倍数
                self.special_attention_patterns[pattern]['weight_multiplier'] = max(
                    1.5, 
                    self.special_attention_patterns[pattern]['weight_multiplier'] * weight_adj
                )
                print(f"- 更新模式权重: {pattern} -> 新权重倍数: {self.special_attention_patterns[pattern]['weight_multiplier']:.2f}")
            else:
                # 添加新关注模式
                self.special_attention_patterns[pattern] = {
                    'weight_multiplier': weight_adj,
                    'last_occurrence': datetime.now(),
                    'error_count': 0
                }
                print(f"- 新增关注模式: {pattern} -> 权重倍数: {weight_adj:.2f}")
    
    def adjust(self, accuracy, error_patterns=None):
        """根据准确率和错误模式动态调整权重 - 增强敏感度并关联错误频率"""
        self.accuracy_history.append(accuracy)
        
        # 计算近期准确率趋势 (最近10次)
        trend = np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else accuracy
        
        # 根据趋势和错误模式调整权重
        adjustment_made = False
        total_errors = 0
        
        # 1. 基于错误模式调整 - 增强敏感度
        if error_patterns:
            for last_zodiac, patterns in error_patterns.items():
                for actual, count in patterns.items():
                    total_errors += count
                    pattern_key = f"{last_zodiac}-{actual}"
                    
                    # 增加错误模式中实际出现生肖的权重
                    if actual in self.zodiacs:
                        # 增强调整幅度：与错误频率直接关联
                        adjustment = min(0.1, count * 0.015)  # 每15次错误增加0.1权重
                        if f'freq_{actual}' in self.weights:
                            self.weights[f'freq_{actual}'] = min(0.30, self.weights.get(f'freq_{actual}', 0) + adjustment)
                        adjustment_made = True
                        print(f"错误学习调整: {last_zodiac}->{actual} 模式, 增加 {actual} 权重 +{adjustment:.3f} (基于{count}次错误)")
                        
                        # 记录特殊关注模式
                        self.special_attention_patterns[pattern_key] = {
                            'weight_multiplier': 1.5 + (count * 0.05),  # 错误次数越多，权重倍数越高
                            'last_occurrence': datetime.now(),
                            'error_count': count
                        }
        
        # 2. 基于准确率动态调整 - 大幅提升敏感度
        # 当准确率低于60%时，大幅增加ML模型权重
        if accuracy < 0.60:
            # 计算调整幅度：与准确率偏差成比例
            accuracy_deficit = 0.60 - accuracy
            adjustment_factor = min(0.3, 0.15 + (accuracy_deficit * 1.5))
            
            # 大幅增加ML模型权重
            self.weights['ml_model'] = min(0.40, self.weights['ml_model'] + adjustment_factor)
            
            # 降低其他权重
            for factor in ['frequency', 'transition', 'season', 'festival', 'rolling_7d', 'rolling_30d', 'rolling_100d', 'combo', 'feature_imp']:
                self.weights[factor] = max(0.05, self.weights[factor] - (adjustment_factor * 0.5))
            
            print(f"准确率调整: 准确率低 ({accuracy:.2%})，大幅增加ML模型权重(+{adjustment_factor:.3f})")
            adjustment_made = True
        # 当准确率较高时，增加滚动窗口权重
        elif accuracy > 0.65:
            # 计算调整幅度：与准确率超值成比例
            accuracy_surplus = accuracy - 0.65
            adjustment_factor = min(0.2, 0.10 + (accuracy_surplus * 1.0))
            
            # 增加滚动窗口权重
            self.weights['rolling_7d'] = min(0.30, self.weights['rolling_7d'] + adjustment_factor)
            self.weights['rolling_30d'] = min(0.25, self.weights['rolling_30d'] + adjustment_factor)
            self.weights['rolling_100d'] = min(0.20, self.weights['rolling_100d'] + adjustment_factor)
            
            # 降低其他权重
            for factor in ['frequency', 'transition', 'season', 'festival', 'combo', 'feature_imp', 'ml_model']:
                self.weights[factor] = max(0.05, self.weights[factor] - (adjustment_factor * 0.3))
            
            print(f"准确率调整: 准确率高 ({accuracy:.2%})，增加滚动窗口权重(+{adjustment_factor:.3f})")
            adjustment_made = True
        
        # 3. 根据因子有效性调整权重
        if self.factor_validity:
            total_validity = sum(self.factor_validity.values())
            for factor, validity in self.factor_validity.items():
                if factor in self.weights:
                    # 增强有效性对权重的影响
                    validity_weight = 0.85 * (validity / total_validity)  # 有效性占比提高到85%
                    # 当前权重占15%
                    current_weight = 0.15 * self.weights[factor]
                    self.weights[factor] = validity_weight + current_weight
            print("因子有效性调整: 基于因子表现更新权重")
            adjustment_made = True
        
        # 4. 当连续3次准确率低于55%时，触发紧急优化
        if len(self.accuracy_history) >= 3 and all(acc < 0.55 for acc in self.accuracy_history[-3:]):
            emergency_adjust = 0.20  # 紧急调整幅度
            
            # 大幅增加ML模型权重
            self.weights['ml_model'] = min(0.45, self.weights['ml_model'] + emergency_adjust)
            
            # 重置其他权重
            for factor in ['frequency', 'transition', 'season', 'festival', 'rolling_7d', 'rolling_30d', 'rolling_100d', 'combo', 'feature_imp']:
                self.weights[factor] = max(0.05, self.weights[factor] * 0.7)  # 降低30%
            
            print(f"紧急优化: 连续3次准确率<55%，大幅增加ML模型权重(+{emergency_adjust:.3f})")
            
            # 触发模型重新训练标志
            self.retrain_needed = True
            adjustment_made = True
        
        # 归一化权重
        if adjustment_made:
            total = sum(self.weights.values())
            for key in self.weights:
                self.weights[key] = round(self.weights[key] / total, 3)  # 保留3位小数提高精度
            print(f"调整后权重: {self.weights}")
        
        return self.weights
    
    def update_factor_performance(self, factor_accuracy):
        """更新因子表现记录"""
        for factor, acc in factor_accuracy.items():
            self.factor_performance[factor].append(acc)
        print("因子表现更新:", factor_accuracy)
    
    def evaluate_factor_validity(self, df, window=100):
        """回测验证因子有效性 - 增强版"""
        # 添加特征存在性检查
        required_columns = ['zodiac', 'season', 'is_festival']
        for col in required_columns:
            if col not in df.columns:
                print(f"警告: 数据中缺失特征列 '{col}'，使用默认值")
                if col == 'is_festival':
                    df[col] = 0  # 默认非节日
                elif col == 'season':
                    df[col] = '未知'  # 默认季节未知
                elif col == 'zodiac':
                    # 如果缺少生肖列，无法继续验证
                    print("错误: 数据中缺少生肖列，无法进行因子有效性验证")
                    return {}
        
        if len(df) < window:
            print(f"数据不足{window}期，无法进行因子有效性验证")
            return {}
        
        # 使用最近window期数据进行回测
        recent = df.iloc[-window:]
        factor_scores = {}
        
        # 1. 频率因子验证
        freq_acc = self._validate_frequency_factor(recent)
        factor_scores['frequency'] = freq_acc
        
        # 2. 转移概率因子验证
        trans_acc = self._validate_transition_factor(recent)
        factor_scores['transition'] = trans_acc
        
        # 3. 季节因子验证
        season_acc = self._validate_season_factor(recent)
        factor_scores['season'] = season_acc
        
        # 4. 节日因子验证
        festival_acc = self._validate_festival_factor(recent)
        factor_scores['festival'] = festival_acc
        
        # 5. 滚动窗口因子验证 - 更新为7天、30天和100天
        rolling_acc = self._validate_rolling_factor(recent)
        factor_scores['rolling_7d'] = rolling_acc.get('rolling_7d', 0)
        factor_scores['rolling_30d'] = rolling_acc.get('rolling_30d', 0)
        factor_scores['rolling_100d'] = rolling_acc.get('rolling_100d', 0)
        
        # 6. 组合概率因子验证
        combo_acc = self._validate_combo_factor(recent)
        factor_scores['combo'] = combo_acc
        
        # 7. ML模型因子验证
        if self.ml_model:
            ml_acc = self._validate_ml_model(recent)
            factor_scores['ml_model'] = ml_acc
        else:
            factor_scores['ml_model'] = 0.0
        
        # 更新因子表现
        self.update_factor_performance(factor_scores)
        
        # 计算因子有效性评分（加权平均准确率）
        for factor, scores in self.factor_performance.items():
            if scores:
                # 使用指数加权平均，近期表现权重更高
                weights = np.array([0.5 ** i for i in range(len(scores))][::-1])
                weights = weights[:len(scores)]  # 确保权重长度匹配
                weights /= weights.sum()
                weighted_avg = np.dot(scores, weights)
                self.factor_validity[factor] = weighted_avg
        
        print(f"因子有效性评分: {self.factor_validity}")
        return factor_scores
    
    def _validate_ml_model(self, df):
        """验证ML模型有效性"""
        if not self.ml_model:
            return 0.0
            
        try:
            from sklearn.preprocessing import LabelEncoder
            
            # 准备数据
            X = df.drop(columns=['zodiac', 'date', 'expect', 'special'])
            # 只保留数值型特征
            X = X.select_dtypes(include=['number'])
            y = df['zodiac']
            
            # 检查数据有效性
            if X.empty or len(X) < 10:
                print("ML模型验证失败: 数据不足或无效")
                return 0.0
                
            # 编码目标变量
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # 预测
            predictions = self.ml_model.predict(X)
            acc = accuracy_score(y_encoded, predictions)
            return acc
            
        except Exception as e:
            print(f"ML模型验证失败: {e}")
            return 0.0
    
    def _validate_frequency_factor(self, df):
        """验证频率因子有效性"""
        predictions = []
        actuals = []
        
        for i in range(1, len(df)):
            # 使用历史数据计算频率
            freq = df['zodiac'].iloc[:i].value_counts(normalize=True)
            # 预测最高频率的生肖
            pred = freq.idxmax() if not freq.empty else None
            if pred:
                predictions.append(pred)
                actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def _validate_transition_factor(self, df):
        """验证转移概率因子有效性"""
        predictions = []
        actuals = []
        
        for i in range(1, len(df)):
            # 计算转移矩阵
            transitions = defaultdict(lambda: defaultdict(int))
            for j in range(1, i):
                prev = df['zodiac'].iloc[j-1]
                curr = df['zodiac'].iloc[j]
                transitions[prev][curr] += 1
            
            # 预测最可能的转移
            prev_zodiac = df['zodiac'].iloc[i-1]
            if prev_zodiac in transitions:
                transitions_from_prev = transitions[prev_zodiac]
                # 找到概率最高的转移
                total = sum(transitions_from_prev.values())
                if total > 0:
                    pred = max(transitions_from_prev, key=transitions_from_prev.get)
                    predictions.append(pred)
                    actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def _validate_season_factor(self, df):
        """验证季节因子有效性"""
        predictions = []
        actuals = []
        
        for i in range(len(df)):
            # 获取当前季节
            season = df['season'].iloc[i]
            # 使用同季节历史数据
            season_data = df[df['season'] == season].iloc[:i]
            
            if not season_data.empty:
                # 预测该季节出现频率最高的生肖
                freq = season_data['zodiac'].value_counts(normalize=True)
                pred = freq.idxmax() if not freq.empty else None
                if pred:
                    predictions.append(pred)
                    actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def _validate_festival_factor(self, df):
        """安全验证节日因子有效性"""
        # 确保节日特征存在
        if 'is_festival' not in df.columns:
            print("警告: 数据中缺少节日特征列，使用默认值")
            df['is_festival'] = 0
        
        predictions = []
        actuals = []
        
        for i in range(len(df)):
            # 检查是否是节日
            if df['is_festival'].iloc[i]:
                # 使用节日历史数据
                festival_data = df[df['is_festival'] == 1].iloc[:i]
                
                if not festival_data.empty:
                    # 预测节日出现频率最高的生肖
                    freq = festival_data['zodiac'].value_counts(normalize=True)
                    pred = freq.idxmax() if not freq.empty else None
                    if pred:
                        predictions.append(pred)
                        actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def _validate_rolling_factor(self, df):
        """验证滚动窗口因子有效性 - 更新为7天、30天和100天"""
        # 至少需要100天数据
        if len(df) < 100:
            return {'rolling_7d': 0, 'rolling_30d': 0, 'rolling_100d': 0}
        
        predictions_7d = []
        predictions_30d = []
        predictions_100d = []
        actuals = []
        
        for i in range(100, len(df)):
            # 7天滚动窗口
            rolling_7d = df.iloc[i-7:i]
            freq_7d = rolling_7d['zodiac'].value_counts(normalize=True)
            pred_7d = freq_7d.idxmax() if not freq_7d.empty else None
            
            # 30天滚动窗口
            rolling_30d = df.iloc[i-30:i]
            freq_30d = rolling_30d['zodiac'].value_counts(normalize=True)
            pred_30d = freq_30d.idxmax() if not freq_30d.empty else None
            
            # 100天滚动窗口
            rolling_100d = df.iloc[i-100:i]
            freq_100d = rolling_100d['zodiac'].value_counts(normalize=True)
            pred_100d = freq_100d.idxmax() if not freq_100d.empty else None
            
            if pred_7d and pred_30d and pred_100d:
                predictions_7d.append(pred_7d)
                predictions_30d.append(pred_30d)
                predictions_100d.append(pred_100d)
                actuals.append(df['zodiac'].iloc[i])
        
        acc_7d = accuracy_score(actuals, predictions_7d) if predictions_7d else 0
        acc_30d = accuracy_score(actuals, predictions_30d) if predictions_30d else 0
        acc_100d = accuracy_score(actuals, predictions_100d) if predictions_100d else 0
        
        return {'rolling_7d': acc_7d, 'rolling_30d': acc_30d, 'rolling_100d': acc_100d}
    
    def _validate_combo_factor(self, df):
        """验证组合概率因子有效性"""
        predictions = []
        actuals = []
        
        # 计算组合概率
        self.update_combo_probs(df, window=len(df))
        
        for i in range(1, len(df)):
            last_zodiac = df['zodiac'].iloc[i-1]
            preds = self.get_combo_prediction(last_zodiac, top_n=1)
            if preds:
                predictions.append(preds[0])
                actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def evaluate_feature_importance(self, df):
        """使用随机森林评估特征重要性"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # 准备数据
            X = df.drop(columns=['zodiac', 'date', 'expect', 'special'])
            # 只保留数值型特征
            X = X.select_dtypes(include=['number'])
            y = df['zodiac']
            
            # 检查数据有效性
            if X.empty or len(X) < 10:
                print("特征重要性评估失败: 数据不足或无效")
                return
                
            # 编码目标变量
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X, y_encoded)
            
            # 获取特征重要性
            importance = pd.Series(rf.feature_importances_, index=X.columns)
            self.feature_importance = importance.sort_values(ascending=False)
            
            print("特征重要性评估结果:")
            print(self.feature_importance.head(10))
            
            # 保存模型
            model_path = os.path.join(ML_MODEL_PATH, 'feature_importance_model.pkl')
            joblib.dump(rf, model_path)
            print(f"特征重要性模型已保存到: {model_path}")
            
            # 更新特征重要性权重
            self._update_feature_imp_weights()
            
        except Exception as e:
            print(f"特征重要性评估失败: {e}")
    
    def _update_feature_imp_weights(self):
        """根据特征重要性更新权重"""
        if self.feature_importance is None:
            return
        
        # 创建特征到因子的映射
        feature_to_factor = {
            'freq_': 'frequency',
            'trans_': 'transition',
            'season_': 'season',
            'festival_': 'festival',
            'rolling_7d_': 'rolling_7d',  # 修复：使用d取代p
            'rolling_30d_': 'rolling_30d', # 修复：使用d取代p
            'rolling_100d_': 'rolling_100d', # 新增100天滚动特征
            'combo_': 'combo',
            'ml_feat_': 'ml_model'  # 新增ML特征映射
        }
        
        # 计算每个因子的特征重要性总和
        factor_imp = defaultdict(float)
        for feature, imp in self.feature_importance.items():
            # 解析特征对应的生肖
            if '_' in feature:
                zodiac = feature.split('_')[-1]
                if zodiac in self.zodiacs and feature in features:
                    # 确保值是标量
                    value = features[feature]
                    if isinstance(value, pd.Series):
                        value = value.iloc[-1] if not value.empty else 0
                    feature_scores[zodiac] += value * imp
        
        sorted_zodiac = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:3]] if sorted_zodiac else []
    
    def _fuse_predictions(self, all_predictions):
        """融合多因子预测结果"""
        # 创建生肖得分字典
        zodiac_scores = defaultdict(float)
        
        # 为每个因子的预测结果分配分数
        for factor, preds in all_predictions.items():
            weight = self.weights.get(factor, 0.1)
            for i, zodiac in enumerate(preds):
                # 排名越高，得分越高（指数衰减）
                score = weight * (0.5 ** i)
                
                # 如果是特殊关注生肖，增加得分
                if zodiac in self.zodiac_attention:
                    score *= (1 + self.error_learning_rate * self.zodiac_attention[zodiac])
                
                zodiac_scores[zodiac] += score
        
        # 按得分排序
        sorted_zodiac = sorted(zodiac_scores.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:5]]  # 返回前5个预测结果
    
    @property
    def zodiacs(self):
        """生肖列表属性"""
        return ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]

# 测试函数
if __name__ == "__main__":
    print("===== 测试策略管理器 =====")
    
    # 创建测试策略管理器
    manager = StrategyManager()
    
    # 创建模拟特征数据
    features = {
        'freq_鼠': 0.15, 'freq_牛': 0.12, 'freq_虎': 0.10,
        'season_鼠': 0.20, 'season_牛': 0.15, 'season_虎': 0.12,
        'festival_鼠': 0.25, 'festival_牛': 0.18, 'festival_虎': 0.15,
        'rolling_7d_鼠': 0.18, 'rolling_7d_牛': 0.15, 'rolling_7d_虎': 0.12,
        'rolling_30d_鼠': 0.16, 'rolling_30d_牛': 0.14, 'rolling_30d_虎': 0.11,
        'rolling_100d_鼠': 0.15, 'rolling_100d_牛': 0.13, 'rolling_100d_虎': 0.10,
        'is_festival': 1  # 确保节日特征存在
    }
    
    # 测试生成预测
    last_zodiac = "鼠"
    prediction, factor_preds = manager.generate_prediction(features, last_zodiac)
    print(f"\n综合预测结果: {prediction}")
    print("\n各因子预测详情:")
    for factor, preds in factor_preds.items():
        print(f"- {factor}: {preds}")
    
    # 测试权重调整
    print("\n测试权重调整...")
    errors = {
        "鼠": {"牛": 3, "虎": 5},
        "牛": {"兔": 2}
    }
    new_weights = manager.adjust(0.58, errors)
    print(f"调整后权重: {new_weights}")
    
    # 测试因子报告生成
    print("\n生成因子报告:")
    report = manager.generate_factor_report()
    print(report)
    
    print("\n===== 测试完成 =====")
