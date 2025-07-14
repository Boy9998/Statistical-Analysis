import pandas as pd
import numpy as np
from collections import defaultdict
import joblib
import os
from sklearn.metrics import accuracy_score

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

class StrategyManager:
    def __init__(self):
        # 扩展权重维度
        self.weights = {
            'frequency': 0.15,     # 基础频率
            'transition': 0.15,    # 转移概率
            'season': 0.10,        # 季节
            'festival': 0.10,      # 节日
            'rolling_7d': 0.15,    # 7天滚动特征
            'rolling_30d': 0.10,   # 30天滚动特征
            'combo': 0.15,         # 生肖组合概率
            'feature_imp': 0.10    # 特征重要性
        }
        self.accuracy_history = []
        self.factor_performance = defaultdict(list)  # 记录每个因子独立表现的历史准确率
        self.feature_importance = None  # 存储特征重要性
        self.combo_probs = {}  # 生肖组合概率
        self.factor_validity = {}  # 因子有效性评分
        print(f"初始化策略管理器: 权重={self.weights}")
    
    def adjust(self, accuracy):
        """根据准确率动态调整权重"""
        self.accuracy_history.append(accuracy)
        
        # 计算近期准确率趋势 (最近10次)
        trend = np.mean(self.accuracy_history[-10:]) if len(self.accuracy_history) >= 10 else accuracy
        
        # 根据趋势调整权重
        if trend < 0.35:
            # 准确率低时增加组合概率和特征重要性权重
            self.weights['combo'] = min(0.20, self.weights['combo'] + 0.05)
            self.weights['feature_imp'] = min(0.20, self.weights['feature_imp'] + 0.05)
            # 降低其他权重
            for factor in ['frequency', 'transition', 'season', 'festival', 'rolling_7d', 'rolling_30d']:
                self.weights[factor] = max(0.05, self.weights[factor] - 0.01)
        elif trend > 0.45:
            # 准确率高时增加滚动窗口权重
            self.weights['rolling_7d'] = min(0.20, self.weights['rolling_7d'] + 0.03)
            self.weights['rolling_30d'] = min(0.20, self.weights['rolling_30d'] + 0.02)
            # 降低其他权重
            for factor in ['frequency', 'transition', 'season', 'festival', 'combo', 'feature_imp']:
                self.weights[factor] = max(0.05, self.weights[factor] - 0.01)
        
        # 根据因子有效性调整权重
        if self.factor_validity:
            total_validity = sum(self.factor_validity.values())
            for factor, validity in self.factor_validity.items():
                if factor in self.weights:
                    # 有效性评分占调整权重的50%
                    validity_weight = 0.5 * (validity / total_validity)
                    # 当前权重占50%
                    current_weight = 0.5 * self.weights[factor]
                    self.weights[factor] = validity_weight + current_weight
        
        # 归一化权重
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] = round(self.weights[key] / total, 2)
        
        print(f"调整后权重: {self.weights}")
        return self.weights
    
    def update_factor_performance(self, factor_accuracy):
        """更新因子表现记录"""
        for factor, acc in factor_accuracy.items():
            self.factor_performance[factor].append(acc)
        print("因子表现更新:", factor_accuracy)
    
    def evaluate_factor_validity(self, df, window=100):
        """回测验证因子有效性 - 修复数据不足问题"""
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
        
        # 3. 季节因子验证 - 确保数据中有季节列
        if 'season' in recent.columns:
            season_acc = self._validate_season_factor(recent)
            factor_scores['season'] = season_acc
        else:
            print("警告: 数据中缺少'season'列，跳过季节因子验证")
            factor_scores['season'] = 0.0
        
        # 4. 节日因子验证 - 确保数据中有节日列
        if 'is_festival' in recent.columns:
            festival_acc = self._validate_festival_factor(recent)
            factor_scores['festival'] = festival_acc
        else:
            print("警告: 数据中缺少'is_festival'列，跳过节日因子验证")
            factor_scores['festival'] = 0.0
        
        # 5. 滚动窗口因子验证
        rolling_acc = self._validate_rolling_factor(recent)
        factor_scores['rolling_7d'] = rolling_acc.get('rolling_7d', 0)
        factor_scores['rolling_30d'] = rolling_acc.get('rolling_30d', 0)
        
        # 6. 组合概率因子验证
        combo_acc = self._validate_combo_factor(recent)
        factor_scores['combo'] = combo_acc
        
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
        # 检查是否存在'season'列
        if 'season' not in df.columns:
            print("警告: 数据中缺少'season'列，无法验证季节因子")
            return 0.0
            
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
        """验证节日因子有效性"""
        # 检查是否存在'is_festival'列
        if 'is_festival' not in df.columns:
            print("警告: 数据中缺少'is_festival'列，无法验证节日因子")
            return 0.0
            
        predictions = []
        actuals = []
        
        for i in range(len(df)):
            # 检查是否是节日
            if df['is_festival'].iloc[i]:
                # 使用节日历史数据
                festival_data = df[df['is_festival']].iloc[:i]
                
                if not festival_data.empty:
                    # 预测节日出现频率最高的生肖
                    freq = festival_data['zodiac'].value_counts(normalize=True)
                    pred = freq.idxmax() if not freq.empty else None
                    if pred:
                        predictions.append(pred)
                        actuals.append(df['zodiac'].iloc[i])
        
        return accuracy_score(actuals, predictions) if predictions else 0
    
    def _validate_rolling_factor(self, df):
        """验证滚动窗口因子有效性 - 修复索引问题"""
        predictions_7d = []
        predictions_30d = []
        actuals = []
        
        # 确保有足够的数据
        if len(df) < 30:
            print("数据不足30期，无法验证滚动窗口因子")
            return {'rolling_7d': 0, 'rolling_30d': 0}
        
        for i in range(30, len(df)):
            # 7天滚动窗口
            rolling_7d = df.iloc[i-7:i]
            freq_7d = rolling_7d['zodiac'].value_counts(normalize=True)
            pred_7d = freq_7d.idxmax() if not freq_7d.empty else None
            
            # 30天滚动窗口
            rolling_30d = df.iloc[i-30:i]
            freq_30d = rolling_30d['zodiac'].value_counts(normalize=True)
            pred_30d = freq_30d.idxmax() if not freq_30d.empty else None
            
            if pred_7d and pred_30d:
                predictions_7d.append(pred_7d)
                predictions_30d.append(pred_30d)
                actuals.append(df['zodiac'].iloc[i])
        
        acc_7d = accuracy_score(actuals, predictions_7d) if predictions_7d else 0
        acc_30d = accuracy_score(actuals, predictions_30d) if predictions_30d else 0
        
        return {'rolling_7d': acc_7d, 'rolling_30d': acc_30d}
    
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
            
            # 准备数据 - 只使用数值型特征
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            X = df[numeric_cols].copy()
            
            # 移除不必要的列
            for col in ['date', 'expect', 'special']:
                if col in X.columns:
                    X.drop(columns=[col], inplace=True)
            
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
            'rolling_7d_': 'rolling_7d',
            'rolling_30d_': 'rolling_30d',
            'combo_': 'combo'
        }
        
        # 计算每个因子的特征重要性总和
        factor_imp = defaultdict(float)
        for feature, imp in self.feature_importance.items():
            for prefix, factor in feature_to_factor.items():
                if feature.startswith(prefix):
                    factor_imp[factor] += imp
                    break
        
        # 归一化因子重要性
        total_imp = sum(factor_imp.values())
        if total_imp > 0:
            for factor in factor_imp:
                factor_imp[factor] /= total_imp
            
            # 更新权重
            for factor, imp in factor_imp.items():
                if factor in self.weights:
                    # 特征重要性占权重调整的70%
                    self.weights[factor] = 0.7 * imp + 0.3 * self.weights[factor]
        
        print(f"基于特征重要性更新权重: {self.weights}")
    
    def update_combo_probs(self, df, window=100):
        """更新生肖组合概率 - 修复SettingWithCopyWarning"""
        if len(df) < window:
            print(f"数据不足{window}期，无法计算组合概率")
            return
        
        # 使用最近window期数据 - 创建副本避免警告
        recent = df.iloc[-window:].copy()
        
        # 创建组合列：上期生肖-本期生肖
        recent['combo'] = recent['zodiac'].shift() + '-' + recent['zodiac']
        
        # 删除NaN值
        recent = recent.dropna(subset=['combo'])
        
        # 计算组合概率
        combo_counts = recent['combo'].value_counts(normalize=True)
        self.combo_probs = combo_counts.to_dict()
        
        print(f"已更新生肖组合概率 (基于最近{window}期数据)")
        # 避免f-string嵌套问题
        top5_combos = list(combo_counts.head(5).items())
        print(f"前5个常见组合: {top5_combos}")
    
    def get_combo_prediction(self, last_zodiac, top_n=5):
        """获取基于组合概率的预测"""
        # 过滤以last_zodiac开头的组合
        relevant_combos = {combo: prob for combo, prob in self.combo_probs.items() 
                          if combo.startswith(f"{last_zodiac}-")}
        
        if not relevant_combos:
            return []
        
        # 按概率排序
        sorted_combos = sorted(relevant_combos.items(), key=lambda x: x[1], reverse=True)
        
        # 提取预测生肖（组合的后半部分）
        predictions = [combo.split('-')[1] for combo, _ in sorted_combos[:top_n]]
        return predictions
    
    def generate_factor_report(self):
        """生成因子表现报告"""
        report = "===== 因子表现报告 =====\n"
        
        # 当前权重
        report += "\n当前权重分配:\n"
        for factor, weight in self.weights.items():
            report += f"- {factor}: {weight:.2f}\n"
        
        # 因子历史表现
        report += "\n历史准确率 (最近5次平均):\n"
        for factor, acc_history in self.factor_performance.items():
            if acc_history:
                avg_acc = np.mean(acc_history[-5:])  # 最近5次平均
                report += f"- {factor}: {avg_acc:.2%}\n"
        
        # 特征重要性
        if self.feature_importance is not None:
            report += "\n特征重要性 (Top 5):\n"
            for feature, imp in self.feature_importance.head(5).items():
                report += f"- {feature}: {imp:.4f}\n"
        
        # 组合概率
        if self.combo_probs:
            report += "\n常见生肖组合 (Top 5):\n"
            sorted_combos = sorted(self.combo_probs.items(), key=lambda x: x[1], reverse=True)
            for combo, prob in sorted_combos[:5]:
                report += f"- {combo}: {prob:.2%}\n"
        
        # 因子有效性
        if self.factor_validity:
            report += "\n因子有效性评分 (0-1):\n"
            for factor, validity in self.factor_validity.items():
                report += f"- {factor}: {validity:.4f}\n"
        
        return report
    
    def generate_prediction(self, features, last_zodiac):
        """生成多因子综合预测 - 修复类型错误"""
        # 1. 频率因子预测
        freq_pred = self._frequency_prediction(features)
        
        # 2. 转移概率预测
        trans_pred = self._transition_prediction(last_zodiac)
        
        # 3. 季节因子预测 - 修复类型错误
        season_pred = self._season_prediction(features)
        
        # 4. 节日因子预测 - 修复类型错误
        festival_pred = self._festival_prediction(features)
        
        # 5. 滚动窗口预测
        rolling_pred = self._rolling_prediction(features)
        
        # 6. 组合概率预测
        combo_pred = self.get_combo_prediction(last_zodiac, top_n=5)
        
        # 7. 特征重要性加权预测
        imp_pred = self._importance_weighted_prediction(features)
        
        # 组合所有预测
        all_predictions = {
            'frequency': freq_pred,
            'transition': trans_pred,
            'season': season_pred,
            'festival': festival_pred,
            'rolling_7d': rolling_pred.get('7d', []),
            'rolling_30d': rolling_pred.get('30d', []),
            'combo': combo_pred,
            'feature_imp': imp_pred
        }
        
        # 加权融合预测
        final_prediction = self._fuse_predictions(all_predictions)
        
        return final_prediction, all_predictions
    
    def _frequency_prediction(self, features):
        """基于频率的预测"""
        # 获取各生肖频率特征
        zodiac_freq = {z: features[f'freq_{z}'] for z in self.zodiacs if f'freq_{z}' in features}
        if not zodiac_freq:
            return []
        sorted_zodiac = sorted(zodiac_freq.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:3]]
    
    def _transition_prediction(self, last_zodiac):
        """基于转移概率的预测"""
        if not self.combo_probs:
            return []
        
        # 获取各生肖转移概率
        trans_probs = {z: self.combo_probs.get(f"{last_zodiac}-{z}", 0) for z in self.zodiacs}
        sorted_zodiac = sorted(trans_probs.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:3]]
    
    def _season_prediction(self, features):
        """基于季节的预测 - 修复类型错误"""
        # 获取季节特征 - 确保传入的是Series或dict
        if not isinstance(features, (pd.Series, dict)):
            print(f"错误: 季节预测需要Series或dict, 得到 {type(features)}")
            return []
            
        season_probs = {z: features[f'season_{z}'] for z in self.zodiacs 
                       if f'season_{z}' in features}
        if not season_probs:
            return []
        sorted_zodiac = sorted(season_probs.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:2]]
    
    def _festival_prediction(self, features):
        """基于节日的预测 - 修复类型错误"""
        # 检查是否是节日
        if not features.get('is_festival', False):
            return []
        
        # 获取节日特征 - 确保传入的是Series或dict
        if not isinstance(features, (pd.Series, dict)):
            print(f"错误: 节日预测需要Series或dict, 得到 {type(features)}")
            return []
            
        festival_probs = {z: features[f'festival_{z}'] for z in self.zodiacs 
                        if f'festival_{z}' in features}
        if not festival_probs:
            return []
        sorted_zodiac = sorted(festival_probs.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:2]]
    
    def _rolling_prediction(self, features):
        """基于滚动窗口的预测 - 修复Series比较错误"""
        # 7天滚动窗口
        rolling_7d = {}
        for z in self.zodiacs:
            feature_name = f'rolling_7d_{z}'
            if feature_name in features:
                # 确保值是标量而非Series
                value = features[feature_name]
                if isinstance(value, pd.Series):
                    # 取最后一个值
                    value = value.iloc[-1] if not value.empty else 0
                rolling_7d[z] = value
        
        sorted_7d = sorted(rolling_7d.items(), key=lambda x: x[1], reverse=True)
        pred_7d = [z for z, _ in sorted_7d[:2]] if sorted_7d else []
        
        # 30天滚动窗口
        rolling_30d = {}
        for z in self.zodiacs:
            feature_name = f'rolling_30d_{z}'
            if feature_name in features:
                # 确保值是标量而非Series
                value = features[feature_name]
                if isinstance(value, pd.Series):
                    # 取最后一个值
                    value = value.iloc[-1] if not value.empty else 0
                rolling_30d[z] = value
        
        sorted_30d = sorted(rolling_30d.items(), key=lambda x: x[1], reverse=True)
        pred_30d = [z for z, _ in sorted_30d[:2]] if sorted_30d else []
        
        return {'7d': pred_7d, '30d': pred_30d}
    
    def _importance_weighted_prediction(self, features):
        """基于特征重要性的预测"""
        if self.feature_importance is None:
            return []
        
        # 只考虑前10个重要特征
        top_features = self.feature_importance.head(10)
        feature_scores = defaultdict(float)
        
        # 计算每个生肖的特征加权得分
        for feature, imp in top_features.items():
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
            weight = self.weights[factor]
            for i, zodiac in enumerate(preds):
                # 排名越高，得分越高（指数衰减）
                score = weight * (0.5 ** i)
                zodiac_scores[zodiac] += score
        
        # 按得分排序
        sorted_zodiac = sorted(zodiac_scores.items(), key=lambda x: x[1], reverse=True)
        return [z for z, _ in sorted_zodiac[:5]]  # 返回前5个预测结果
    
    @property
    def zodiacs(self):
        """生肖列表属性"""
        return ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]
