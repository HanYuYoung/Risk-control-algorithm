"""
模型训练脚本 - 从原始数据加载并提取特征训练
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from tqdm import tqdm
from config import MODEL_CONFIG, DATA_DIR
from models.feature_extractor import FeatureExtractor

def load_and_extract_features(data_file=None):
    """从CSV文件加载原始数据并提取特征"""
    if data_file is None:
        data_file = os.path.join(DATA_DIR, 'training_data.csv')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"训练数据文件不存在: {data_file}\n"
            f"请先运行: python generate_data.py"
        )
    
    print(f"加载训练数据: {data_file}")
    df = pd.read_csv(data_file)
    
    # 按时间戳排序，确保聚集度特征能正确计算
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"  原始样本数: {len(df)}")
    print(f"  正常样本: {np.sum(df['label']==0)}")
    print(f"  异常样本: {np.sum(df['label']==1)}")
    
    # 提取特征
    print("\n提取特征...")
    extractor = FeatureExtractor()
    X = []
    y = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="特征提取", unit="样本"):
        # 重构请求数据格式
        request_data = {
            'phone': str(row['phone']),
            'ip': str(row['ip']),
            'device_id': str(row['device_id']),
            'user_agent': str(row.get('user_agent', '')),
            'timestamp': float(row['timestamp']),
            'device_fingerprint': {
                'canvas_fingerprint': str(row.get('canvas_fingerprint', '')),
                'fonts': str(row.get('fonts', '')),
                'screen_resolution': str(row.get('screen_resolution', '1920x1080')),
                'timezone': str(row.get('timezone', 'Asia/Shanghai')),
                'language': str(row.get('language', 'zh-CN')),
            },
            'behavior': {
                'page_stay_time': float(row['page_stay_time']),
                'click_count': int(row['click_count']),
                'scroll_count': int(row['scroll_count']),
                'path_entropy': float(row['path_entropy']),
                'mouse_trajectory_entropy': float(row.get('mouse_trajectory_entropy', 0.5)),
                'request_frequency': float(row.get('request_frequency', 1.0)),
            }
        }
        
        features = extractor.extract_features(request_data)
        X.append(features)
        y.append(int(row['label']))
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[OK] 特征提取完成")
    print(f"  特征维度: {X.shape[1]} (增强版)")
    
    # 保存特征数据到CSV
    feature_names = [
        # 聚集度特征（8）
        'ip_reg_count', 'device_reg_count', 'ip_device_count', 'device_ip_count',
        'phone_device_count', 'ip_phone_count', 'same_ip_different_devices', 'same_device_different_phones',
        # 行为特征（13）- 实际有13维
        'page_stay_time', 'click_count', 'scroll_count', 'path_entropy',
        'mouse_trajectory_entropy', 'request_frequency', 'clicks_per_second', 'scrolls_per_second',
        'clicks_per_scroll', 'behavior_diversity', 'short_stay', 'low_interaction', 'high_frequency',
        # 账号特征（5）
        'phone_in_blacklist', 'phone_history_count', 'phone_segment', 'phone_is_virtual', 'phone_reg_time_span',
        # 时间特征（6）
        'hour', 'minute', 'day_of_week', 'is_workday', 'is_peak_hour', 'is_night',
        # 设备指纹特征（8）
        'device_hash', 'ip_hash', 'canvas_hash', 'is_mobile', 'is_chrome', 'is_safari',
        'screen_area', 'device_fingerprint_uniqueness',
        # 网络特征（4）
        'estimated_latency', 'is_proxy_likely', 'ip_reputation_score', 'network_segment',
        # 关联特征（4）
        'ip_geolocation_consistency', 'registration_pattern_anomaly', 'cluster_score', 'multi_device_flag',
        # 组合特征（5）
        'risk_score_base', 'behavior_anomaly_score', 'device_anomaly_score', 'timing_anomaly_score', 'comprehensive_risk_score'
    ]
    
    # 创建DataFrame并保存
    feature_df = pd.DataFrame(X, columns=feature_names)
    feature_df['label'] = y
    
    feature_file = os.path.join(DATA_DIR, 'features_data.csv')
    feature_df.to_csv(feature_file, index=False)
    print(f"[OK] 特征数据已保存: {feature_file}")
    
    return X, y

def train_xgboost(X, y, model_path):
    """训练XGBoost模型"""
    print("\n训练XGBoost模型...")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    print(f"[OK] 训练完成")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  验证集准确率: {val_score:.4f}")
    
    joblib.dump(model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return model

def train_isolation_forest(X, model_path, contamination=0.2):
    """训练IsolationForest模型"""
    print("\n训练IsolationForest模型...")
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X)
    
    predictions = model.predict(X)
    anomaly_ratio = (predictions == -1).sum() / len(predictions)
    
    print(f"[OK] 训练完成")
    print(f"  异常样本比例: {anomaly_ratio:.4f}")
    
    joblib.dump(model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return model

def main():
    """主训练函数"""
    print("=" * 60)
    print("风控模型训练")
    print("=" * 60)
    
    # 加载数据并提取特征
    X, y = load_and_extract_features()
    
    # 训练模型
    train_xgboost(X, y, MODEL_CONFIG['xgb_model_path'])
    train_isolation_forest(X[y == 0], MODEL_CONFIG['isolation_forest_path'])
    
    print("\n" + "=" * 60)
    print("[OK] 所有模型训练完成")
    print("=" * 60)

if __name__ == '__main__':
    main()
