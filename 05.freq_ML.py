#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import gc
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# ----------------------
# 配置参数
# ----------------------
CONFIG = {
    "data_path": "/home/user/03.shang_gao/12.semi_supervised_learning/dinuc_data_for_jinran.csv",
    "output_dir": "/home/user/03.shang_gao/12.semi_supervised_learning/06.freq_ML",
    "param_grid": {
        "label_ratio": np.round(np.arange(0.01, 0.15, 0.005), 2),
        "learning_rate": np.round(np.arange(0.01, 0.11, 0.02), 2),
        "rounds": range(100, 101, 100)
    },
    "fixed_params": {
        "test_size": 0.2,
        "random_state": 2023,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 30,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 3
    }
}

# ----------------------
# 核心功能函数
# ----------------------
def dynamic_labeling(series, ratio):
    """根据ratio生成标签"""
    upper = series.quantile(1 - ratio)
    lower = series.quantile(ratio)
    return upper, lower

# ----------------------
# 训练流程
# ----------------------
def train_entry_model(X_train, y_train, config):
    """监督学习模型训练"""
    try:
        train_data = lgb.Dataset(X_train, y_train)
        
        model = lgb.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "learning_rate": config['learning_rate'],
                "num_leaves": config['num_leaves'],
                "max_depth": config['max_depth'],
                "min_child_samples": config['min_child_samples'],
                "lambda_l1": config['lambda_l1'],
                "lambda_l2": config['lambda_l2'],
                "feature_fraction": config['feature_fraction'],
                "bagging_fraction": config['bagging_fraction'],
                "bagging_freq": config['bagging_freq']
            },
            train_data,
            num_boost_round=config['rounds']
        )
        return model
    except Exception as e:
        print(f"训练错误: {str(e)}")
        return None

def train_pipeline(config):
    """训练流水线"""
    df = pd.read_csv(config['data_path'])
    X = df.iloc[:, 1:241].values
    lab = df.iloc[:, 241:].copy()

    # 生成标签
    upper, lower = dynamic_labeling(lab['cor_log'], config['label_ratio'])
    lab['y'] = np.select(
        [lab['cor_log'] > upper, lab['cor_log'] < lower], 
        [1, 0], 
        np.nan  # 中间区域不参与训练
    )
    
    # 只保留有标签的数据
    labeled_mask = lab['y'].notna()
    X_labeled = X[labeled_mask]
    y_labeled = lab[labeled_mask]['y'].astype(int)
    
    # 数据分割
    if len(y_labeled) < 10:
        print(f"有效数据不足: {len(y_labeled)}")
        return None
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_labeled, y_labeled,
        test_size=config['test_size'],
        stratify=y_labeled,
        random_state=config['random_state']
    )
    
    # 训练模型
    model = train_entry_model(X_train, y_train, config)
    if model is None:
        return None
    
    # 模型评估
    test_pred = model.predict(X_test)
    y_pred = (test_pred >= 0.5).astype(int)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, test_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred).ravel(),
        'label_dist': {
            'positive': len(y_labeled[y_labeled == 1]),
            'negative': len(y_labeled[y_labeled == 0]),
            'total': len(y_labeled)
        }
    }

# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(CONFIG['output_dir'], f"results_{timestamp}.csv")
    
    param_combinations = list(itertools.product(
        CONFIG['param_grid']['label_ratio'],
        CONFIG['param_grid']['learning_rate'],
        CONFIG['param_grid']['rounds']
    ))
    
    all_results = []
    with tqdm(param_combinations, desc="网格搜索进度") as pbar:
        for lr, lr_rate, n_rounds in pbar:
            current_config = {
                "data_path": CONFIG["data_path"],
                "output_dir": CONFIG["output_dir"],
                **CONFIG['fixed_params'],
                "label_ratio": float(lr),
                "learning_rate": float(lr_rate),
                "rounds": int(n_rounds)
            }
            
            result = train_pipeline(current_config)
            
            if result:
                record = {
                    'label_ratio': lr,
                    'learning_rate': lr_rate,
                    'rounds': n_rounds,
                    'positive': result['label_dist']['positive'],
                    'negative': result['label_dist']['negative'],
                    'total': result['label_dist']['total'],
                    'accuracy': round(result['accuracy'], 4),
                    'auc': round(result['auc'], 4),
                    'f1': round(result['f1'], 4),
                    'TN': result['confusion_matrix'][0],
                    'FP': result['confusion_matrix'][1],
                    'FN': result['confusion_matrix'][2],
                    'TP': result['confusion_matrix'][3]
                }
                all_results.append(record)
                pd.DataFrame(all_results).to_csv(result_file, index=False)
                
            gc.collect()

    print(f"\n执行完成! 结果文件: {result_file}")
