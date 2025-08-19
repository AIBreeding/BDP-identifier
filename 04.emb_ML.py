#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import gc
import glob
import itertools
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
import traceback
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# ----------------------
# 配置参数
# ----------------------
CONFIG = {
    "feature_dir": "/home/user/03.shang_gao/12.semi_supervised_learning/02.generate_embedding_1/",
    "label_path": "/home/user/03.shang_gao/12.semi_supervised_learning/01.generate_input_sequence/interseq_dist1000_effobs1000_pair_info.csv",
    "output_dir": "/home/user/03.shang_gao/12.semi_supervised_learning/05.emb_ML/",
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
    try:
        upper = series.quantile(1 - ratio)
        lower = series.quantile(ratio)
        return upper, lower
    except:
        return series.quantile(0.9), series.quantile(0.1)

# ----------------------
# 修改后的训练流程
# ----------------------
def load_feature_data(feature_path):
    """加载特征数据和标签"""
    # 加载特征
    with open(feature_path, 'rb') as f:
        feature_df = pickle.load(f)
    
    # 解析文件名信息
    fname = Path(feature_path).stem
    length, layer = fname.split('_layer')
    length = length.replace('bp', '')
    
    # 加载标签数据
    label_df = pd.read_csv(CONFIG['label_path'])
    
    # 合并数据
    X = feature_df.iloc[:, 0:-1].values
    lab = label_df[['cor_log']]
    
    return X, lab, length, layer

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
        error_msg = (
            f"训练错误详情:\n"
            f"异常类型: {type(e).__name__}\n"
            f"异常信息: {str(e)}\n"
            f"堆栈跟踪:\n{traceback.format_exc()}"
        )
        print(error_msg)
        return None

def train_pipeline(feature_path):
    """单特征文件训练流水线"""
    # 加载数据
    X, lab, length, layer = load_feature_data(feature_path)
    
    # 过滤特定参数：1024bp 和 24层
    if not (length == '1024' and layer == '24'):
        print(f"跳过非目标参数文件：{length}bp_layer{layer}")
        return
    
    # 生成结果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        CONFIG['output_dir'],
        f"results_{length}bp_layer{layer}_{timestamp}.csv"
    )
    
    # 生成参数组合
    param_combinations = list(itertools.product(
        CONFIG['param_grid']['label_ratio'],
        CONFIG['param_grid']['learning_rate'],
        CONFIG['param_grid']['rounds']
    ))
    
    all_results = []
    with tqdm(param_combinations, desc=f"Training {Path(feature_path).name}") as pbar:
        for lr, lr_rate, n_rounds in pbar:
            current_config = {
                **CONFIG['fixed_params'],
                "label_ratio": float(lr),
                "learning_rate": float(lr_rate),
                "rounds": int(n_rounds)
            }
            
            # 标签生成
            upper, lower = dynamic_labeling(lab['cor_log'], current_config['label_ratio'])
            
            # 生成二分类标签
            y = np.select(
                [lab['cor_log'] > upper, lab['cor_log'] < lower], 
                [1, 0], 
                np.nan  # 中间区域不参与训练
            )
            
            # 过滤有效数据
            valid_mask = ~np.isnan(y)
            X_valid = X[valid_mask]
            y_valid = y[valid_mask].astype(int)
            
            if len(np.unique(y_valid)) < 2:
                print(f"有效数据不足或只有单一类别: {len(y_valid)}")
                continue
                
            # 数据分割
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_valid, y_valid,
                    test_size=current_config['test_size'],
                    stratify=y_valid,
                    random_state=current_config['random_state']
                )
            except ValueError as e:
                print(f"数据分割失败: {str(e)}")
                continue
                
            # 训练模型
            model = train_entry_model(X_train, y_train, current_config)
            if model is None:
                continue
                
            # 评估模型
            test_pred = model.predict(X_test)
            y_pred = (test_pred >= 0.5).astype(int)
            
            record = {
                'length': length,
                'layer': layer,
                'label_ratio': lr,
                'learning_rate': lr_rate,
                'rounds': n_rounds,
                'positive': sum(y_valid == 1),
                'negative': sum(y_valid == 0),
                'total': len(y_valid),
                'accuracy': round(accuracy_score(y_test, y_pred), 4),
                'auc': round(roc_auc_score(y_test, test_pred), 4),
                'f1': round(f1_score(y_test, y_pred), 4),
                'TN': confusion_matrix(y_test, y_pred).ravel()[0],
                'FP': confusion_matrix(y_test, y_pred).ravel()[1],
                'FN': confusion_matrix(y_test, y_pred).ravel()[2],
                'TP': confusion_matrix(y_test, y_pred).ravel()[3]
            }
            all_results.append(record)
            
            # 实时保存结果
            pd.DataFrame(all_results).to_csv(result_file, index=False)
            gc.collect()

# ----------------------
# 执行入口
# ----------------------
if __name__ == "__main__":
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 获取所有特征文件
    feature_files = glob.glob(os.path.join(CONFIG['feature_dir'], "*.pkl"))
    
    # 遍历处理每个特征文件
    for feature_path in feature_files:
        print(f"\nProcessing {Path(feature_path).name}")
        try:
            train_pipeline(feature_path)
        except Exception as e:
            print(f"处理文件 {feature_path} 时出错: {str(e)}")
            continue
            
    print("\n全部特征文件处理完成！")
