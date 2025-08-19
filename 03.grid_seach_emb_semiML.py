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
from scipy.optimize import minimize_scalar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

# ----------------------
# 配置参数
# ----------------------
CONFIG = {
    "feature_dir": "/home/user/03.shang_gao/12.semi_supervised_learning/02.generate_embedding_1/",
    "label_path": "/home/user/03.shang_gao/12.semi_supervised_learning/01.generate_input_sequence/interseq_dist1000_effobs1000_pair_info.csv",
    "output_dir": "/home/user/03.shang_gao/12.semi_supervised_learning/03.grid_seach_emb_semiML/",
    "param_grid": {
        "label_ratio": np.round(np.arange(0.01, 0.15, 0.005), 2),
        "learning_rate": np.round(np.arange(0.01, 0.11, 0.02), 2),
        "rounds": range(100, 101, 100)
    },
    "fixed_params": {
        "lambda_weight": 1,
        "test_size": 0.2,
        "random_state": 2023,
        "num_leaves": 127,
        "max_depth": 9,
        "min_child_samples": 30,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 3,
        "max_em_iter": 15
    }
}

# ----------------------
# 核心功能函数
# ----------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

def compute_entropy(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def estimate_xi(p_hat, s):
    """ξ参数估计"""
    H = compute_entropy(p_hat)
    log_H = np.log(H + 1e-8)
    
    valid_mask = np.isfinite(log_H) & (H > 1e-6) & (s != -1)
    if valid_mask.sum() < 10:
        return np.array([0.0, 0.0])
    
    try:
        X = np.column_stack([np.ones(valid_mask.sum()), log_H[valid_mask]])
        model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        model.fit(X, s[valid_mask])
        return np.array([model.intercept_[0], model.coef_[0][1]])
    except Exception as e:
        print(f"ξ估计失败: {str(e)}")
        return np.array([0.0, 0.0])

def custom_loss(xi, lambda_weight=1):
    """梯度计算函数"""
    def _loss(preds, train_data):
        y = train_data.get_label()
        s = train_data.get_weight()
        
        p = sigmoid(preds)
        eps = 1e-8
        
        # 监督部分梯度
        grad_sup = np.where(s == 1, p - y, 0.0)
        hess_sup = np.where(s == 1, p * (1 - p), 0.0)
        
        # 无监督部分梯度
        log_ratio = np.log(p + eps) - np.log(1 - p + eps)
        grad_unsup = np.where(s == 0, log_ratio, 0.0)
        hess_unsup = np.where(s == 0, 1/(p*(1-p)+eps), 0.0)
        
        # 缺失机制梯度
        H = compute_entropy(p)
        log_H = np.log(H + eps)
        sigma = sigmoid(xi[0] + xi[1] * log_H)
        log_diff = np.log(1 - p + eps) - np.log(p + eps)
        
        grad_miss = lambda_weight * xi[1] * (sigma - s) * log_diff
        hess_miss = lambda_weight * (xi[1]**2) * sigma * (1 - sigma) * (log_diff**2)
        
        grad = grad_sup + grad_unsup - grad_miss
        hess = hess_sup + hess_unsup + hess_miss
        
        return grad, hess
    return _loss

def dynamic_labeling(series, ratio):
    """直接根据ratio设置分位数阈值"""
    try:
        upper = series.quantile(1 - ratio)
        lower = series.quantile(ratio)
        return upper, lower
    except:
        return series.quantile(0.9), series.quantile(0.1)

# ----------------------
# 训练流程
# ----------------------
def train_entry_model(X, y, s, config):
    """模型训练主函数"""
    try:
        model_params = [
            'num_leaves', 'max_depth', 'min_child_samples',
            'lambda_l1', 'lambda_l2', 'feature_fraction',
            'bagging_fraction', 'bagging_freq'
        ]
        
        # 初始训练
        init_mask = (s == 1)
        if np.sum(init_mask) < 10:
            raise ValueError(f"有效标签不足: {np.sum(init_mask)}")
            
        init_data = lgb.Dataset(
            X[init_mask], 
            y[init_mask],
            weight=np.ones(init_mask.sum()),
            free_raw_data=False
        )
        
        model = lgb.train(
            {
                "objective": "binary",
                "verbosity": -1,
                "learning_rate": 0.1,
                **{k: config[k] for k in model_params}
            },
            init_data,
            num_boost_round=100
        )
        
        # EM迭代
        xi = np.array([0.0, 1.0])
        full_data = lgb.Dataset(X, label=y, weight=s, free_raw_data=False)
        
        for em_iter in range(config['max_em_iter']):
            p_hat = sigmoid(model.predict(X, raw_score=True))
            xi_new = estimate_xi(p_hat, s)
            
            if np.linalg.norm(xi_new - xi) < 1e-4:
                break
            xi = xi_new
            
            model = lgb.train(
                {
                    "objective": "custom",
                    "learning_rate": config['learning_rate'],
                    "verbosity": -1,
                    **{k: config[k] for k in model_params}
                },
                full_data,
                num_boost_round=config['rounds'],
                fobj=custom_loss(xi, config['lambda_weight']),
                init_model=model,
                keep_training_booster=True
            )
            
        return model
    except Exception as e:
        # 打印完整堆栈跟踪
        error_msg = (
            f"训练错误详情:\n"
            f"异常类型: {type(e).__name__}\n"
            f"异常信息: {str(e)}\n"
            f"堆栈跟踪:\n{traceback.format_exc()}"
        )
        print(error_msg)
        
        # 打印数据调试信息
        print("\n数据调试信息:")
        print(f"输入矩阵形状: {X.shape}")
        print(f"首行数据类型: {type(X[0][0])}")
        print(f"首行数据示例: {X[0][:5]}")
        
        return None

# ----------------------
# 修改后的训练流水线
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
    
    # # 合并数据
    # merged_df = feature_df.merge(label_df[['gene_id', 'cor_log']], on='gene_id', how='inner')
    
    # # 提取特征矩阵和标签
    # feature_cols = [col for col in feature_df.columns if col != 'gene_id']
    # X = merged_df[feature_cols].values
    # lab = merged_df[['cor_log']].copy()
    X = feature_df.iloc[:, 0:-1].values
    lab = label_df[['cor_log']]
    
    return X, lab, length, layer

def train_pipeline(feature_path):
    """单特征文件训练流水线"""
    # 加载数据
    X, lab, length, layer = load_feature_data(feature_path)
    
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
            
            lab['y'] = np.select(
                [lab['cor_log'] > upper, lab['cor_log'] < lower], 
                [1, 0], 
                np.nan
            )
            lab['s'] = lab['y'].notna().astype(int)
            
            # 数据分割
            labeled_data = lab[lab['s'] == 1]
            X_labeled = X[lab['s'] == 1]
            y_labeled = labeled_data['y'].values
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_labeled, y_labeled,
                    test_size=current_config['test_size'],
                    stratify=y_labeled,
                    random_state=current_config['random_state']
                )
            except ValueError as e:
                print(f"数据分割失败: {str(e)}")
                continue
                
            # 合并数据
            X_full = np.vstack([X_train, X[lab['s'] == 0]])
            y_full = np.concatenate([y_train, np.full(X[lab['s'] == 0].shape[0], np.nan)])
            s_full = np.concatenate([np.ones_like(y_train), np.zeros(X[lab['s'] == 0].shape[0])])
            
            # 训练模型
            model = train_entry_model(X_full, np.nan_to_num(y_full), s_full, current_config)
            if model is None:
                continue
                
            # 评估模型
            test_pred = sigmoid(model.predict(X_test))
            y_pred = (test_pred >= 0.5).astype(int)
            
            record = {
                'length': length,
                'layer': layer,
                'label_ratio': lr,
                'learning_rate': lr_rate,
                'rounds': n_rounds,
                'positive': lab['y'].sum(),
                'negative': (lab['y'] == 0).sum(),
                'unlabeled': lab['y'].isna().sum(),
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
            
    # print("\n全部特征文件处理完成！")


    # 遍历处理每个特征文件
    # for feature_path in feature_files[0:1]:
    #    train_pipeline(feature_path)
            
    print("\n全部特征文件处理完成！")

