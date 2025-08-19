import pandas as pd
import numpy as np
import torch
import pickle
from pathlib import Path
from evo2 import Evo2

# 常量定义
LAYERS = ['blocks.24.mlp.l3', 'blocks.26.mlp.l3',
          'blocks.28.mlp.l3', 'blocks.30.mlp.l3']
BASE_NAMES = ['128bp', '256bp', '512bp', '1024bp']
INPUT_DIR = "/home/user/03.shang_gao/12.semi_supervised_learning/01.generate_input_sequence/"
OUTPUT_DIR = "/home/user/03.shang_gao/12.semi_supervised_learning/02.generate_embedding_1/"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 初始化模型 (单例模式)
_model = None
_tokenizer = None

def initialize_model():
    """安全初始化模型"""
    global _model, _tokenizer
    if _model is None:
        print(f"Loading Evo2 model on {DEVICE}...")
        
        # 1. 创建模型实例
        _model = Evo2('evo2_7b')
        
        # # 2. 手动将模型参数移动到目标设备
        # for param in _model.parameters():  # 遍历所有参数
        #     param.data = param.data.to(DEVICE)
        #     if param.requires_grad and param.grad is not None:
        #         param.grad.data = param.grad.data.to(DEVICE)
        
        # # 3. 设置评估模式
        # _model.eval()
        
        # 4. 初始化分词器
        _tokenizer = _model.tokenizer
        
        print(f"Model successfully loaded on {DEVICE}")
    return _model, _tokenizer

def get_dna_embedding(sequence: str, layer_name: str) -> torch.Tensor:
    """安全获取序列嵌入"""
    model, tokenizer = initialize_model()
    
    try:
        # 生成设备兼容的输入
        input_ids = torch.tensor(
            tokenizer.tokenize(sequence),
            dtype=torch.int,
        ).unsqueeze(0).to(DEVICE)
        
        # 前向计算
        with torch.no_grad():
            _, embeddings = model(input_ids, 
                                 return_embeddings=True, 
                                 layer_names=[layer_name])
            
        return embeddings[layer_name]
    except RuntimeError as e:
        print(f"Error processing sequence: {str(e)}")
        return torch.zeros(1, 1, 2560).to(DEVICE)  # 返回空嵌入保持维度

def process_sequences(df: pd.DataFrame, base_name: str):
    """带错误处理的序列处理"""
    sequences = df['sequence'].tolist()
    
    for layer in LAYERS:
        print(f"Processing {base_name} | Layer {layer}")
        
        layer_embeddings = []
        for i, seq in enumerate(sequences):
            try:
                # 获取嵌入
                emb = get_dna_embedding(seq, layer)
                
                # 确保数据在CPU且为numpy兼容类型
                emb = emb.mean(dim=1).squeeze().cpu()
                
                # 转换为numpy前检查类型
                if emb.dtype == torch.bfloat16:
                    emb = emb.to(torch.float32)
                    
                emb_np = emb.numpy()
                layer_embeddings.append(emb_np)
                
                # 每100个序列打印进度
                if (i+1) % 100 == 0:
                    print(f"Processed {i+1}/{len(sequences)} sequences")
                    
            except Exception as e:
                print(f"Error at sequence {i}: {str(e)}")
                layer_embeddings.append(np.zeros(2560))  # 保持维度一致
        
        # 保存结果
        emb_df = pd.DataFrame(layer_embeddings)
        emb_df['gene_id'] = df['gene_id']
        
        output_path = Path(OUTPUT_DIR) / f"{base_name}_layer{layer.split('.')[1]}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(emb_df, f)
            
        print(f"Saved {output_path} with {len(emb_df)} embeddings")

def main():
    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # 初始化模型
    initialize_model()
    
    # 处理所有文件
    for bp in BASE_NAMES:
        input_file = Path(INPUT_DIR) / f"interseq_dist1000_effobs1000_{bp}.csv"
        try:
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} sequences from {input_file}")
            process_sequences(df, bp)
        except FileNotFoundError:
            print(f"Warning: File {input_file} not found, skipping")

if __name__ == "__main__":
    main()
