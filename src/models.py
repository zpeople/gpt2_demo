# %% [markdown]
# # GPT-2 Model
# 

# %%

import sys
import os

# 检查是否在Jupyter环境中
try:
    # 如果是Jupyter，会定义这个变量
    get_ipython
    # 使用当前工作目录作为基准（Jupyter的工作目录）
    current_dir = os.getcwd()
except NameError:
    # 普通Python环境，使用__file__
    current_dir = os.path.dirname(os.path.abspath(__file__))

# 将父目录添加到Python路径（根据你的目录结构调整）
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
import torch
import torch.nn as nn 
from src import tool ,model_wrapper



# %% [markdown]
# ### Config

# %%
IS_SKIP_TEST =True

GPT_CONFIG = {
    "num_epochs":10,
    "batch_size":4,
    "vocab_size": 50257,     # 词汇表大小
    "context_len": 256,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 8,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # dropout率
    "qkv_bias": False ,      # 查询-键-值偏置
}


# %% [markdown]
# ### Set device to (type='cuda')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ## Define GPT-2 Model

# %%
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_len'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks =  nn.Sequential(
            *[model_wrapper.TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = model_wrapper.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg['emb_dim'],cfg['vocab_size'],bias=False
        )
       
    def forward(self,in_idx):
        batch_size, seq_len = in_idx.shape  
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))  
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
        

# %% [markdown]
# ### View structure of model 

# %%
#GPT2 小型（Small）：12 层 Transformer 解码器，隐藏层维度 768，注意力头数 12，总参数约 1.2 亿
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_GPT2_model():
    CONFIG = {
    "num_epochs":1,
    "batch_size":1,
    "vocab_size": 50257,     
    "context_len": 512,  
    "emb_dim": 768,          
    "n_heads": 8,          
    "n_layers": 12,          
    "drop_rate": 0.1,       
    "qkv_bias": False ,      
    }   
    model = GPTModel(CONFIG)
    model.to(device)

    # attention_new 参数减少量 = (304,556,544 - 163,008,000)
    total_params =sum(p.numel() for p in model.parameters())

    print(f"Total number of parameters: {total_params:,}") #163,008,000

    #权重共享， W_emb和W_out指向同一块内存，模型训练时只会更新这一个矩阵，避免了维护两个独立矩阵的开销
    total_params_gpt2 = total_params - sum(p.numel()for p in model.out_head.parameters())
   
    print(f"Number of trainable parameters "
        f"considering weight tying: {total_params_gpt2:,}") #124,017,408
    return model
    
test_GPT2_model()


