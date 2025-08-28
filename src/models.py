# %% [markdown]
# # GPT-2 Model
# 

# %%
import sys
import os


try:
    get_ipython
    current_dir = os.getcwd()
except NameError:
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Set path，temporary path expansion
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
import torch
import torch.nn as nn 
from src import tool ,model_wrapper

# %% [markdown]
# ### Config

# %%
IS_SKIP_TEST =False

# %% [markdown]
# ## Define GPT-2 Model

# %%
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.initializer_range =0.02
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
        self.apply(self._init_weights)
       
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
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 基础初始化：均值0，标准差initializer_range
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    


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
    "context_len": 1024,  
    "emb_dim": 768,          
    "n_heads": 8,          
    "n_layers": 12,          
    "drop_rate": 0.1,      
    'initializer_range':0.02, 
    "qkv_bias": True ,      
    }   
    model = GPTModel(CONFIG)
 

    # multi attention_new 参数减少量 = (304,556,544 - 163,008,000)
    total_params =sum(p.numel() for p in model.parameters())

    print(f"Total number of parameters: {total_params:,}") # 163,008,000

    #权重共享， W_emb和W_out指向同一块内存，模型训练时只会更新这一个矩阵，避免了维护两个独立矩阵的开销
    total_params_gpt2 = total_params - sum(p.numel()for p in model.out_head.parameters())
   
    print(f"Number of trainable parameters "
        f"considering weight tying: {total_params_gpt2:,}") #124,017,408  -->gpt2 124m
    return model
    
test_GPT2_model()

# %% [markdown]
# ## Define GPT-2 Model with KVCache

# %%
class GPTModel_KVCache(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.initializer_range =0.02
        self.vocab_size = cfg['vocab_size']
        self.context_len = cfg['context_len']
        self.padding_idx = cfg.get('padding_idx', 0)
         
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'],padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(cfg['context_len'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(  # 不再使用nn.Sequential，以便传递缓存,允许我们手动控制每个模块的输入输出和状态传递
            [model_wrapper.TransformerBlock_KVCache(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = model_wrapper.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg['emb_dim'],cfg['vocab_size'],bias=False
        )
        self.apply(self._init_weights)
       
    def forward(self,in_idx, past_kvs=None, use_cache=False, attention_mask=None):
        batch_size, new_seq_len = in_idx.shape  
        if attention_mask is None:
            # 自动生成掩码：1表示有效token，0表示padding
            attention_mask = (in_idx != self.padding_idx).float()
        assert attention_mask.shape == (batch_size, new_seq_len), \
            f"attention_mask形状错误，应为({batch_size}, {new_seq_len})"
        
        # 计算输入序列的位置索引（考虑历史缓存长度）
        if past_kvs is None:
            # 首次调用，位置从0开始
            start_pos = 0
        else:
            # 非首次调用，位置从历史长度开始（取第一层缓存的长度）
            start_pos = past_kvs[0][0].size(2) if past_kvs else 0
        end_pos = start_pos + new_seq_len
        assert end_pos <= self.context_len, f"input sequence exceeds the maximum context length {self.context_len}"
        
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(start_pos, end_pos, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        if past_kvs is None:
            past_kvs = [None] * len(self.trf_blocks)  # 初始化空缓存列表
        
        present_kvs = [] if use_cache else None
        for block, past_kv in zip(self.trf_blocks, past_kvs):
            x, present_kv = block(x, 
                                  past_kv=past_kv, 
                                  use_cache=use_cache,
                                  attention_mask=attention_mask )
            if use_cache:
                present_kvs.append(present_kv)
  
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, present_kvs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 基础初始化：均值0，标准差initializer_range
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
      

# %% [markdown]
# ## Define GPT-2 Model（MOE） with KVCache

# %%
class GPTModel_MOE_KVCache(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.initializer_range =0.02
        self.vocab_size = cfg['vocab_size']
        self.context_len = cfg['context_len']
        self.padding_idx = cfg.get('padding_idx', 0)
         
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'],padding_idx=self.padding_idx)
        self.pos_emb = nn.Embedding(cfg['context_len'],cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(  # 不再使用nn.Sequential，以便传递缓存,允许我们手动控制每个模块的输入输出和状态传递
            [model_wrapper.TransformerBlock_MOE_KVCache(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = model_wrapper.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg['emb_dim'],cfg['vocab_size'],bias=False
        )
        self.apply(self._init_weights)
       
    def forward(self,in_idx, past_kvs=None, use_cache=False, attention_mask=None):
        batch_size, new_seq_len = in_idx.shape  
        if attention_mask is None:
            # 自动生成掩码：1表示有效token，0表示padding
            attention_mask = (in_idx != self.padding_idx).float()
        assert attention_mask.shape == (batch_size, new_seq_len), \
            f"attention_mask形状错误，应为({batch_size}, {new_seq_len})"
        
        # 计算输入序列的位置索引（考虑历史缓存长度）
        if past_kvs is None:
            # 首次调用，位置从0开始
            start_pos = 0
        else:
            # 非首次调用，位置从历史长度开始（取第一层缓存的长度）
            start_pos = past_kvs[0][0].size(2) if past_kvs else 0
        end_pos = start_pos + new_seq_len
        assert end_pos <= self.context_len, f"input sequence exceeds the maximum context length {self.context_len}"
        
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(start_pos, end_pos, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        if past_kvs is None:
            past_kvs = [None] * len(self.trf_blocks)  # 初始化空缓存列表
        
        present_kvs = [] if use_cache else None
        for block, past_kv in zip(self.trf_blocks, past_kvs):
            x, present_kv = block(x, 
                                  past_kv=past_kv, 
                                  use_cache=use_cache,
                                  attention_mask=attention_mask )
            if use_cache:
                present_kvs.append(present_kv)
  
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, present_kvs
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 基础初始化：均值0，标准差initializer_range
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
      


