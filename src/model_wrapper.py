# %% [markdown]
# #  Model Wrapper
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
from torch.testing import assert_close
torch.manual_seed(42)
from src import tool
from tqdm import tqdm
import math


# %% [markdown]
# ### Config

# %%

IS_SKIP_TEST =True

TEST_CONFIG = {
    "num_epochs":10,
    "batch_size":4,
    "vocab_size": 50257,     # 词汇表大小
    "context_len": 256,  # 上下文长度
    "emb_dim": 512,          # 嵌入维度
    "n_heads": 8,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # dropout率
    "initializer_range":0.02,
    "qkv_bias": False ,      # 查询-键-值偏置
}

TOKEN_TYPE="gpt2"


# %% [markdown]
# ### Set device to (type='cuda')

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% [markdown]
# ## Define Test Model

# %%

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
    def forward(self,x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, norm_shape,eps=1e-5):
        super().__init__()
        
    def forward(self,x):
        return x
        

class DummyGPT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim']) #  “字典表”  (vocab_size, emb_dim) vocab_size 行，每一行对应一个 token 的emb_dim维的向量 
        self.pos_emb = nn.Embedding(cfg['context_len'],cfg['emb_dim']) # (context_len, emb_dim)
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks =  nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg['emb_dim'],cfg['vocab_size'],bias=False 
        )# y = x · W^T + b   W的形状为[vocab_size,emb_dim] 本质是计算x与W的相似度 ，得到vocab_size个y向量
       
    def forward(self,in_idx):
        #in_idx 通常是一个整数张量（Tensor），形状一般为 (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape  #in_idx 每个元素都是 token 的索引（范围是 [0, vocab_size-1])
        tok_embeds = self.tok_emb(in_idx) #查“字典表”映射  嵌入向量(batch_size, seq_len)-->(batch_size, seq_len, emb_dim) 

        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))  #生成一个从 0 到 seq_len-1 的整数序列 (seq_len,) -->(seq_len, emb_dim)
   
        x = tok_embeds + pos_embeds #pos_embeds会自动广播为 -->(batch_size, seq_len, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x) #-->(batch_size, seq_len, emb_dim)
        logits = self.out_head(x) #(batch_size, seq_len, emb_dim)-->(batch_size, seq_len, vocab_size)
        return logits
        
        

# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_dummyModel():
    model = DummyGPT(TEST_CONFIG)
    return model

test_dummyModel()

# %% [markdown]
# ## Define LayerNorm

# %%
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self,x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim =-1 ,keepdim =True, unbiased =False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x + self.shift

# %% [markdown]
# ### test layerNorm

# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_layer_norm():
    batch_size = 2
    seq_len = 5
    emb_dim = 3  
    x = torch.randn(batch_size, seq_len, emb_dim)  # 随机生成输入张量
    
    custom_ln = LayerNorm(emb_dim)
    official_ln = nn.LayerNorm(emb_dim, eps=1e-5, elementwise_affine=True)
    

    official_ln.weight.data.copy_(custom_ln.scale.data)
    official_ln.bias.data.copy_(custom_ln.shift.data)
 
    custom_out = custom_ln(x)
    official_out = official_ln(x)
    print(custom_out)
    print(official_out)

    assert_close(
        custom_out, 
        official_out, 
        rtol=1e-5,  # 相对误差容忍度
        atol=1e-5   # 绝对误差容忍度
    )
    print("自定义LayerNorm与官方实现输出一致")

test_layer_norm()

# %% [markdown]
# ## Define Activation Function
# 
# 高斯误差线性单元 GELU
# Φ(x) ≈ 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

# %%
class GELU(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,x):
        return 0.5*x*(1+ 
                      torch.tanh(torch.sqrt(torch.tensor(2/torch.pi))
                                 *(x+0.044715*torch.pow(x,3))
                                )
                      )
        


# %% [markdown]
# ### test gelu

# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_gelu():
    x = torch.tensor([-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0])
    
    custom_gelu = GELU()
    official_gelu = nn.GELU()

    custom_out = custom_gelu(x)
    official_out = official_gelu(x)
    
    # 打印结果进行直观对比
    print("输入值:", x)
    print("自定义GELU输出:", custom_out)
    print("官方GELU输出:", official_out)
 
    assert_close(
        custom_out,
        official_out,
        rtol=1e-3,  # 相对误差容忍度
        atol=1e-3   # 绝对误差容忍度
    )
    print("\n自定义GELU与官方实现近似一致")
    
test_gelu()

# %% [markdown]
# ## Define FFN
# 通过两层线性变换和激活函数，对注意力机制输出的特征进行非线性加工，增强模型表达能力。

# %%
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    
        #中间层hidden_dim通常设为4*emb_dim（如原始 Transformer 中为 512→2048→512），通过扩展维度捕捉更丰富的特征
        self.c_fc=nn.Linear(cfg['emb_dim'],4*cfg['emb_dim'])
        self.act=   GELU()
        self.dropout=   nn.Dropout(cfg['drop_rate'])
        self.c_proj=  nn.Linear(4*cfg['emb_dim'],cfg['emb_dim'])
        self.c_proj.weight.data.normal_(
            mean=0.0, 
            std=cfg['initializer_range'] / math.sqrt(2 * cfg['n_layers'])  # 层数相关的缩放
        )
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# %% [markdown]
# ## Define MultiAttention

# %%
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out,context_len,dropout,qkv_bias=False):
        super().__init__()
        self.d_out =d_out
        self.W_q = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.dropout = nn.Dropout(dropout)
        #缓冲区（buffer）是模型中不需要被训练的参数（与 nn.Parameter 不同，后者是可学习参数），但会随模型一起保存（state_dict 中包含）
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_len,context_len),diagonal=1)
        )
        
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)
        
        att_score = queries @ keys.transpose(1,2)
        att_score.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf) # 上面的register_buffer  形状为 (num_tokens, num_tokens) 的子矩阵
        att_weight = torch.softmax(att_score/keys.shape[-1]**0.5, dim=-1)
        att_weight = self.dropout(att_weight)
        context_vec = att_weight @ values
        return context_vec
        
class MultiHeadAttendtion(nn.Module):
    def __init__(self, d_in, d_out,context_len,dropout,num_heads,qkv_bias=False):
        super().__init__()
        # ModuleList与nn.Sequential不同，它不自动执行前向传播，而是需要手动遍历调用，适合需要单独处理每个子模块的场景
        self.heads = nn.ModuleList(
            [CausalAttention(d_in,d_out,context_len,dropout,qkv_bias) for _ in range(num_heads)]
        )
        
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads],dim=-1)

# %%
# TODO 更高效的MutiAttention 减少计算量

#参数规模更小（num_heads×d_model×d_model 对比 num_heads×d_model×head_dim)
class MultiHeadAttendtion_new(nn.Module):
    def __init__(self, d_in, d_out,context_len,dropout,num_heads,initializer_range,n_layer,qkv_bias=False):
        super().__init__()
        self.d_out =d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_q = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias= qkv_bias)
        self.c_proj =nn.Linear(d_out,d_out) # out_proj 可以学习如何 “融合” 这些头的信息（例如对不同头的特征赋予不同权重），而不是简单保留原始拼接结果
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', 
            torch.triu(torch.ones(context_len,context_len),diagonal=1)
        )
        self.c_proj.weight.data.normal_(
            mean=0.0, 
            std=initializer_range / math.sqrt(2 *n_layer)  # 层数相关的缩放
        )
    
    def forward(self,x):
        b,num_tokens,d_in = x.shape
        keys = self.W_k(x)
        queries = self.W_q(x)
        values = self.W_v(x)
        
        keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
        values = values.view(b,num_tokens,self.num_heads,self.head_dim)
        
        #(b,num_tokens,num_heads,head_dim) --> (b,num_heads,num_tokens,head_dim)   
        keys = keys.transpose(1,2)
        queries = queries .transpose(1,2)
        values = values.transpose(1,2)
        
        
        att_score = queries @ keys.transpose(2,3)
        att_score.masked_fill_(self.mask.bool()[:num_tokens,:num_tokens],-torch.inf)
        att_weight = torch.softmax(att_score/keys.shape[-1]**0.5, dim=-1)
        # att_weight = self.dropout(att_weight)
        context_vec = (att_weight @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        context_vec = self.c_proj(context_vec)
        context_vec = self.dropout(context_vec)
        return context_vec

# %% [markdown]
# ## Define Transformer block

# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg['emb_dim']) #norm1：用于注意力模块（self.att）的输入归一化
        self.att = MultiHeadAttendtion_new(
            d_in= cfg["emb_dim"],
            d_out= cfg['emb_dim'],
            context_len=  cfg['context_len'],
            num_heads= cfg["n_heads"],
            dropout= cfg["drop_rate"],
            initializer_range=cfg['initializer_range'],
            n_layer=cfg['n_layers'],
            qkv_bias=cfg["qkv_bias"],
            
        )
        self.norm2 = LayerNorm(cfg['emb_dim']) #norm2：用于前馈网络（self.ff）的输入归一化
        self.ff =FeedForward(cfg)
        self.dropout = nn.Dropout(cfg['drop_rate'])
        
    
    def forward(self,x):
        # 注意力分支：LayerNorm -> 注意力 -> Dropout -> 残差连接
        x = x + self.dropout(self.att(self.norm1(x))) 
        # FFN分支：LayerNorm -> FFN -> Dropout -> 残差连接
        x = x + self.dropout(self.ff(self.norm2(x)))  
        return x
        
         

# %% [markdown]
# 
# ## Tokenizer

# %%
# ! pip install tiktoken
import tiktoken

# %% [markdown]
# ### Tokenizer

# %%

def text_to_tokenIds(text,tokenizer):
    encoded = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor =torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokenIds_to_text(token_ids,tokenizer):
    flat =token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())




# %% [markdown]
# ### Tokenizer with padding
#  将文本转换为token ID张量，支持padding
#  自回归模型采用 “自左向右” 的生成方式，注意力机制只关注当前 token 左侧的内容，推荐右填充

# %%
def texts_to_tokenIds(text, tokenizer, max_length=None, padding_side="right"):
    """
        text: 输入文本（单个字符串或字符串列表）
        tokenizer: tiktoken编码器
        max_length: 最大序列长度，超过会截断，不足会填充
        padding_side: 填充方向（left/right）
    """
    if isinstance(text, str):
        text = [text]
    
    # 获取特殊标记ID
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
 
    pad_id = eos_id #GPT2没有专门的pad_id
    
    encoded_list = []
    for t in text:
        
        encoded = tokenizer.encode(t, allowed_special={'<|endoftext|>'})
        # 只在文本末尾加1个eos标记
        if encoded and encoded[-1] != eos_id:
            encoded.append(eos_id)
        
        # 截断过长序列（保留最后一个eos）
        if max_length and len(encoded) > max_length:
            encoded = encoded[:max_length-1] + [eos_id]  # 确保最后一个是eos
        
        encoded_list.append(encoded)
    
    # calculate maxlength
    max_len = max_length if max_length else max(len(seq) for seq in encoded_list)
    
    # padding
    padded_encoded = []
    for seq in encoded_list:
        pad_length = max_len - len(seq)
        if pad_length > 0:
            pad_tokens = [pad_id] * pad_length
            padded_seq = seq + pad_tokens if padding_side == "right" else pad_tokens + seq
        else:
            padded_seq = seq
        padded_encoded.append(padded_seq)
    
    return torch.tensor(padded_encoded)


def tokenIds_to_texts(token_ids, tokenizer):
    eos_id = tokenizer.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    
    if len(token_ids.shape) == 2:
        results = []
        for seq in token_ids:
            flat = seq.squeeze(0).tolist()
            flat_filtered = [id for id in flat if id != eos_id]
            results.append(tokenizer.decode(flat_filtered))
        return results
    else:
        # single text
        flat = token_ids.squeeze(0).tolist()
        flat_filtered = [id for id in flat if id != eos_id]
        return tokenizer.decode(flat_filtered)

# %% [markdown]
# ## Generate text
# 
# max_new_tokens: 往后生成n个token
# 
# context_size: 更关注最近的上下文，只取size数量的token

# %%

def  generate_text_greedy(model,idxs,max_new_tokens,context_size):
    model.eval()
    for _ in range(max_new_tokens):
        idx_condition = idxs[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_condition)
            
        #生成时：只需要最后一个位置的 logits
        logits = logits[:,-1,:]
        probas =torch.softmax(logits,dim=-1)
        idx_next = torch.argmax(probas,dim=-1,keepdim=True)
        idxs = torch.cat((idxs,idx_next),dim=1)
    return idxs

def  generate_text_withsample(model,idxs,max_new_tokens,context_size,
                              temperature=0.0,top_k=None,top_p=1,eos_id=None):
    model.eval()
    for _ in range(max_new_tokens):
        idx_condition = idxs[:,-context_size:]
        
        with torch.no_grad():
            logits = model(idx_condition)
        logits = logits[:,-1,:]
        # region top k
        if top_k is not None:
            top_logits,_ = torch.topk(logits,top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(logits<min_val,torch.tensor(float('-inf')).to(logits.device),logits)
        # endregion
        
        # region top p
        if top_p >0 and top_p<1: # top_p ==1 continue
            sorted_logits, sorted_idx =torch.sort(logits,dim = -1,descending = True)
            sorted_probs = torch.softmax(sorted_logits,dim = -1)
            cumulative_probs = torch.cumsum(sorted_probs,dim= -1)
            
            mask = cumulative_probs>top_p
            max_mask_idx = torch.argmax(mask.float(), dim=-1, keepdim=True)
            mask = mask.scatter(-1, max_mask_idx, False)
            sorted_logits[mask] = -torch.inf
            _, original_indices = torch.sort(sorted_idx, dim=-1)
            logits = torch.gather(sorted_logits, dim=-1, index=original_indices)
        elif top_p != 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        # endregion
        
        # region temperature
        if temperature > 0:
            probas =torch.softmax(logits/temperature,dim=-1)
            idx_next = torch.multinomial(probas,num_samples=1) # 高温度下，概率分布平缓，采样会更大概率选中次优选项
        else:
            probas =torch.softmax(logits,dim=-1)
            idx_next = torch.argmax(probas,dim=-1,keepdim=True)# greedy 
            
        if eos_id is not None:
            # 若batch中任何一个样本生成eos_id，终止该样本（这里简化为单样本处理）
            if torch.any(idx_next == eos_id):
                break  
        
        # endregion
        idxs = torch.cat((idxs,idx_next),dim=1)
    return idxs

# %% [markdown]
# ### test gernerate

# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_tokenizer():
    model =DummyGPT(TEST_CONFIG)
    #test_context ="今天的天气是晴天，适合出去走走"
    test_context = "I like the weather"
    print(f'{test_context}--ori')
    tokenizer =tiktoken.get_encoding(TOKEN_TYPE)
    tokenids =text_to_tokenIds(test_context,tokenizer)
    print(f'{tokenIds_to_text(tokenids,tokenizer)}--recover') 


    tokenids_g = generate_text_greedy(model,tokenids,max_new_tokens=10,context_size=TEST_CONFIG['context_len'])

    print(f'{tokenIds_to_text(tokenids_g,tokenizer)}--greedy') 
    
    tokenids_s = generate_text_withsample(model,tokenids,max_new_tokens=10,context_size=TEST_CONFIG['context_len'],
                                        temperature=0.5,top_k=50,top_p=1,eos_id=None)

    print(f'{tokenIds_to_text(tokenids_s,tokenizer)}--sample') 

test_tokenizer()


# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_tokenizer_padding(max_len=128):
    model =DummyGPT(TEST_CONFIG)
    # test_context ="今天的天气是晴天，适合出去走走"
    test_context = "I like the weather"
    print(f'{test_context}--ori')
    tokenizer =tiktoken.get_encoding(TOKEN_TYPE)
    tokenids =texts_to_tokenIds(test_context,tokenizer,max_length=max_len)
    print(f'{tokenIds_to_texts(tokenids[0],tokenizer)}--recover') 


    tokenids_g = generate_text_greedy(model,tokenids,max_new_tokens=10,context_size=TEST_CONFIG['context_len'])

    print(f'{tokenIds_to_texts(tokenids_g[0],tokenizer)}--greedy') 
    
    tokenids_s = generate_text_withsample(model,tokenids,max_new_tokens=10,context_size=TEST_CONFIG['context_len'],
                                        temperature=0.5,top_k=50,top_p=1,eos_id=None)

    print(f'{tokenIds_to_texts(tokenids_s[0],tokenizer)}--sample') 

test_tokenizer_padding()


# %% [markdown]
# Epoch 过程中查看生成的文本
# 
# 查看模型生成的新 token 数量（max_new_tokens）:
# * 训练监控（最常用）：20-50 个 token
# * 轻量化验证（追求效率）：10-20 个 token
# * 深度观察（关键节点）：50-100 个 token

# %%

def generate_and_print(model,tokenizer,device,start_context,max_new_tokens, temperature=0,top_k=None,top_p=1,eos_id=None):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    # print(context_size)
    encoded = texts_to_tokenIds(start_context,tokenizer=tokenizer,max_length=context_size).to(device)
    with torch.no_grad():
        token_ids = generate_text_withsample(model,idxs=encoded,max_new_tokens=max_new_tokens,context_size=context_size, temperature=temperature,top_k=top_k,top_p=top_p,eos_id=eos_id)
        decoded_text = tokenIds_to_texts(token_ids[0],tokenizer)
    print(decoded_text.replace("\n"," "))
 
        

# %% [markdown]
# ## GPTDataLoader
# 
# 预训练以长文本为主，注重上下文连续性，较少使用padding，以截断 + 滑动窗口为主

# %%

from torch.utils.data import DataLoader,Dataset

class GPTDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_len: int, stride: int):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        self.max_len = max_len
        self.stride = stride
        
        for idx, text in enumerate(tqdm(texts, desc="Process text")):
            
            if not isinstance(text, str):
                raise TypeError(f"The type of the {idx}-th element is {type(text)}")
            
            if not text.strip():
                continue
            
            # encode single text
            tokenids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
          
            token_len= len(tokenids)
            # print('token len:',token_len)
            
            if token_len < max_len + 1:
                continue  # 连一个完整样本都无法生成，直接跳过
            
            # 计算该文本可生成的样本数
            max_start = token_len - max_len - 1 # 最后一个有效起始位置
            num_samples = (max_start// stride) + 1 if max_start >= 0 else 0
            
            if num_samples > 0:
                # 滑动窗口生成样本
                for i in range(0, max_start, stride):
                    input_chunk = tokenids[i:i+max_len]
                    target_chunk = tokenids[i+1:i+max_len+1]  # 目标是输入的下一个token
                    if len(target_chunk) < max_len:
                            continue  # 跳过不完整的目标
                    self.input_ids.append(torch.tensor(input_chunk))
                    self.target_ids.append(torch.tensor(target_chunk))
        
        print(f"Total samples: {len(self.input_ids)}")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        try:
            input_batch = self.input_ids[idx]
            target_batch =self.target_ids[idx]
            return input_batch,target_batch
        except Exception as e:
            print(f'Failed to load{idx}:{str(e)}')
            raise
    

'''
DataLoader 本质是一个批次生成器迭代索引：
自动生成从 0 到 len(dataset)-1 的索引，通过 dataset.__getitem__(idx) 逐个获取样本
'''
def GPTDataloader(txts:list[str],token_type,batch_size=4,max_len=246,stride=128,shuffle=True,drop_last=True,num_works=0):
    tokenizer =tiktoken.get_encoding(token_type)
    ds = GPTDataset(txts,tokenizer,max_len,stride)
    dl = DataLoader(
        ds,
        batch_size =batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_works
    )
    return dl

# %% [markdown]
# ## Loss funcion

# %%
def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch) # 隐式调用 model.forward(input_batch)
    loss =torch.nn.functional.cross_entropy(logits.flatten(0,1),target_batch.flatten())
    return loss

#快速验证：指定 num_batchs=n，只跑前n个批次，节省时间。
def calc_loss_loader(data_loader,model,device,num_batchs=None):
    total_loss = 0
    total_batchs =len(data_loader)
    # print('total batch count:' ,total_batchs)
    if  total_batchs == 0:
        return float('nan')
    elif num_batchs is None:
        num_batchs = total_batchs 
    else:
        num_batchs = min(num_batchs,total_batchs)
    
    for i ,(input_batch,target_batch) in enumerate(data_loader):# dataset.__getitem__(idx)
        if i < num_batchs:
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batchs

def evaluate_model(model,train_loader,valid_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device,num_batchs=eval_iter)
        valid_loss = calc_loss_loader(valid_loader,model,device,num_batchs=eval_iter)
    model.train()
    return train_loss,valid_loss

# %% [markdown]
# ### test loss function

# %%
@tool.skip_execution(skip=IS_SKIP_TEST)
def test_loss():
    model = DummyGPT(TEST_CONFIG)
    model.to(device)
    file_path ="../datasets/the-verdict.txt"
    with open (file_path,"r",encoding="utf-8") as file:
        text_data =file.read()
        
    split_idx = int(0.8*len(text_data))
    train_data = text_data[:split_idx]
    print(len(train_data))
    
    train_loader = GPTDataloader(
        [train_data],
        TOKEN_TYPE,
        batch_size = TEST_CONFIG['batch_size'],
        max_len = TEST_CONFIG["context_len"],
        stride = TEST_CONFIG["context_len"] // 2, 
        drop_last=True,
        shuffle= True, 
        num_works=0   
    )

    return calc_loss_loader(train_loader,model,device=device)

test_loss()

# %% [markdown]
# ### Save model

# %%
def savemodel(path,model,optimizer,config):
    
    if False: # view model 
        for name, param in model.state_dict().items():
            print(f"{name}: {param.shape}")
            
    save_data = {"model_state_dict": model.state_dict()}
    
    if optimizer is not None:
        save_data["optimizer_state_dict"] = optimizer.state_dict()
    
    if config is not None:
        save_data["config"] = config
    try:
        torch.save(save_data, path)
    except Exception as e:
        raise IOError(f"save model fail: {e}") from e
    



# %%


def save_checkpoint(model, optimizer, epoch, global_step, train_losses, val_losses, track_tokens_seen, save_path):
    """保存训练状态用于后续恢复"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,  # 当前训练到的 epoch（下一次应从该 epoch 继续）
        'global_step': global_step,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'track_tokens_seen': track_tokens_seen,
        'torch_rng_state': torch.get_rng_state(),  # 保存随机数状态
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")
    
def load_checkpoint(model, optimizer, load_path):
    """加载训练状态，返回恢复后的训练进度信息"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint {load_path} not found")
    
    checkpoint = torch.load(load_path)
    
    # 恢复模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    # 恢复优化器参数（重要，确保学习率、动量等状态正确）
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # 恢复随机数状态（保证复现性）
    torch.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    # 返回训练进度信息
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'track_tokens_seen': checkpoint['track_tokens_seen']
    }


