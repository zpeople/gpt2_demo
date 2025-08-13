# %%
from datasets import load_dataset,load_from_disk

# ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered")

# ds.save_to_disk("../datasets/wikipedia") 

# %%
def load_local_data():
    dataset = load_from_disk("../datasets/wikipedia")

    # 如果是多拆分数据集（DatasetDict类型，如包含train/test）
    print(dataset.keys())

    train_df = dataset["train"].to_pandas()
    print( train_df.head(5))

    completion_list = train_df["completion"].tolist()
    print(len(completion_list))
    
    return completion_list

# %%
load_local_data()
print('load data')

# %% [markdown]
# ### load  txt of en

# %%
# en txt for debug

def load_data_en(file_path,train_ratio=0.8):
    with open (file_path,"r",encoding="utf-8") as file:
        text_data =file.read()
        
    print(f'total char: {len(text_data)}') #character count
    
    split_idx = int(train_ratio*len(text_data))
    train_data = text_data[:split_idx]
    valid_data = text_data[split_idx:]
    print(f'train char: {len(train_data)}\nvalid char: {len(valid_data)} \n')
    return [train_data], [valid_data]

# %% [markdown]
# ### load  txt of cn

# %%
def load_data_cn(part=False,train_ratio=0.8):
    txts = load_local_data()
    if part:
        txts =txts[:10]
    len(txts)
    split_idx = int(train_ratio*len(txts))
    train_data = txts[:split_idx]
    valid_data = txts[split_idx:]
    print(f'train sentence: {len(train_data)}\nvalid sentence: {len(valid_data)} \n')
    return train_data,valid_data


