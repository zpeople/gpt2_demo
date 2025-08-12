# %%
from datasets import load_dataset,load_from_disk

# ds = load_dataset("pleisto/wikipedia-cn-20230720-filtered")

# ds.save_to_disk("../datasets/wikipedia") 

# %%
def load_local_data():
    dataset = load_from_disk("../datasets/wikipedia")

    print(dataset.keys())

    # 如果是多拆分数据集（DatasetDict类型，如包含train/test）
    train_df = dataset["train"].to_pandas()
    train_df.head()
    completion_list = train_df["completion"].tolist()
    return completion_list
