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
    




import fire 
import json
from src import model_wrapper,models
import torch
import tiktoken

# python -m interactive.interactive_generate  gpt2.pt gpt2 --max_new_tokens 100 --temperature 0.7 --top_k 50  --eos_id 50256 --models_dir "./model/" 



def interact_model(
   
    modelname,
    token_type,
    max_new_tokens,
    temperature=1,
    top_k=None,
    top_p=1,
    eos_id='',
    models_dir='../model/',
):   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(models_dir+modelname, map_location=device)
    config = checkpoint["config"]
    model = models.GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval() 
    context_size = model.pos_emb.weight.shape[0]
    tokenizer = tiktoken.get_encoding(token_type)
    with torch.no_grad():
        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            encoded = model_wrapper.text_to_tokenIds(raw_text,tokenizer).to(device)
            token_ids = model_wrapper.generate_text_withsample(model,idxs=encoded,max_new_tokens=max_new_tokens,context_size=context_size, temperature=temperature,top_k=top_k,top_p=top_p,eos_id=eos_id)
            decoded_text = model_wrapper.tokenIds_to_text(token_ids,tokenizer)
            print(decoded_text.replace("\n"," "))
            
if __name__ == '__main__':
    fire.Fire(interact_model)

