#%%
import torch
from torch.utils.data import Dataset, DataLoader

# 简单的 GPT dataset 类
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # A 将整个文本进行分词

        for i in range(0, len(token_ids) - max_length, stride):  # B 使用滑动窗口将书籍分块为最大长度的重叠序列。
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):  # C 返回数据集的总行数
        return len(self.input_ids)

    def __getitem__(self, idx):  # D 从数据集中返回指定行
        return self.input_ids[idx], self.target_ids[idx]


import tiktoken


# Listing 2.6 A data loader to generate batches with input-with pairs
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")  # A 初始化分词器。简单起见，没有去实现它，而是直接使用了 tiktoken 库

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # B 创建GPTDatasetV1类

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  # C drop_last=True会在最后一批次小于指定的 batch_size 时丢弃该批次，以防止训练期间的损失峰值
        num_workers=0  # D 用于预处理的 CPU 进程数量
    )

    return dataloader


#%% Listing 2.7 Load the book and create a data loader
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(dataloader)  # A 将数据加载器转换为 Python 迭代器，以便通过 Python 的内置 next() 函数获取下一个数据条目。

first_batch = next(data_iter)
print(first_batch)

