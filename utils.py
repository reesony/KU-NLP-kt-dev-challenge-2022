import os
import random

import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class T5NERDataset(Dataset):
    def __init__(self, file, tokenizer, max_len=256):
        super().__init__()
        self.data = pd.read_csv(file).dropna()
        self.data = self.data.sample(frac=1)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.len = self.data.shape[0]

        self.pad_index = self.tokenizer.pad_token_id

    def __len__(self):
        return self.len

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        tokens = self.tokenizer(instance['source'])
        input_ids = tokens['input_ids']
        input_ids = self.add_padding_data(input_ids)
        attention_mask = tokens['attention_mask']
        attention_mask = self.add_padding_data(attention_mask)

        label_ids = self.tokenizer.encode(instance['target'])
        label_ids = self.add_padding_data(label_ids)

        return {
            'input_ids': np.array(input_ids, dtype=np.int_),
            'attention_mask': np.array(attention_mask, dtype=np.int_),
            'labels': np.array(label_ids, dtype=np.int_),
        }


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)
