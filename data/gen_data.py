import re
import random
import os

import numpy as np
import pandas as pd

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)

def generate_data(data):
    gen_data = []
    answer = []

    pattern = r'<.*?:.*?>'
    for row in data:
        items = re.findall(pattern, row)

        new_row = row
        for item in items:
            new_row = new_row.replace(item, item.split(":")[0].lstrip("<"))
        
        gen_data.append("ner: " + new_row)
        answer.append("".join(items))

    return gen_data, answer


if __name__ == "__main__":
    seed_everything(981029)

    with open("./data/klue_ner_train_80.t", "r") as f:
        train_data = f.read().splitlines()

    with open("./data/klue_ner_test_20.t", "r") as f:
        test_data = f.read().splitlines()

    df = pd.DataFrame(columns=["source", "target"])
    df['source'], df['target'] = generate_data(train_data)

    df = df.sample(frac=1).reset_index(drop=True)

    split_point = int(len(df) * 0.8)

    train_df = df[:split_point]
    val_df = df[split_point:]
    
    test_df = pd.DataFrame(columns=["source", "target"])
    test_df['source'], test_df['target'] = generate_data(test_data)

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)
