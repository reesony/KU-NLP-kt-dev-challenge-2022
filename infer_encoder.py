import os
from itertools import product

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import KFold
import re
from preprocessor import preprocessor
import argparse

from transformers import (
    AutoConfig,
    DefaultDataCollator,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)

def extract_ne_list(labels):
    ne_list = []
    ne_counts = []

    pattern = r'<.*?:.*?>'

    for row in labels:
        items = re.findall(pattern, row)

        ne_items = [item.strip("<>").split(":") for item in items]

        ne_counts.append(len(items))
        ne_list.append(ne_items)

    return ne_list, ne_counts


def compute_f1_score(labels, counts, predictions):
    precision_all, recall_all, f1_all = 0, 0, 0

    for index in range(len(labels)):
        ne_item = labels[index]
        ne_count = counts[index]
        pred_item = predictions[index]

        tp, fp = 0, 0
        
        
        for ne in ne_item:
            if ne in pred_item:
                tp += 1
            else:
                for pred_ne in pred_item:
                    pred_ne_str = pred_ne[0]
                    pred_ne_type = pred_ne[1]
                    if pred_ne_str in ne[0] and pred_ne_type == ne[1]:
                        tp += 1
                        break
                    elif ne[0] in pred_ne_str and pred_ne_type == ne[1]:
                        tp += 1
                        break

        for pred in pred_item:
            if pred not in ne_item:
                fp += 1

        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if ne_count == 0 else tp / (ne_count)
        f1 = 0 if precision == 0 and recall == 0 else 2 / (1 / precision + 1 / recall)

        precision_all += precision
        recall_all += recall
        f1_all += f1

    precision_all /= len(labels)
    recall_all /= len(labels)
    f1_all /= len(labels)

    return {
        "precision": precision_all,
        "recall": recall_all,
        "f1": f1_all
    }

if(__name__=="__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name_or_path', default='/home/work/team02/model/encoder_only_models')
    parser.add_argument('--test_file', default="/home/work/team02/kijun/data/test_onlyentity.csv")
    parser.add_argument('--test_model')
    parser.add_argument('--data_path')
    
    args = parser.parse_args()
    
    ent_dict = {
        0 : 'O',
        1 : 'QT',
        2 : 'DT',
        3 : 'PS',
        4 : 'LC',
        5 : 'TI',
        6 : 'OG'
    }
    
    model_path = args.model_name_or_path
    data_path = args.data_path
    test_file = args.test_file
    test_model_path = args.test_model
    
    
    dataset, data, ent = preprocessor(data_path, model_path)
    
    df = pd.DataFrame(columns=['input_ids', 'attention_mask'])
    
    for i,d in enumerate(data):
        df.loc[i] = [d['input_ids'].view(-1).tolist(), d['attention_mask'].view(-1).tolist()]
        
    test_input = Dataset.from_pandas(df)
    
    data_collator = DefaultDataCollator()
    
    model = AutomodelForTokenClassification.from_pretrained(test_model_path)
    tokenzier = T5Tokenizer.from_pretrained(model_path)
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator
    )
    
    prediction = trainer.predict(test_input)[0]
    
    prediction = np.argmax(prediction, axis=-1)
    
    result = []
    temp_result = []
    
    for i in range(len(df['input_ids'])):
        temp = []

        for j in range(len(prediction[i])):
            if df['attention_mask'][i][j] == 0:
                break
            if prediction[i][j] != 0:
                temp.append(df['input_ids'][i][j])
                if (j+1) == 256 or prediction[i][j] != prediction[i][j+1]:
                    output = tokenzier.decode(temp)
                    output = '<' + output + ':' + ent_dict[prediction[i][j]] + '>'
                    temp_result.append(output)
                    temp = []

        result.append(temp_result)
        temp_result = []
        
    predictions = []
    for row in result:
        row_str = ''
        for entity in row:
            row_str += entity

        predictions.append(row_str)


    test_data = pd.read_csv(test_file)
    inputs = test_data["source"].tolist()
    labels = test_data["target"].tolist()

    ne_list, ne_counts = extract_ne_list(labels)
    pred_ne_list, _ = extract_ne_list(predictions)
    
    pred_data = pd.DataFrame(columns=['source', 'target'])
    pred_data['source'] = inputs
    pred_data['target'] = predictions
    
    save_dir = './encoder_model_pred'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_name = test_model_path.split('/')[-2]
    pred_data.to_csv(save_dir + '/' + file_name + '.csv', index=False)

    print(compute_f1_score(ne_list, ne_counts, pred_ne_list))
