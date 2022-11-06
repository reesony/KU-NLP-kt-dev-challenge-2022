import re

import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration, pipeline
from tqdm import tqdm
import argparse

from utils import T5NERDataset


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path')
    parser.add_argument('--test_file', default="/home/work/team02/kijun/data/test_onlyentity.csv")
    parser.add_argument('--batch_size', type=int, default="128")
    parser.add_argument('--num_beams', type=int, default="1")

    args = parser.parse_args()

    model_name = args.model_name_or_path
    test_file = args.test_file
    batch_size = args.batch_size
    num_beams = args.num_beams

    model = T5ForConditionalGeneration.from_pretrained(model_name).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    test_data = pd.read_csv(test_file)
    inputs = test_data["source"].tolist()
    labels = test_data["target"].tolist()
    predictions = []

    for batch in tqdm([inputs[i: i + batch_size] for i in range(0, len(inputs), batch_size)]):
        input_ids = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        ).input_ids.cuda()

        outputs = model.generate(
            input_ids,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            max_length=256
        )

        predictions += tokenizer.batch_decode(outputs, skip_special_tokens=True)

    ne_list, ne_counts = extract_ne_list(labels)
    pred_ne_list, _ = extract_ne_list(predictions)

    print(compute_f1_score(ne_list, ne_counts, pred_ne_list))
    