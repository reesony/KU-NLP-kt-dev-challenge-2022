import os
from itertools import combinations
import re
import pandas as pd
from tqdm import tqdm

def extract_ne_list(labels):
    ne_list = []
    ne_counts = []

    pattern = r'<.*?:.*?>'

    for row in labels:
        if isinstance(row, str):
            items = re.findall(pattern, row)

            ne_items = [item.strip("<>").split(":") for item in items]

            ne_counts.append(len(items))
            ne_list.append(ne_items)
        else:
            ne_counts.append(0)
            ne_list.append([])

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

def gen_ensemble(ensemble_num):
    RESULT_LIST_DIR = './ensemble'
    
    result_list = os.listdir(RESULT_LIST_DIR)
    if '.ipynb_checkpoints' in result_list:
        result_list.remove('.ipynb_checkpoints')
    combination_result_list = list(combinations(result_list, ensemble_num))
    result_name_list = []
    
    for results in combination_result_list: 
        result_list = []
        for result in results:
            path = os.path.join(RESULT_LIST_DIR, result)
            result_list.append(path)
        result_name_list.append(result_list)
            
    return result_name_list

def voted_pred(predictions, label):
    max_str_len = 0
    voted_pred_index = 0
    pred_dict = {}
    extraction_flag = False
    voted_pred_line = ''
    
    for pred in predictions:
        str_pred = str(pred)
        if str_pred not in pred_dict.keys():
            pred_dict[str_pred] = 1
        else:
            pred_dict[str_pred] += 1
        
    voted_std_dict = sorted(pred_dict.items(), key = lambda item: item[1], reverse=True)
    if voted_std_dict[0][1] == 5:
        voted_pred_line = voted_std_dict[0][0]
        return voted_pred_line
    
    if voted_std_dict[0][1] != voted_std_dict[1][1]:
        voted_pred_line = voted_std_dict[0][0]
        # extraction_flag = True
    else:
        if voted_std_dict[0][1] == 2:
            max_str_len = len(voted_std_dict[0][0])
            if max_str_len >= len(voted_std_dict[1][0]):
                voted_pred_line = voted_std_dict[0][0]
            else:
                voted_pred_line = voted_std_dict[1][0]
            # extraction_flag = True
        else:
            for data in voted_std_dict:
                str_len = len(data[0])
                
                if max_str_len < str_len:
                    max_str_len = str_len
                    voted_pred_line = data[0]
                    # extraction_flag = True
                else:
                    continue

    return voted_pred_line

def voted_pred_by_f1(pred_items, label_items):
    max_f1 = 0
    voted_pred_index = 0
    ne_counts = [len(label_items)]
    label_items = [label_items]
    for i in range(len(pred_items)):
        pred_item = [pred_items[i]]
        f1 = compute_f1_score(label_items , ne_counts, pred_item)['f1']
        if max_f1 < f1:
            max_f1 = f1
            voted_pred_index = i
    
    return voted_pred_index

def optimal_pred(predictions,labels):
    opt_predictions = []
    
    for i in range(len(predictions[0])):

        refined_predictions = []
        for pred in predictions:
            pred_row = pred[i]
            refined_predictions.append(pred_row)
        voted_pred_line = voted_pred(refined_predictions, labels[i])
        opt_predictions.append(voted_pred_line)
            
    return opt_predictions
    
def gen_pred(result_name_list):
    
    test_file = 'test.csv'
    predictions = []
    
    for result in result_name_list:
        
        pred_data = pd.read_csv(result)
        preds = pred_data['target'].tolist()
        predictions.append(preds)
        
    test_data = pd.read_csv(test_file)
    labels = test_data['target'].tolist()
        
    predictions = optimal_pred(predictions,labels)
    
    return predictions, labels
    
   
if __name__ == "__main__":
    ensemble_num = [5]
    f = open('./voted_ensemble_result.txt','w')
    max_f1_score = 0
    for num in ensemble_num:
        result_name_list = gen_ensemble(num)
        
        for result_list in tqdm(result_name_list):
            predictions, labels = gen_pred(result_list)

            ne_list, ne_counts = extract_ne_list(labels)
            pred_ne_list, _ = extract_ne_list(predictions)

            f1_score = compute_f1_score(ne_list, ne_counts, pred_ne_list)['f1']
            
            result_text = ''
            for result in result_list:
                result_text += os.path.basename(result).replace('.csv','') + '\n'

            result_text += str(f1_score) + '\n\n'
            
            if max_f1_score < f1_score:
                max_f1_score = f1_score
            elif max_f1_score >= f1_score:
                continue
            
            f.write(result_text)
    
    f.close()
