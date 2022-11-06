import re
import pandas as pd
from transformers import T5Tokenizer
import torch
import numpy as np

def _gen_entity_dict(data_list):
    pattern = r'<.*?:.*?>'
    entities = []
    for data in data_list:
        items = re.findall(pattern, data)

        for item in items:
            entities.append(item.split(":")[1].rstrip(">"))
    
    entity_dict = {}
    entity_num = 1
    
    for entity in entities:
        if entity not in entity_dict:
            entity_dict[entity] = entity_num
            entity_num += 1

    return entity_dict

def _check_entity(data):
    pattern = r'<.*?:.*?>'
    entities = []
    
    items = re.findall(pattern, data)

    for item in items:
        entities.append(item.split(":")[1].rstrip(">"))
    
    return entities

def _check_arrow(l_arrow_index, quot_index, r_arrow_index):
    idx = 0
    end_idx = min(len(l_arrow_index), len(quot_index), len(r_arrow_index))
    
    while True:
        end_idx = min(len(l_arrow_index), len(quot_index), len(r_arrow_index))
        if idx >= end_idx: break
        
        if int(quot_index[idx]) > int(r_arrow_index[idx]):
            l_arrow_index = np.delete(l_arrow_index,idx)
            r_arrow_index = np.delete(r_arrow_index,idx)
            
        elif int(quot_index[idx]) < int(l_arrow_index[idx]):
            quot_index = np.delete(quot_index,idx)
            
        else:
            idx += 1
        
    l_arrow_index = l_arrow_index[:end_idx]
    quot_index = quot_index[:end_idx]
    r_arrow_index = r_arrow_index[:end_idx]
        
    return l_arrow_index, quot_index, r_arrow_index

def preprocessor(file_path, model_name, max_len = 256):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    remove_token_text = '<:>'
    remove_token_tensor = tokenizer(remove_token_text, return_tensors='pt')
    remove_num_list = remove_token_tensor['input_ids'].tolist()
    
    l_arrow_num, quot_num, r_arrow_num, eos_num = remove_num_list[0][:]
    
    f = open(file_path, 'r')
    dataset = []
    while True:
        line = f.readline()
        if not line: break
        dataset.append(line)
    f.close()
    
    entity_dict = _gen_entity_dict(dataset)
    
    preprocessed_data = []
    error_index = []
    error_flag = False
    
    for index in range(len(dataset)):
        row = []
        label_ids = []
        pattern = r'[^\s]<.*?:.*?>'
        items = re.findall(pattern, dataset[index])
        
        for text in items:
            post_text = text[0] + ' ' + text[1:]
            dataset[index] = dataset[index].replace(text,post_text)
        
        
        pre_data = tokenizer(dataset[index], truncation=True, padding='max_length', max_length=max_len, return_tensors='pt')
        input_ids = pre_data['input_ids']
        
        l_arrow_index = ((input_ids == l_arrow_num).nonzero(as_tuple=True)[1]).numpy()
        quot_index = ((input_ids == quot_num).nonzero(as_tuple=True)[1]).numpy()
        r_arrow_index = ((input_ids == r_arrow_num).nonzero(as_tuple=True)[1]).numpy()
        entities = _check_entity(dataset[index])
        
        l_arrow_index, quot_index, r_arrow_index = _check_arrow(l_arrow_index, quot_index, r_arrow_index)
        
        pre_data['label_ids'] = torch.zeros(input_ids.shape, dtype=torch.int64)
        
        
        for idx in range(len(l_arrow_index)):
            try:
                entity_len = r_arrow_index[idx] - l_arrow_index[idx] + 1
                entity_num = entity_dict[entities[idx]]
                entity_mask = torch.ones([entity_len, ], dtype=torch.int64) * entity_num
                start_idx = l_arrow_index[idx]

                pre_data['label_ids'][:,start_idx:start_idx + entity_len] = entity_mask
                
            except Exception as e:
                error_index.append(index)
                error_flag =True
                print(e)
                break
        
        if error_flag: 
            error_flag = False
            continue
        
        for idx in range(len(l_arrow_index)):
            for key in pre_data.keys():
                pre_data[key] = torch.cat((pre_data[key][:,:int(l_arrow_index[idx])],
                                              pre_data[key][:,int(l_arrow_index[idx])+1:int(quot_index[idx])],
                                             pre_data[key][:,int(r_arrow_index[idx])+1:]), axis=1)
                
            disc_num = int(l_arrow_index[idx]) + 1 - int(l_arrow_index[idx]) + int(r_arrow_index[idx]) + 1 - int(quot_index[idx])
            if idx + 1 != len(l_arrow_index):
                l_arrow_index[idx + 1:] -= disc_num
                quot_index[idx + 1:] -= disc_num
                r_arrow_index[idx + 1:] -= disc_num

        try:
            input_ids = pre_data['input_ids']
            for idx in range(len(l_arrow_index)):
                for key in pre_data.keys():
                    pre_data[key] = torch.cat((pre_data[key][:,:],
                                              torch.zeros([1,max_len - pre_data[key].shape[1]], dtype=torch.int64)),axis=1)

            eos_index = (input_ids == eos_num).nonzero(as_tuple=True)[1]
            label_mask_len = max_len - int(eos_index)
            label_mask = torch.ones([label_mask_len, ], dtype=torch.int64) * (-100)
            
            pre_data['label_ids'][:,eos_index:] = label_mask
        except Exception as e:
            error_index.append(index)
            error_flag =True
            print(e)
            
        if not error_flag:
            preprocessed_data.append(pre_data)
        
        error_flag = False
    
    print(error_index)
    
    return dataset, preprocessed_data, entity_dict
