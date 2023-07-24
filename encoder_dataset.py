import re
import torch
from transformers import PretrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm

class NERDataset(Dataset):
    """
    input 구조(txt)
    sent1\n
    sent2\n
    sent3\n
    ...
    """
    def __init__(
        self,
        file_path: str,
        tokenizer: PretrainedTokenizer,
        label2id: dict,
        max_length: int = 512,
    ) -> None:
        super(NERDataset, self).__init__()
        self.tokenizer = tokenizer
        self.datas = self._load_datas(file_path)
        self.label2id = label2id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        
        random.shuffle(self.datas)
        
    def _gen_labels(self, sequence, word2ids):
        labels = [0] * len(sequence)
        
        for v in word2ids.values():
            target_ids = v['target_ids']
            label_id = v['label_id']
            
            i=0
            target_ids_length = len(target_ids)
            
            while i < len(sequence):
                if sequence[i:i + target_ids_length] == target_ids:
                    for j in range(i, i + target_ids_length):
                        labels[j] = label_id
                    i = i + target_ids_length
                else:
                    i += 1
                    
        return labels
    
    def _add_padding(self, item):
        num_of_tensors_to_add = self.max_length - len(item['input_ids'])
        extended_input_ids = [self.pad_token_id] * num_of_tensors_to_add
        extended_attention_masks = [0] * num_of_tensors_to_add
        extended_labels = [-100] * num_of_tensors_to_add
        
        item['input_ids'].extend(extended_input_ids)
        item['attention_masks'].extend(extended_attention_masks)
        item['labels'].extend(extended_labels)
        
        return item
        
        
    def _load_datas(self, file_path):
        datas = list()
        
        with open(file_path, mode='r', encoding='utf-8') as reader:
            for line in tqdm(reader):
                text = line.strip()
                tagged_words = re.findall('<.*?:.*?>', text)
                
                # 태깅이 존재하는 텍스트에서 태깅된 토큰의 id와 라벨 id 추출하여 dict로 저장
                # 태깅이 존재하는 텍스트 -> 원본 텍스트
                word2ids = dict()
                for tagged_word in tagged_words:
                    word, tag = tagged_word.strip('<>').split(':')
                    if word not in word2ids:
                        word2ids[word] = {
                            'target_ids': self.tokenizer.encode(word)[1:-1],
                            'label_id': self.label2id[tag]
                        }
                        text = text.replace(tagged_word, word)
                    
                
                input_ids = self.tokenizer.encode(text, truncation=True, max_length=self.max_length)
                
                # 태깅된 토큰의 id와 라벨 id가 존재하는 dict를 활용하여 labels 생성
                labels = self._gen_labels(input_ids, word2ids)
                
                item = {
                    'input_ids': input_ids,
                    'attention_masks': [1] * len(input_ids),
                    'labels': labels,
                }
                
                # 패딩 추가
                item = self._add_padding(item)
                
                # 텐서로 변환
                for k in item.keys():
                    item[k] = torch.LongTensor(item[k])
                
                datas.append(item)
                
        return datas
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.datas[idx]['input_ids'],
            'attention_masks': self.datas[idx]['attention_masks'],
            'labels': self.datas[idx]['labels'],
        }
