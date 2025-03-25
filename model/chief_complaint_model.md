## LLaMA Tokenizer 관련
학습할 때는 right padding, 추론할 때는 left padding 해줘야 함.

```python
from sklearn.model_selection import train_test_split
from datasets import DatasetDict

EOS_TOKEN = tokenizer.eos_token

# 기본 토크나이즈 함수 (right padding)
def tokenize(element):
    formatted = tokenizer.apply_chat_template(
        chat_cc_format(element), tokenize=False
    )
    formatted += EOS_TOKEN
    outputs = tokenizer(formatted)
    return {
        "input_ids": outputs['input_ids'],
        "attention_mask": outputs['attention_mask']
    }


def tokenize_for_test(element):
    formatted = tokenizer.apply_chat_template(
        chat_cc_format(element), tokenize=False
    )
    formatted += EOS_TOKEN
    outputs = tokenizer(formatted)
    return {
        "input_ids": outputs['input_ids'],
        "attention_mask": outputs['attention_mask']
    }

from datasets import DatasetDict

# 1. 먼저 원본 데이터셋을 split
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
test_valid_split = dataset_split['test'].train_test_split(test_size=0.5, seed=42)

dataset_splits = DatasetDict({
    'train': dataset_split['train'],
    'validation': test_valid_split['train'],
    'test': test_valid_split['test']
})

# 2. padding 설정: train/val은 right, test는 left
tokenizer.padding_side = "right"
tokenized_train = dataset_splits['train'].map(tokenize)
tokenized_val = dataset_splits['validation'].map(tokenize)

tokenizer.padding_side = "left"
tokenized_test = dataset_splits['test'].map(tokenize_for_test)

# 3. 다시 모으기
from datasets import DatasetDict

tokenized_dataset_splits = DatasetDict({
    'train': tokenized_train,
    'validation': tokenized_val,
    'test': tokenized_test
})

```
```python
import torch
from torch.utils.data import DataLoader
tokenizer.padding_side = "left"

def find_subsequence(full_list, sub_list):
    for i in range(len(full_list) - len(sub_list) + 1):
        if full_list[i:i+len(sub_list)] == sub_list:
            return i+7
    return -1  # 못 찾았을 때


def extract_prompt(example):
    input_ids = example["input_ids"]
    cut_idx = find_subsequence(input_ids, response_template_ids)
    return {"input_ids": input_ids[:cut_idx]}  # response 이전까지 자름

# 테스트 데이터에서 response 부분 제외한 프롬프트만 준비
test_prompts = tokenized_dataset_splits["test"].map(extract_prompt)

def collate_fn(batch_samples):
    input_ids_list = [torch.tensor(sample["input_ids"]) for sample in batch_samples]
    max_len = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    for ids in input_ids_list:
        pad_len = max_len - len(ids)
        # ⬅️ 왼쪽 padding: [PAD, PAD, ..., token, token]
        padded = torch.cat([torch.full((pad_len,), tokenizer.pad_token_id), ids])
        padded_input_ids.append(padded)
    
    input_ids = torch.stack(padded_input_ids)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# DataLoader를 사용하여 배치 처리
batch_size = 3  # 적절한 배치 크기 설정
test_dataloader = DataLoader(test_prompts, batch_size=batch_size, collate_fn=collate_fn)
```

매우 복잡하게 코드를 짰지만 간단하게 하는 방법이 있을 거 같다.
- 먼저 train, valid, test 전부를 right padding 하고 test만 나중에 left padding 하는 방식으로 했다.
- 그리고 test_dataloader 할 때 batch 단위로 아무 설정 없으면 자동으로 right padding이 되므로 이 때도 collate_fn을 수정하는 방식을 이용했다.
