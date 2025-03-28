# 25.03.29
KAGGLE version 65
```
이번 실험의 의의:  처음으로 유의미한 FT한 결과가 나온 실험이다.
이전까지의 가장 큰 문제점은 출력 형식이었다. 파인튜닝 후에 출력형식에서 똑같은 말이 반복되어 나타나다 보니 실질적으로 쓸 수 없었다.
이번에는 프롬프트엔지니어링을 통해 출력 형식을 강제해보았더니 FT후에도 결과가 좋게 나타났다.
```
```
앞으로 보완해야 할 점
- 프롬프트를 한국어로 할지 영어로 할지
- 데이터 불균형 해결 방안
```


## 프롬프트의 변화
- 춣력 형식을 강제하였음
```python
def chat_cc_format_json(sample):
    return [
        {
            "role": "user",
            "content": f"""Can you classify the patient's main complaint into one of the following categories?

['구강병변', '두통', '목덩이', '삼킴곤란/삼킴통증', '시각증상', '객혈', '기침', '두근거림','유두분비물', '유방증상', '허리통증', '호흡곤란', '흉통/흉부불편감', '구토', '설사', '변비','복부덩이', '복부팽만', '복통', '소화불량', '토혈/흑색변/혈변', '단백뇨', '무월경/희발월경','배뇨증상', '소변색변화', '다뇨', '핍뇨', '월경통/월경과다', '질분비물', '질출혈', '관절증상','근력저하', '감각이상', '보행실조', '사지통증', '손떨림', '고혈압', '발달지연', '발작', '발열','부종', '비정상검사소견', '의식저하/실신']

Return your answer in the format: [classify_cc(text=...)]  
NO other text must be included.

Patient statement: \"{sample['문제']}\"\n\n"""
        },
        {
            "role": "assistant",
            "content": f"[classify_cc(text=\"{sample['chapter_title']}\")]"
        }
    ]
```

- 예시
```python
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Can you classify the patient's main complaint into one of the following categories?

['구강병변', '두통', '목덩이', '삼킴곤란/삼킴통증', '시각증상', '객혈', '기침', '두근거림','유두분비물', '유방증상', '허리통증', '호흡곤란', '흉통/흉부불편감', '구토', '설사', '변비','복부덩이', '복부팽만', '복통', '소화불량', '토혈/흑색변/혈변', '단백뇨', '무월경/희발월경','배뇨증상', '소변색변화', '다뇨', '핍뇨', '월경통/월경과다', '질분비물', '질출혈', '관절증상','근력저하', '감각이상', '보행실조', '사지통증', '손떨림', '고혈압', '발달지연', '발작', '발열','부종', '비정상검사소견', '의식저하/실신']

Return your answer in the format: [classify_cc(text=...)]  
NO other text must be included.

Patient statement: "55세 남자가 1주 전에 감기에 걸린 후 기침과 가래가 나오다가 2일 전부터 숨이 차서 병원에 왔다"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
[classify_cc(text="호흡곤란")]
```

## 토크나이징 방식 수정
- `apply_chat_template` 을 사용했을 떄, 내부적으로 `.strip()`을 사용해서 그런지 줄바꿈이 원하는 대로 나오지 않았다.
- 그래서 직접 구현하는 방식을 사용하였다.

```python
EOS_TOKEN = tokenizer.eos_token

tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize(element): # right padding
    messages = chat_cc_format_json(element)
    formatted = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + messages[0]["content"]
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        + messages[1]["content"]
    )
    
    formatted += EOS_TOKEN
    outputs = tokenizer(formatted, add_special_tokens=False)
    return {
        "input_ids": outputs['input_ids'],
        "attention_mask": outputs['attention_mask']
    }


def tokenize_for_test(element): # left_padding
    messages = chat_cc_format_json(element)
    formatted = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        + messages[0]["content"]
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    outputs = tokenizer(formatted, add_special_tokens=False)
    return {
        "input_ids": outputs['input_ids'],
        "attention_mask": outputs['attention_mask']
    }
tokenizer.padding_side = "right"
tokenized_train = dataset_splits['train'].map(tokenize)
tokenized_val = dataset_splits['validation'].map(tokenize)

tokenizer.padding_side = "left"
tokenized_test = dataset_splits['test'].map(tokenize_for_test)

tokenized_dataset_splits = DatasetDict({
    'train': tokenized_train,
    'validation': tokenized_val,
    'test': tokenized_test
})
```

## Result
### Before FT (Temperature 설정 모름)
정답 개수: 112/374
정답률: 29.95%


###  Train
| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 19   | 0.299100      | 0.324919        |
| 38   | 0.298800      | 0.186408        |
| 57   | 0.080300      | 0.089588        |
| 76   | 0.093300      | 0.079246        |

```global_step=93
training_loss=0.22870307419729488
metrics={'train_runtime': 3861.812,
  'train_samples_per_second': 0.773,
  'train_steps_per_second': 0.024, 
  'total_flos': 1.8958207197364224e+16, 
  'train_loss': 0.22870307419729488})
```

### After FT - temperature 0.7
정답 개수: 295/374
정답률: 78.88%

### After FT - temperature 0.1
정답 개수: 302/374
정답률: 80.75%
