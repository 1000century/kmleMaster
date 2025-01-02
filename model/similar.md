# 방식
> rank_bm25 사용  
> bert-base-multilingual-cased 토크나이저 사용  
> 상위 2개 사용
> '문제' 부분만 사용

```python
from transformers import AutoTokenizer
from rank_bm25 import BM25Okapi

class MultilingualBM25Retriever:
    def __init__(self, documents, document_ids, model_name='bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.documents = documents
        self.document_ids = document_ids
        self.tokenized_documents = [self.tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_documents)
    
    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens
    
    def search(self, query, current_id=None, top_k=2):
        tokenized_query = self.tokenize(query)
        print("Tokenized query:", tokenized_query)
        scores = self.bm25.get_scores(tokenized_query)

        filtered_results = [
            (i, scores[i]) for i in range(len(scores)) if self.document_ids[i] != current_id
        ]
        top_n = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.document_ids[i], self.documents[i], score) for i, score in top_n]
```
---

| **원본 문제**                                                                                                                                                                      | **찾은 문제 1**                                                                                                                                                              | **찾은 문제 2**                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **문제** 33세 여자가 1개월 전부터 목이 붓고 열이 나서 병원에 왔다. 손이 떨리고 두근거림이 있다고 한다. 1개월간 체중이 2 kg 빠졌다고 한다. 키 162 cm, 체중 46 kg이다. 혈압 110/80 mmHg, 맥박 120회/분, 체온 37.5 ℃이다. 갑상샘이 전체적으로 커져 있고 심한 압통이 있다. 검사 결과는 다음과 같다. 진단은? <br> **검사 결과**: <br> 혈액: 유리 T4 2.46 ng/dL (참고치, 0.8-1.7) <br> 갑상샘자극호르몬 ＜0.01 mIU/L (참고치, 0.34-4.25) <br> 항갑상샘자극호르몬수용체항체 0.9 mIU/L (참고치, ≤1.75) <br> 항갑상샘과산화효소항체 13.2 IU/L (참고치, ＜35) <br> 24 시간 방사성요오드 섭취율: 0.9 % (참고치, 5-20) <br> **선지**: <br> ① 하시모토병 <br> ② 그레이브스병 <br> ③ 급성갑상샘염 <br> ④ 단순갑상샘종 <br> ⑤ 아급성갑상샘염 | **문제** 80세 여자가 6개월 전부터 목이 부어서 병원에 왔다. 체중이 1개월 동안 3 kg 빠졌고, 손떨림과 두근거림이 있다고 한다. 키 153 cm, 체중 54 kg이다. 혈압 144/68 mmHg, 맥박 110회/분, 호흡 20회/분, 체온 36.8 ℃이다.  진단은?  <br> 키 153 cm, 체중 54 kg. 혈압 144/68 mmHg, 맥박 110회/분, 체온 36.8 ℃. <br> **검사 결과**: <br> 백혈구 8,800/mm3, 혈색소 12.1 g/dL, 혈소판 <br> 224,000/mm3 적혈구침강속도 11 mm/시간 (참고치, ＜ 20)C 반응단백질 0.3 mg/L (참고치, ＜10)갑상샘자극호르몬 0.008 <br> mIU/L (참고치, 0.34-4.25)유리 T4 2.10 ng/dL (참고치 <br> 0.8-1.7)갑상샘자극호르몬수용체항체 0.1 mIU/L (참고치, ≤1.75)  <br> **선지**: <br> ① 하시모토병 <br> ② 그레이브스병 <br> ③ 급성갑상샘염 <br> ④ 독성갑상샘종 <br> ⑤ 아급성갑상샘염 | **문제** 33세 여자가 2~3주 전부터 가슴이 두근거리고 목이 부어서 병원에 왔다. 체중이 1개월 동안 3 kg 줄었다고 한다. 9개월 전 분만했으며, 모유 수유 중이다. 키 162 cm, 몸무게 47 kg이다. 혈압 135/67 mmHg, 맥박 118회/분, 호흡 16회/분, 체온 36.5 ℃이다. 갑상샘은 전반적으로 커져 있으나, 압통은 없다. 치료는? <br> **검사 결과**: <br> T3 320 ng/dL (참고치, 77-135), <br> 유리 T4 3.8 ng/dL (참고치, 0.8-1.7), 갑상샘자극호르몬 ＜0.01 mIU/L (참고치, 0.34-4.25) <br> 항갑상선과산화효소항체 양성 <br> 항갑상선자극호르몬 수용체 항체 양성 <br> **선지**: <br> ① 경과관찰 <br> ② 메티마졸 <br> ③ 레보티록신 <br> ④ 프레드니솔론 <br> ⑤ 방사성요오드 |

