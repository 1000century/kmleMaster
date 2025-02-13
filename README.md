# kmleMaster

## 0. 데이터 수집
- 의사 국가고시, 임상종합의학평가 문제들
- 방식: PDF 전처리 후 OCR 작업 수행

## 1. 유사 문제 검색
- 유사도 검색 알고리즘: rank_bm25
- 토크나이저: bert-base-multilingual-cased
- 사용 범위: 각 문제의 'text' 부분으로 한정
- 출력 결과 설정: Top2
- https://github.com/1000century/kmleMaster/blob/main/model/similar.md

### 1-1. 토크나이저 분석결과
![image](https://github.com/user-attachments/assets/4848a6a4-655d-43b2-8a9f-3eb9cad603f5)

- 
