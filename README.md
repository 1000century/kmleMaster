# kmleMaster

## 0. 데이터 수집
- 의사 국가고시, 임상종합의학평가 문제들
- 방식: PDF 전처리 후 OCR 작업 수행

## 1. 유사 문제 검색
- rank_bm25 사용
- bert-base-multilingual-cased 토크나이저 사용
- 상위 2개 사용 '문제' 부분만 사용
- https://github.com/1000century/kmleMaster/blob/main/model/similar.md
